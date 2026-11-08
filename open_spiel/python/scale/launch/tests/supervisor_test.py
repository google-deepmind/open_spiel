# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests of launch.supervisor over a scripted fake Backend.

launch.supervisor over a scripted fake Backend: the attempt loop's
outcomes, bounds, idempotent run identity and state in the run's storage.
"""

import json
import sys

import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import supervisor as supervisor_lib


class FakeBackend(backend_lib.Backend):
  """A scripted fake Backend.

  Attempt n polls through script[n] (a list of Status, or a callable
  (handle) -> Status); provision raises NoCapacityError for the first
  `stockouts`
  calls.

  Attributes:
    script: per attempt, its remaining poll Statuses.
    stockouts: provision calls left that raise NoCapacityError.
    adoptable: whether adopt re-attaches.
    started_t: the attempt start time recorded in handles, or None.
    files: the in-memory storage, uri -> text.
    calls: every backend call made, in order.
  """

  name = "fake"

  def __init__(self, script, stockouts=0, adoptable=False, started_t=None):
    self.script, self.stockouts, self.adoptable = (
        [list(s) for s in script],
        stockouts,
        adoptable,
    )
    self.started_t = started_t
    self.files, self.calls = {}, []

  def run_uri(self, run_id):
    """The run's in-memory root.

    Args:
      run_id: the run.

    Returns:
      mem://runs/<run_id>.
    """
    return f"mem://runs/{run_id}"

  def read_text(self, uri):
    """The in-memory object's text, or None.

    Args:
      uri: the object.

    Returns:
      its text, or None.
    """
    return self.files.get(uri)

  def exists(self, uri):
    """Whether the in-memory object exists.

    Args:
      uri: the object.

    Returns:
      True when it exists.
    """
    return uri in self.files

  def write_text(self, uri, text):
    """Stores the in-memory object's text.

    Args:
      uri: the object.
      text: its text.
    """
    self.files[uri] = text

  def prepare(self, run):
    """Records the prepare call.

    Args:
      run: the run.
    """
    self.calls.append(("prepare", run.run_id))

  def provision(self, run, attempt):
    """A handle, after `stockouts` NoCapacityError calls.

    Args:
      run: the run.
      attempt: the attempt number.

    Returns:
      the handle.

    Raises:
      NoCapacityError: the first `stockouts` calls.
    """
    del run  # Unused.
    self.calls.append(("provision", attempt))
    if self.stockouts:
      self.stockouts -= 1
      raise backend_lib.NoCapacityError(
          [{"candidate": "m1", "zone": "z1", "error": "exhausted"}]
      )
    h = {"backend": self.name, "attempt": attempt}
    if self.started_t is not None:
      h["started_t"] = self.started_t(attempt)
    return h

  def launch(self, handle, run, attempt):
    """Records the launch.

    Args:
      handle: the attempt's handle.
      run: the run.
      attempt: the attempt number.

    Returns:
      the handle, marked launched.
    """
    del run  # Unused.
    self.calls.append(("launch", attempt))
    return dict(handle, launched=True)

  def poll(self, handle):
    """The attempt's next scripted Status.

    Args:
      handle: the attempt's handle.

    Returns:
      the Status.
    """
    st = self.script[handle["attempt"]].pop(0)
    return st(handle) if callable(st) else st

  def abort(self, handle, exc):
    """Records the abort.

    Args:
      handle: the attempt's handle.
      exc: the exception that aborted it.
    """
    self.calls.append(("abort", handle["attempt"], type(exc).__name__))

  def teardown(self, handle):
    """Records the teardown.

    Args:
      handle: the attempt's handle.
    """
    self.calls.append(("teardown", handle["attempt"]))

  def adopt(self, handle):
    """The handle when adoptable, else None.

    Args:
      handle: the previous attempt's handle.

    Returns:
      the handle, or None.
    """
    self.calls.append(("adopt", handle["attempt"]))
    return dict(handle, adopted=True) if self.adoptable else None


RUN = backend_lib.RunSpec(
    "kuhn-a", train_args=("--config=mmd_kuhn_poker_jax", "--seed=0")
)
R, OK = backend_lib.Status(backend_lib.RUNNING), backend_lib.Status(
    backend_lib.SUCCEEDED, 0
)
PRE = backend_lib.Status(backend_lib.PREEMPTED, 143, "notice")


def _sup(be, **kw):
  """A supervisor on a fake clock.

  Args:
    be: The backend.
    **kw: Supervisor arguments.

  Returns:
    (supervisor, a one-element list holding the clock time).
  """
  t = [0.0]
  kw.setdefault("poll_s", 1.0)
  return (
      supervisor_lib.Supervisor(
          be,
          clock=lambda: t[0],
          sleep=lambda s: t.__setitem__(0, t[0] + s),
          **kw,
      ),
      t,
  )


def _state(be):
  """The supervisor state of run kuhn-a.

  Args:
    be: The in-memory backend.

  Returns:
    The state dict.
  """
  return json.loads(be.files["mem://runs/kuhn-a/supervisor.json"])


def test_preempted_then_done_known_answer():
  """A preempted attempt then a done one: the known state."""
  be = FakeBackend([[R, R, PRE], [R, OK]])
  sup, _ = _sup(be)
  assert sup.run(RUN) == 0
  st = _state(be)
  assert [a["outcome"] for a in st["attempts"]] == ["preempted", "done"]
  assert [a["rc"] for a in st["attempts"]] == [143, 0]
  assert st["final"] == "done" and st["rc"] == 0 and st["run"] == RUN.record()
  # prepare once before the run's state is taken over (a refusal leaves it),
  # then per attempt
  assert be.calls == [
      ("prepare", "kuhn-a"),
      ("prepare", "kuhn-a"),
      ("provision", 0),
      ("launch", 0),
      ("teardown", 0),
      ("prepare", "kuhn-a"),
      ("provision", 1),
      ("launch", 1),
      ("teardown", 1),
  ]


def test_a_done_run_is_idempotent_and_a_different_run_is_refused():
  """A done run is idempotent; a different run under its id is refused."""
  be = FakeBackend([[OK]])
  sup, _ = _sup(be)
  assert sup.run(RUN) == 0
  n = len(be.calls)
  assert sup.run(RUN) == 0 and len(be.calls) == n  # nothing launched again
  with pytest.raises(supervisor_lib.RunMismatchError, match="different run"):
    sup.run(backend_lib.RunSpec("kuhn-a", train_args=("--config=other",)))


def test_failed_is_final_and_retry_continues():
  """FAILED is final; retry=True continues."""
  be = FakeBackend(
      [[R, backend_lib.Status(backend_lib.FAILED, 2, "boom")], [OK]]
  )
  sup, _ = _sup(be)
  assert sup.run(RUN) == 2
  assert _state(be)["final"] == "failed"
  assert sup.run(RUN) == 2  # no retry: the stored rc
  assert sup.run(RUN, retry=True) == 0
  assert [a["outcome"] for a in _state(be)["attempts"]] == ["failed", "done"]


def test_no_capacity_waits_and_tries_again():
  """No capacity waits and tries again."""
  be = FakeBackend([[OK]], stockouts=2)
  sup, t = _sup(be, capacity_wait_s=600)
  assert sup.run(RUN) == 0
  st = _state(be)
  assert (
      len(st["capacity_waits"]) == 2
      and st["capacity_waits"][0]["tried"][0]["zone"] == "z1"
  )
  assert [w["after_attempts"] for w in st["capacity_waits"]] == [0, 0]
  assert len(st["attempts"]) == 1 and t[0] == 1200


def test_bounds_max_preemptions_and_deadline():
  """max_preemptions and the deadline bound the run."""
  be = FakeBackend([[PRE]] * 3)
  sup, _ = _sup(be, max_preemptions=1)
  assert sup.run(RUN) == supervisor_lib.EXIT_GAVE_UP
  st = _state(be)
  assert [a["outcome"] for a in st["attempts"]] == ["preempted", "preempted"]
  assert st["final"] == "gave_up" and "max_preemptions" in st["final_detail"]
  be = FakeBackend([[OK]], stockouts=5)
  sup, _ = _sup(be, capacity_wait_s=600, deadline_s=1000)
  assert sup.run(RUN) == supervisor_lib.EXIT_GAVE_UP and _state(be)[
      "final_detail"
  ].startswith("no capacity")


def test_a_job_that_cannot_resume_ends_preempted():
  """A job that cannot resume ends PREEMPTED."""
  be = FakeBackend([[PRE], [OK]])
  sup, _ = _sup(be, resume_on_preemption=False)
  assert sup.run(RUN) == 143
  st = _state(be)
  assert st["final"] == "preempted" and [
      a["outcome"] for a in st["attempts"]
  ] == ["preempted"]


def test_the_attempt_is_aborted_when_the_supervisor_raises():
  """The attempt is aborted, not torn down, when the supervisor raises."""
  for exc in (RuntimeError("poll failed"), KeyboardInterrupt()):

    class Boom(FakeBackend):
      """poll raises exc."""

      def poll(self, handle, exc=exc):
        raise exc

    be = Boom([[OK]])
    sup, _ = _sup(be)
    with pytest.raises(type(exc)):
      sup.run(RUN)
    assert ("abort", 0, type(exc).__name__) in be.calls and (
        "teardown",
        0,
    ) not in be.calls
    assert _state(be)["attempts"][0]["outcome"] == "failed"


def test_abort_still_runs_when_the_state_write_fails():
  """`abort` still runs when the state write fails."""

  class Down(FakeBackend):
    """Storage that fails the write of a launched handle."""

    def write_text(self, uri, text):
      if '"launched": true' in text:
        raise RuntimeError("storage down")
      super().write_text(uri, text)

  be = Down([[OK]])
  sup, _ = _sup(be)
  with pytest.raises(RuntimeError, match="storage down"):
    sup.run(RUN)
  assert ("abort", 0, "RuntimeError") in be.calls


def test_a_handle_poll_extended_is_persisted_at_once():
  """A handle extended by poll is persisted at once.

  What an attempt learns while it runs (an id) reaches the run's state
  before the attempt ends: a restarted supervisor knows it.
  """
  be = FakeBackend([[]])
  seen = []

  def learn(handle):
    """Adds a job id to the handle.

    Args:
      handle: the attempt's handle.

    Returns:
      RUNNING.
    """
    handle["job_id"] = 77
    return R

  def check(handle):
    """Records the stored job id.

    Args:
      handle: the attempt's handle.

    Returns:
      SUCCEEDED.
    """
    del handle
    seen.append(_state(be)["attempts"][0]["handle"].get("job_id"))
    return OK

  be.script[0] = [learn, check]
  sup, _ = _sup(be)
  assert sup.run(RUN) == 0
  assert seen == [77]


def test_heartbeats_while_an_attempt_runs():
  """The owner's heartbeat is refreshed while an attempt runs."""
  be = FakeBackend([[R] * 10 + [OK]])
  sup, _ = _sup(be, poll_s=100, heartbeat_s=300)
  assert sup.run(RUN) == 0
  # owner.heartbeat_t moved during the 1000 s of polling
  assert _state(be)["owner"]["heartbeat_t"] >= 900


def test_a_run_whose_supervisor_may_live_is_refused():
  """Same host: refused while the owner's pid lives.

  Another host: until its heartbeat is 3 x heartbeat_s old.
  """

  def state_with(owner):
    """A FakeBackend whose stored state has this owner.

    Args:
      owner: the owner record.

    Returns:
      the backend.
    """
    be = FakeBackend([[OK]])
    be.files["mem://runs/kuhn-a/supervisor.json"] = json.dumps({
        "run": RUN.record(),
        "attempts": [],
        "capacity_waits": [],
        "final": None,
        "rc": None,
        "owner": owner,
    })
    return be

  me = dict(host="box", pid=4242, token="other", heartbeat_t=0.0)
  be = state_with(me)
  sup, _ = _sup(be, host="box", pid=1, alive=lambda pid: pid == 4242)
  with pytest.raises(backend_lib.RunInFlightError, match="may still run"):
    sup.run(RUN)
  sup, _ = _sup(be, host="box", pid=1, alive=lambda pid: False)
  assert sup.run(RUN) == 0  # its pid is gone
  be = state_with(dict(me, host="elsewhere", heartbeat_t=-100.0))
  sup, _ = _sup(be, host="box", heartbeat_s=300)
  with pytest.raises(backend_lib.RunInFlightError):
    sup.run(RUN)
  be = state_with(dict(me, host="elsewhere", heartbeat_t=-1000.0))
  sup, _ = _sup(be, host="box", heartbeat_s=300)
  assert sup.run(RUN) == 0  # its heartbeat is stale
  be = state_with(dict(me, host="box", pid=1))
  sup, _ = _sup(be, host="box", pid=1, alive=lambda pid: True)
  # this very process (a previous supervisor object in it)
  assert sup.run(RUN) == 0


def test_capacity_is_waited_for_until_the_deadline_has_passed():
  """The wait begins before the deadline passes; one more try follows it."""
  be = FakeBackend([[OK]], stockouts=5)
  sup, t = _sup(be, capacity_wait_s=600, deadline_s=1000)
  assert sup.run(RUN) == supervisor_lib.EXIT_GAVE_UP
  assert [c for c in be.calls if c[0] == "provision"] == [("provision", 0)] * 3
  assert t[0] == 1200
  be = FakeBackend([[OK]], stockouts=1)
  sup, _ = _sup(be, capacity_wait_s=900, deadline_s=1000)
  # the try after the wait ends before the deadline is checked again
  assert sup.run(RUN) == 0


def test_the_deadline_window_opens_at_the_first_attempts_start():
  """The deadline window opens at the first attempt's start.

  A launch that queued (started_t later than the supervisor's start) keeps
  its whole window; the window survives a restart (it is in the state).
  """

  def pre_after(dt):
    """A poll that advances the clock by dt, then reports PREEMPTED.

    Args:
      dt: seconds.

    Returns:
      the poll.
    """

    def poll(handle):
      """Advances the clock; PREEMPTED.

      Args:
        handle: the attempt's handle.

      Returns:
        PREEMPTED.
      """
      del handle
      clock[0] += dt
      return PRE

    return poll

  be = FakeBackend(
      [[pre_after(1200)], [pre_after(400)], [OK]], started_t=lambda n: 500.0
  )
  sup, clock = _sup(be, deadline_s=1000)
  assert sup.run(RUN) == supervisor_lib.EXIT_GAVE_UP
  st = _state(be)
  assert st["window_start_t"] == 500.0
  # attempt 0 ended at 1200 < 1500 (a relaunch); attempt 1 ended at 1600: past
  # the deadline
  assert [a["outcome"] for a in st["attempts"]] == ["preempted", "preempted"]
  assert st["final_detail"] == "run deadline passed"


def test_a_retry_opens_a_new_window():
  """A retry opens a new deadline window."""
  be = FakeBackend(
      [[backend_lib.Status(backend_lib.FAILED, 2, "x")], [OK]],
      started_t=lambda n: 0.0,
  )
  sup, clock = _sup(be, deadline_s=1000)
  assert sup.run(RUN) == 2
  clock[0] = 5000.0
  be.started_t = lambda n: 5000.0
  assert sup.run(RUN, retry=True) == 0
  assert _state(be)["window_start_t"] == 5000.0


def test_a_restarted_supervisor_adopts_or_loses_the_attempt_in_flight():
  """A restarted supervisor adopts the attempt in flight, or loses it."""
  for adoptable, outcomes in ((True, ["done"]), (False, ["lost", "done"])):
    be = FakeBackend([[OK], [OK]], adoptable=adoptable)
    sup, _ = _sup(be)
    st = {
        "run": RUN.record(),
        "attempts": [{
            "n": 0,
            "handle": {"backend": "fake", "attempt": 0},
            "started": "t",
            "ended": None,
            "outcome": None,
            "rc": None,
            "detail": "",
        }],
        "capacity_waits": [],
        "final": None,
        "rc": None,
    }
    be.files["mem://runs/kuhn-a/supervisor.json"] = json.dumps(st)
    assert sup.run(RUN) == 0
    assert [a["outcome"] for a in _state(be)["attempts"]] == outcomes
    assert ("adopt", 0) in be.calls and ("teardown", 0) in be.calls


def test_an_adopted_attempt_preempted_without_resume_is_final():
  """An adopted attempt preempted without a resume is final."""
  be = FakeBackend([[PRE], [OK]], adoptable=True)
  sup, _ = _sup(be, resume_on_preemption=False)
  st = {
      "run": RUN.record(),
      "attempts": [{
          "n": 0,
          "handle": {"backend": "fake", "attempt": 0},
          "started": "t",
          "ended": None,
          "outcome": None,
          "rc": None,
          "detail": "",
      }],
      "capacity_waits": [],
      "final": None,
      "rc": None,
  }
  be.files["mem://runs/kuhn-a/supervisor.json"] = json.dumps(st)
  assert sup.run(RUN) == 143
  got = _state(be)
  assert got["final"] == "preempted" and got["rc"] == 143
  assert [a["outcome"] for a in got["attempts"]] == [
      "preempted"
  ]  # no new attempt


def test_runspec_contract():
  """RunSpec refuses a bad run_id and bad train_args."""
  with pytest.raises(ValueError, match="run_id"):
    backend_lib.RunSpec("Bad_ID")
  with pytest.raises(ValueError, match="checkpoint_dir"):
    backend_lib.RunSpec("ok", train_args=("--checkpoint_dir=/x",))
  r = backend_lib.RunSpec(
      "ok", train_args=("--config=c",), env={"A": "1"}, plugins=("p.q",)
  )
  assert r.trainer_argv("gs://bucket/runs/ok", "py") == [
      "py",
      "-u",
      "-m",
      backend_lib.TRAINER_MODULE,
      "--config=c",
      "--checkpoint_dir=gs://bucket/runs/ok/ckpt",
  ]
  assert r.trainer_env() == {"A": "1", plugins.ENV: "p.q"}
  assert r.trainer_shell("/r", "python3") == (
      f"A=1 {plugins.ENV}=p.q python3 -u -m {backend_lib.TRAINER_MODULE} "
      "--config=c "
      "--checkpoint_dir=/r/ckpt"
  )
  # The same code from a subprocess (-signal) and from a shell (128 + signal):
  # SIGTERM is a preemption, SIGKILL (the OOM killer) a failure.
  assert [
      backend_lib.classify_exit(c) for c in (0, 143, -15, -9, 137, 1, 2)
  ] == [
      backend_lib.SUCCEEDED,
      backend_lib.PREEMPTED,
      backend_lib.PREEMPTED,
      backend_lib.FAILED,
      backend_lib.FAILED,
      backend_lib.FAILED,
      backend_lib.FAILED,
  ]


def test_a_backend_that_refuses_the_run_leaves_its_state_as_it_was():
  """A backend that refuses the run leaves its state as it was."""

  class Refuses(FakeBackend):
    """prepare refuses every run."""

    def prepare(self, run):
      raise supervisor_lib.RunMismatchError("not this checkpoint dir")

  be = Refuses([[backend_lib.Status(backend_lib.FAILED, 2, "x")]])
  be.files["mem://runs/kuhn-a/supervisor.json"] = json.dumps({
      "run": RUN.record(),
      "attempts": [],
      "capacity_waits": [],
      "final": "failed",
      "rc": 2,
  })
  before = be.files["mem://runs/kuhn-a/supervisor.json"]
  sup, _ = _sup(be)
  with pytest.raises(supervisor_lib.RunMismatchError):
    sup.run(RUN, retry=True)
  assert be.files["mem://runs/kuhn-a/supervisor.json"] == before


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
