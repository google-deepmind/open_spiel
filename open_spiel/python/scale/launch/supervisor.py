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

"""The supervisor: one run, many attempts, through a Backend.

Preemptible machines make long runs cheap only if something relaunches them.
The supervisor drives one run to a final outcome: it provisions, launches
and polls attempts, relaunches after a preemption (the trainer resumes from
its checkpoint), waits out capacity shortages and stops at a deadline.

  sup = Supervisor(backend)
  rc = sup.run(RunSpec("kuhn-a", train_args=("--config=mmd_kuhn_poker_jax",
  "--seed=0")))

The run's identity is its run_id. Its state is one JSON object in the run's
own storage, <backend.run_uri(run_id)>/supervisor.json (never a file on the
launching machine), rewritten after every step:

  {"run": RunSpec.record(), "created": UTC time, "backend": backend name,
   "attempts": [{"n", "handle", "started", "ended", "outcome", "rc",
   "detail"}...], "capacity_waits": [{"after_attempts", "at", "tried"}...],
   "final": null | "done" | "failed" | "preempted" | "gave_up",
   "rc": null | int, "final_detail": str, "finished": UTC time (once final),
   "window_start_t": epoch seconds, "owner": {"host", "pid", "token",
   "heartbeat_t"}}

Idempotent: running a run whose state says done returns 0 and launches
nothing; a run that failed or gave up returns its rc unless retry=True (a
retry opens a new deadline window); a state written for a different RunSpec
under the same run_id is refused (RunMismatchError).

One supervisor per run, best effort: the state names its owner (host, pid, a
token) and the owner refreshes heartbeat_t every heartbeat_s while it waits
on an attempt. A run whose state is not final and whose owner may still be alive
is refused (backend.RunInFlightError): on the same host, while the owner's pid
lives;
on another host, until its heartbeat is 3 x heartbeat_s old. A supervisor
that died mid-attempt is replaced by another one started the same way: it
adopts the attempt in flight (Backend.adopt) or, when the backend cannot,
tears it down and records it "lost" (a preemption-like outcome: the next
attempt resumes from the checkpoint); a backend that can tell the attempt
may still be running raises RunInFlightError instead. The claim is a read, then
a
write with no precondition on the stored object, so two supervisors started
at the same moment can both see no live owner and both claim the run.

Each attempt: prepare, provision (ordered fallback; NoCapacityError: wait
capacity_wait_s and try again), launch, poll every poll_s until terminal
(a handle that poll extended is persisted at once), teardown (always; on an
exception in the supervisor, Backend.abort with it). Outcome:
  done       the trainer exited 0                 -> final done, rc 0
  preempted  the machine was preempted / the      -> the next attempt, which
             trainer checkpointed on a notice        resumes (at most
  lost       the attempt vanished                    max_preemptions)
  failed     anything else                        -> final failed, its rc
Bounds: max_preemptions relaunches; deadline_s from the run's window start,
the first attempt's start (its handle's started_t, else when provision
returned or reported no capacity; persisted, so a restarted supervisor keeps
the window): no new attempt after it (final gave_up), and capacity is
waited for while the deadline has not passed when the wait begins (one more
try follows the wait). resume_on_preemption False (a job that cannot
resume): a preempted attempt is final ("preempted", its rc); capacity is
still waited for.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import time
from typing import Any, Callable, Optional
import uuid

from absl import logging

from open_spiel.python.scale.launch import backend as backend_lib

STATE_FILE = "supervisor.json"
FINALS = ("done", "failed", "preempted", "gave_up")
EXIT_GAVE_UP = 3
HEARTBEAT_STALE = (
    3  # heartbeats missed before another host's owner counts as gone
)


def utc(t: Optional[float] = None) -> str:
  """T (default now) as an ISO-8601 UTC timestamp.

  Args:
    t: epoch seconds (default now).

  Returns:
    the timestamp.
  """
  return datetime.datetime.fromtimestamp(
      time.time() if t is None else t, datetime.timezone.utc
  ).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_alive(pid: int) -> bool:
  """Whether a process with this pid exists on this host.

  Args:
    pid: the pid.

  Returns:
    True when it exists.
  """
  try:
    os.kill(int(pid), 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


class RunMismatchError(ValueError):
  """The run_id's stored state is for a different RunSpec."""


class Supervisor:
  """Drives runs through one Backend (module doc).

  Attributes:
    backend: the Backend.
    max_preemptions: relaunches after preemptions or lost attempts.
    capacity_wait_s: the wait after NoCapacityError before trying again.
    poll_s: the wait between polls of a running attempt.
    deadline_s: no new attempt after this many seconds from the run's window
      start (None: no deadline).
    clock: the time source.
    sleep: the sleep function.
    poll_sleep: the sleep between polls.
    resume_on_preemption: False for a job that cannot resume.
    heartbeat_s: how often the owner refreshes its heartbeat.
    host: this supervisor's host name (its identity, with pid).
    pid: this supervisor's pid.
    alive: the liveness check of another owner's pid.
    token: this supervisor's unique ownership token.
    state: the run's state as last written (read-only to callers).
    launched: attempts the last run() started.
  """

  def __init__(
      self,
      backend: backend_lib.Backend,
      *,
      max_preemptions: int = 20,
      capacity_wait_s: float = 600.0,
      poll_s: float = 30.0,
      deadline_s: Optional[float] = None,
      clock: Callable[[], float] = time.time,
      sleep: Callable[[float], Any] = time.sleep,
      on_event: Optional[Callable[[str], Any]] = None,
      resume_on_preemption: bool = True,
      heartbeat_s: float = 300.0,
      host: Optional[str] = None,
      pid: Optional[int] = None,
      alive: Callable[[int], bool] = pid_alive,
      poll_sleep: Optional[Callable[[float], Any]] = None,
  ):
    """Initializes the supervisor.

    Args:
      backend: the Backend.
      max_preemptions: relaunches after preemptions or lost attempts.
      capacity_wait_s: the wait after NoCapacityError before trying again.
      poll_s: the wait between polls of a running attempt.
      deadline_s: no new attempt after this many seconds from the run's window
        start (None: no deadline).
      clock: the time source (tests inject a fake).
      sleep: the sleep function (tests inject a fake).
      on_event: called with every event line.
      resume_on_preemption: False for a job that cannot resume: a preempted
        attempt is then final.
      heartbeat_s: how often the owner refreshes its heartbeat.
      host: this supervisor's host name (its identity, with pid).
      pid: this supervisor's pid.
      alive: the liveness check of another owner's pid.
      poll_sleep: the sleep between polls; defaults to sleep, and a backend
        whose poll itself waits needs none.

    Raises:
      ValueError: a wait, count or deadline is out of range.
    """
    if (
        max_preemptions < 0
        or capacity_wait_s < 0
        or poll_s <= 0
        or heartbeat_s <= 0
    ):
      raise ValueError(
          "max_preemptions, capacity_wait_s >= 0 and poll_s, heartbeat_s > 0"
      )
    if deadline_s is not None and deadline_s <= 0:
      raise ValueError(f"deadline_s must be > 0, got {deadline_s}")
    self.backend = backend
    self.max_preemptions = int(max_preemptions)
    self.capacity_wait_s = float(capacity_wait_s)
    self.poll_s = float(poll_s)
    self.deadline_s = deadline_s
    self.clock, self.sleep = clock, sleep
    self.poll_sleep = sleep if poll_sleep is None else poll_sleep
    self._on_event = on_event
    self.resume_on_preemption = bool(resume_on_preemption)
    self.heartbeat_s = float(heartbeat_s)
    self.host = socket.gethostname() if host is None else str(host)
    self.pid = os.getpid() if pid is None else int(pid)
    self.alive = alive
    self.token = uuid.uuid4().hex
    self.state: Optional[dict[str, Any]] = (
        None  # the run's state as last written (read-only to callers)
    )
    self.launched = 0  # attempts the last run() started

  # -- state -------------------------------------------------------------
  def state_uri(self, run_id: str) -> str:
    """Where the run's state lives: <run_uri>/supervisor.json.

    Args:
      run_id: the run.

    Returns:
      the URI.
    """
    return backend_lib.join(self.backend.run_uri(run_id), STATE_FILE)

  def read_state(self, run_id: str) -> Optional[dict[str, Any]]:
    """The run's stored state, or None; raises ValueError for another file.

    Args:
      run_id: the run.

    Returns:
      the state, or None.

    Raises:
      ValueError: the file is not a supervisor state.
    """
    text = self.backend.read_text(self.state_uri(run_id))
    if text is None:
      return None
    st = json.loads(text)
    if not isinstance(st, dict) or "run" not in st or "attempts" not in st:
      raise ValueError(f"{self.state_uri(run_id)} is not a supervisor state")
    return st

  def _write(self, st: dict[str, Any]) -> None:
    self.backend.write_text(
        self.state_uri(st["run"]["run_id"]),
        json.dumps(st, indent=1, sort_keys=True) + "\n",
    )
    self.state = json.loads(json.dumps(st))

  def _say(self, msg: str) -> None:
    logging.info("supervisor: %s", msg)
    if self._on_event is not None:
      self._on_event(msg)

  def _owner_may_live(self, owner: Optional[dict[str, Any]]) -> bool:
    """Whether the run's owner may still be alive.

    Args:
      owner: The owner record, or None.

    Returns:
      True when another live supervisor may own the run.
    """
    if not owner or owner.get("token") == self.token:
      return False
    if owner.get("host") == self.host:
      return int(owner["pid"]) != self.pid and bool(
          self.alive(int(owner["pid"]))
      )
    beat = float(owner.get("heartbeat_t") or 0.0)
    return self.clock() - beat < HEARTBEAT_STALE * self.heartbeat_s

  def _claim(self, st: dict[str, Any]) -> None:
    st["owner"] = {
        "host": self.host,
        "pid": self.pid,
        "token": self.token,
        "heartbeat_t": float(self.clock()),
    }

  def _deadline(self, st: dict[str, Any]) -> Optional[float]:
    """The run's deadline time, if it has one.

    Args:
      st: The run state.

    Returns:
      The deadline, or None.
    """
    if self.deadline_s is None or st.get("window_start_t") is None:
      return None
    return float(st["window_start_t"]) + float(self.deadline_s)

  def _open_window(self, st: dict[str, Any], t: Optional[float] = None) -> None:
    if st.get("window_start_t") is None:
      st["window_start_t"] = float(self.clock() if t is None else t)

  # -- the loop ----------------------------------------------------------
  def run(self, run: backend_lib.RunSpec, *, retry: bool = False) -> int:
    """Runs `run` to a final outcome (module doc); returns the exit code.

    Args:
      run: the run.
      retry: continue a run that already has a final outcome.

    Returns:
      the exit code.

    Raises:
      RunMismatchError: run_id already holds a different run.
      RunInFlightError: the run is not final and its supervisor may still run
      it.
      BaseException: whatever interrupted an attempt (re-raised after the
          backend aborts it).
    """
    self.launched = 0
    st = self.read_state(run.run_id)
    if st is None:
      st = {
          "run": run.record(),
          "attempts": [],
          "capacity_waits": [],
          "final": None,
          "rc": None,
          "created": utc(self.clock()),
          "backend": self.backend.name,
      }
    elif st["run"] != run.record():
      raise RunMismatchError(
          f"run_id {run.run_id!r} already holds a different run: stored "
          f"{st['run']}, asked {run.record()}"
      )
    if st["final"] == "done":
      self.state = st
      self._say(
          f"{run.run_id}: already done ({len(st['attempts'])} attempts); "
          "nothing to do"
      )
      return 0
    if st["final"] is None and self._owner_may_live(st.get("owner")):
      raise backend_lib.RunInFlightError(
          f"run {run.run_id!r} is not final and its supervisor may still run "
          f"it: {st['owner']} (state {self.state_uri(run.run_id)})"
      )
    if st["final"] is not None and not retry:
      self.state = st
      self._say(
          f"{run.run_id}: final {st['final']} (rc {st['rc']}); retry=True to "
          "continue"
      )
      return int(st["rc"])
    # Before the state changes hands: a backend that refuses this run (its
    # prepare raises) leaves the stored state as it was.
    self.backend.prepare(run)
    if st["final"] is not None:
      self._say(f"{run.run_id}: retrying after final {st['final']}")
      st["final"], st["rc"] = None, None
      st.pop("window_start_t", None)  # a retry opens a new deadline window
    self._claim(st)
    self._write(st)
    if st["attempts"] and st["attempts"][-1]["ended"] is None:
      self._recover(st, run)
      if st["final"] is not None:
        return self._finish(
            st, st["final"], int(st["rc"]), "the adopted attempt ended"
        )
    after_wait = False
    while True:
      deadline = self._deadline(st)
      if not after_wait and deadline is not None and self.clock() >= deadline:
        return self._finish(st, "gave_up", EXIT_GAVE_UP, "run deadline passed")
      after_wait = False
      preemptions = sum(
          a["outcome"] in ("preempted", "lost") for a in st["attempts"]
      )
      if preemptions > self.max_preemptions:
        return self._finish(
            st,
            "gave_up",
            EXIT_GAVE_UP,
            f"{preemptions - 1} resumes used (max_preemptions)",
        )
      n = len(st["attempts"])
      # Again before every attempt: an attempt runs the code as it is when it
      # starts (GcpBackend stages the current code tree, content-addressed, so
      # an unchanged tree uploads nothing), and a backend may refuse the run
      # now even if it accepted it at the claim.
      self.backend.prepare(run)
      try:
        handle = backend_lib.check_handle(self.backend.provision(run, n))
      except backend_lib.NoCapacityError as e:
        self._open_window(st)
        st["capacity_waits"].append(
            {"after_attempts": n, "at": utc(self.clock()), "tried": e.tried}
        )
        self._write(st)
        deadline = self._deadline(st)
        if deadline is not None and self.clock() >= deadline:
          return self._finish(
              st, "gave_up", EXIT_GAVE_UP, "no capacity before the deadline"
          )
        # One line (backend.NoCapacityError's message); the raw provider
        # errors are in the state file's capacity_waits and the backend's
        # debug log.
        self._say(
            f"{run.run_id} attempt {n}: {e}; retrying in "
            f"{self.capacity_wait_s:.0f} s"
        )
        self.sleep(self.capacity_wait_s)
        after_wait = True
        continue
      self._open_window(st, handle.get("started_t"))
      rec = {
          "n": n,
          "handle": handle,
          "started": utc(self.clock()),
          "ended": None,
          "outcome": None,
          "rc": None,
          "detail": "",
      }
      st["attempts"].append(rec)
      self.launched += 1
      try:
        self._write(st)
        handle = backend_lib.check_handle(self.backend.launch(handle, run, n))
        rec["handle"] = handle
        self._write(st)
        self._say(
            f"{run.run_id} attempt {n}: launched "
            f"{json.dumps(handle, sort_keys=True)}"
        )
        status = self._wait(st, rec)
      except BaseException as e:
        try:
          self._end(
              st,
              rec,
              "failed",
              None,
              f"the supervisor raised {type(e).__name__} (see its log)",
          )
        finally:
          self.backend.abort(rec["handle"], e)
        raise
      self.backend.teardown(rec["handle"])
      outcome = {
          backend_lib.SUCCEEDED: "done",
          backend_lib.FAILED: "failed",
          backend_lib.PREEMPTED: "preempted",
      }[status.state]
      self._end(st, rec, outcome, status.rc, status.detail)
      self._say(
          f"{run.run_id} attempt {n}: {outcome} (rc {status.rc}) "
          f"{status.detail}"
      )
      if outcome == "done":
        return self._finish(st, "done", 0, "")
      if outcome == "failed":
        return self._finish(
            st,
            "failed",
            int(status.rc if status.rc is not None else 1),
            status.detail,
        )
      if not self.resume_on_preemption:
        return self._finish(
            st,
            "preempted",
            int(status.rc if status.rc is not None else 1),
            "not resumed (resume_on_preemption off)",
        )

  def _wait(
      self, st: dict[str, Any], rec: dict[str, Any]
  ) -> backend_lib.Status:
    """Polls rec's attempt until terminal.

    Persists a handle poll extended and the owner's heartbeat.

    Args:
      st: the run's state.
      rec: the attempt's record.

    Returns:
      the attempt's terminal Status.
    """
    seen = json.dumps(rec["handle"], sort_keys=True)
    while True:
      status = self.backend.poll(rec["handle"])
      now = json.dumps(rec["handle"], sort_keys=True)
      beat = (
          self.clock() - float(st["owner"]["heartbeat_t"]) >= self.heartbeat_s
      )
      if now != seen or beat:
        seen = now
        st["owner"]["heartbeat_t"] = float(self.clock())
        self._write(st)
      if status.terminal:
        return status
      self.poll_sleep(self.poll_s)

  def _recover(self, st: dict[str, Any], run: backend_lib.RunSpec) -> None:
    """The last attempt has no end: a previous supervisor died during it.

    Args:
      st: The run state.
      run: The RunSpec.
    """
    rec = st["attempts"][-1]
    got = self.backend.adopt(rec["handle"])
    if got is None:
      self._say(
          f"{run.run_id} attempt {rec['n']}: in flight under a previous "
          "supervisor and "
          "not adoptable; tearing it down (lost)"
      )
      self.backend.teardown(rec["handle"])
      self._end(
          st, rec, "lost", None, "not adoptable after a supervisor restart"
      )
      return
    self._say(f"{run.run_id} attempt {rec['n']}: adopted")
    rec["handle"] = backend_lib.check_handle(got)
    self._write(st)
    status = self._wait(st, rec)
    self.backend.teardown(rec["handle"])
    outcome = {
        backend_lib.SUCCEEDED: "done",
        backend_lib.FAILED: "failed",
        backend_lib.PREEMPTED: "preempted",
    }[status.state]
    self._end(st, rec, outcome, status.rc, status.detail)
    if outcome in ("done", "failed"):
      st["final"], st["rc"] = outcome, 0 if outcome == "done" else int(
          status.rc or 1
      )
    # a preempted attempt is final, as in run()
    elif not self.resume_on_preemption:
      st["final"], st["rc"] = "preempted", int(
          status.rc if status.rc is not None else 1
      )

  def _end(
      self,
      st: dict[str, Any],
      rec: dict[str, Any],
      outcome: str,
      rc: Optional[int],
      detail: str,
  ) -> None:
    """Records the end of an attempt and writes the state.

    Args:
      st: the run's state.
      rec: the attempt's record.
      outcome: the attempt's outcome.
      rc: the return code, or None.
      detail: the end's detail.
    """
    rec.update(
        ended=utc(self.clock()),
        outcome=outcome,
        rc=None if rc is None else int(rc),
        detail=str(detail),
    )
    self._write(st)

  def _finish(self, st: dict[str, Any], final: str, rc: int, why: str) -> int:
    """Records the run's final outcome.

    Args:
      st: The run state.
      final: The final outcome.
      rc: The exit code.
      why: The detail.

    Returns:
      The exit code.
    """
    assert final in FINALS, final
    st["final"], st["rc"], st["final_detail"] = final, int(rc), why
    st["finished"] = utc(self.clock())
    self._write(st)
    self._say(f"{st['run']['run_id']}: final {final} (rc {rc}) {why}")
    return int(rc)
