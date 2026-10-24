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

"""launch.preemption: the preemption notice a run checkpoints on.

* SIGTERM, inside on_sigterm (the "sigterm" source), sets the notice
instead of killing the process; the previous handler comes back;
* notice sources: registered by name, started and stopped together, an
unknown name raises and starts nothing, the defaults are settable;
* the GCE watch (launch.gcp.watch_preemption) reads the metadata server's
instance/preempted: "TRUE" sets the notice (reason gce); off GCE (the
first read fails) it stops and says so, never setting the notice;
* end to end: a real trainer process SIGTERMed mid-run
exits PREEMPTED_EXIT after a resume checkpoint, and a second process
continues the run from it.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import gcp
from open_spiel.python.scale.launch import preemption as preemption_lib
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.league import resume as resume_ckpt


def test_sigterm_raises_the_notice_and_restores_the_handler():
  """SIGTERM raises the notice; the handler is restored after the block."""
  before = signal.getsignal(signal.SIGTERM)
  n = preemption_lib.Notice()
  assert not n() and n.reason is None
  with preemption_lib.on_sigterm(n):
    os.kill(os.getpid(), signal.SIGTERM)
    deadline = time.time() + 5
    while not n() and time.time() < deadline:
      time.sleep(0.01)
    assert n() and n.reason == "SIGTERM"
  assert signal.getsignal(signal.SIGTERM) is before
  n.set("gce")  # the first reason stays
  assert n.reason == "SIGTERM"


class _Resp:
  """A fake metadata server response.

  Attributes:
    body: the response body.
    headers: the response headers.
  """

  def __init__(self, body, etag=None):
    self.body = body
    self.headers = {} if etag is None else {"ETag": etag}

  def read(self):
    """The body as bytes.

    Returns:
      the bytes.
    """
    return self.body.encode()

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False


def test_gce_watch_known_answers():
  """The GCE watch reads, long-polls and sets the notice on TRUE."""
  urls = []
  answers = iter(["FALSE", "FALSE", "TRUE"])

  def opener(req, timeout):
    """A metadata server that answers from a script.

    Args:
      req: the request.
      timeout: its timeout.

    Returns:
      the response.
    """
    del timeout
    urls.append(req.full_url)
    assert req.get_header("Metadata-flavor") == "Google"
    return _Resp(next(answers))

  n = preemption_lib.Notice()
  th = gcp.watch_preemption(n, opener=opener, retry_s=0.0)
  th.join(10)
  assert not th.is_alive() and n() and n.reason == "gce"
  assert urls[0] == gcp.PREEMPTED_URL  # the plain read first ...
  # ... then the long polls
  assert all("wait_for_change=true" in u for u in urls[1:])
  assert len(urls) == 3


def test_gce_watch_polls_from_the_last_etag_so_a_flip_between_polls_is_seen():
  """The watch polls from the last ETag, so a flip between polls is seen."""
  urls = []
  answers = iter([("FALSE", "e1"), ("FALSE", "e2")])

  def opener(req, timeout):
    """A metadata server that flips between two polls.

    Args:
      req: the request.
      timeout: its timeout.

    Returns:
      the response.

    Raises:
      URLError: the scripted failures.
    """
    del timeout
    urls.append(req.full_url)
    if len(urls) == 3:
      # the flip happens during the retry wait
      raise urllib.error.URLError("reset")
    if len(urls) == 4:
      return _Resp("TRUE", "e3")
    return _Resp(*next(answers))

  n = preemption_lib.Notice()
  th = gcp.watch_preemption(n, opener=opener, retry_s=0.0)
  th.join(10)
  assert not th.is_alive() and n() and n.reason == "gce"
  assert "last_etag" not in urls[0] and urls[1].endswith("&last_etag=e1")
  assert urls[2].endswith("&last_etag=e2") and urls[3].endswith("&last_etag=e2")


def test_gce_watch_off_gce_stops_without_the_notice():
  """Off GCE the watch stops without setting the notice."""

  def opener(req, timeout):
    raise urllib.error.URLError(
        "metadata.google.internal: Name or service not known"
    )

  n = preemption_lib.Notice()
  th = gcp.watch_preemption(n, opener=opener, retry_s=0.0)
  th.join(10)
  assert not th.is_alive() and not n()
  assert "off GCE" in th.status


def test_gce_watch_retries_a_failed_poll_after_the_first_read():
  """The watch retries a failed poll after the first read."""
  calls = []

  def opener(req, timeout):
    """A metadata server whose first long poll fails.

    Args:
      req: the request.
      timeout: its timeout.

    Returns:
      the response.

    Raises:
      URLError: the first long poll.
    """
    del timeout
    calls.append(req.full_url)
    if len(calls) == 1:
      return _Resp("FALSE")
    if len(calls) == 2:
      raise urllib.error.URLError("reset")
    return _Resp("TRUE")

  n = preemption_lib.Notice()
  th = gcp.watch_preemption(n, opener=opener, retry_s=0.0)
  th.join(10)
  assert n() and len(calls) == 3


def test_stop_ends_the_watch():
  """stop() ends the watch."""
  ev = threading.Event()

  def opener(req, timeout):
    """A metadata server that never flips.

    Args:
      req: the request.
      timeout: its timeout.

    Returns:
      the response.
    """
    del timeout
    if "wait_for_change" in req.full_url:
      ev.wait(0.05)
    return _Resp("FALSE")

  n = preemption_lib.Notice()
  th = gcp.watch_preemption(n, opener=opener, retry_s=0.0)
  th.stop()
  th.join(10)
  assert not th.is_alive() and not n()


def test_notice_sources_start_and_stop_together(monkeypatch):
  """Notice sources start together and stop together, in reverse order."""
  # Reaches private _SOURCES of the code under test.
  monkeypatch.setattr(preemption_lib, "_SOURCES", dict(preemption_lib._SOURCES))  # pylint: disable=protected-access
  # Reaches private _DEFAULT of the code under test.
  monkeypatch.setattr(preemption_lib, "_DEFAULT", list(preemption_lib._DEFAULT))  # pylint: disable=protected-access
  events = []

  def source(name):
    """A notice source that records its start and stop.

    Args:
      name: the source's name.

    Returns:
      its start function.
    """

    def start(notice):
      """Records the start.

      Args:
        notice: the notice.

      Returns:
        the stop function.
      """
      del notice
      events.append(("start", name))
      return lambda: events.append(("stop", name))

    return start

  preemption_lib.register_notice_source("a", source("a"))
  preemption_lib.register_notice_source("b", source("b"))
  assert {"sigterm", "gce", "a", "b"} <= set(preemption_lib.sources())
  stop = preemption_lib.start_notice_sources(
      preemption_lib.Notice(), ["a", "b"]
  )
  stop()
  assert events == [
      ("start", "a"),
      ("start", "b"),
      ("stop", "b"),
      ("stop", "a"),
  ]
  events.clear()
  with pytest.raises(KeyError, match="unknown notice sources"):
    preemption_lib.start_notice_sources(preemption_lib.Notice(), ["a", "nope"])
  assert not events
  with pytest.raises(ValueError, match="already registered"):
    preemption_lib.register_notice_source("a", source("a"))
  preemption_lib.set_default_sources(["sigterm", "a"])
  assert preemption_lib.default_sources() == ("sigterm", "a")
  with pytest.raises(KeyError):
    preemption_lib.set_default_sources(["nope"])


def test_the_built_in_default_is_sigterm_only():
  """The built-in default sources are sigterm only."""
  r = subprocess_env.run_python(
      [
          "-c",
          (
              f"from {preemption_lib.__name__} import default_sources; "
              "print(default_sources())"
          ),
      ],
      env={plugins.ENV: ""},
  )
  assert r.returncode == 0, r.stderr[-3000:]
  assert r.stdout.strip() == "('sigterm',)"


def test_the_sigterm_source_restores_the_handler():
  """The sigterm source restores the handler when stopped."""
  before = signal.getsignal(signal.SIGTERM)
  n = preemption_lib.Notice()
  stop = preemption_lib.start_notice_sources(n, ["sigterm"])
  assert signal.getsignal(signal.SIGTERM) is not before
  stop()
  assert signal.getsignal(signal.SIGTERM) is before and not n()
  with pytest.raises(KeyError, match="registered"):
    preemption_lib.start_notice_sources(n, ["no_such_source"])


def test_exit_code():
  """PREEMPTED_EXIT is 128 + SIGTERM."""
  assert preemption_lib.PREEMPTED_EXIT == 128 + signal.SIGTERM


def _train(ck, *extra, env=None):
  """The trainer's argv for a tiny kuhn run in ck.

  Args:
    ck: the checkpoint dir.
    *extra: further trainer flags.
    env: unused.

  Returns:
    the argv.
  """
  del env
  return [
      "-m",
      backend_lib.TRAINER_MODULE,
      "--config=mmd_kuhn_poker_jax",
      f"--checkpoint_dir={ck}",
      "--num_envs=8",
      "--num_steps=8",
      "--preemption_sources=sigterm",
      "--compilation_cache_dir=",
      *extra,
  ]


def test_sigterm_mid_run_checkpoints_exits_and_resumes(tmp_path):
  """SIGTERM mid-run checkpoints, exits, and the run resumes.

  A real train process: SIGTERM after its first metrics rows -> it ends
  after the iteration in flight with a resume checkpoint and exit 143; a
  second process (--stop_after past it) continues from that checkpoint.

  Args:
    tmp_path: pytest's per-test directory.
  """
  ck = tmp_path / "ck"
  # A long run (constant LR: the length is not in its resume hash) with a
  # keep only every 10,000 iterations and no timer.
  proc = subprocess.Popen(
      subprocess_env.python_command()
      + _train(
          ck,
          "--num_updates=100000",
          "--resume_every_min=0",
          "--set=keep_every=10000",
      ),
      cwd=subprocess_env.child_cwd(),
      env=subprocess_env.child_env(),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
  )
  metrics = ck / "metrics" / "metrics.jsonl"
  deadline = time.time() + 600
  while time.time() < deadline and proc.poll() is None:
    if metrics.exists() and len(metrics.read_text().splitlines()) >= 3:
      break
    time.sleep(0.5)
  assert proc.poll() is None, proc.communicate()[0][-3000:]
  # metrics.jsonl may lag (flush_every): the process is in its loop by now.
  proc.send_signal(signal.SIGTERM)
  out, _ = proc.communicate(timeout=600)
  assert proc.returncode == preemption_lib.PREEMPTED_EXIT, out[-3000:]
  rs = resume_ckpt.read(str(ck))
  assert (
      rs is not None and rs["reason"] == "preemption" and rs["step"] >= 1
  ), out[-3000:]
  assert run_state.read_train_state(str(ck)) is None
  k = int(rs["step"])
  p2 = subprocess_env.run_python(
      _train(
          ck,
          "--num_updates=100000",
          "--resume_every_min=0",
          "--set=keep_every=10000",
          f"--stop_after={k + 2}",
      ),
      timeout=900,
  )
  assert p2.returncode == 0, (p2.stdout + p2.stderr)[-3000:]
  assert "from its resume checkpoint" in p2.stdout + p2.stderr
  ts = run_state.read_train_state(str(ck))
  assert ts["step"] == k + 2 and resume_ckpt.read(str(ck)) is None
  rows = [json.loads(l) for l in metrics.read_text().splitlines()]
  steps = [r["step"] for r in rows if "train/main/frames" in r]
  assert steps == list(range(1, k + 3)), steps  # no gap, no duplicate


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
