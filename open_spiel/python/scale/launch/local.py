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

"""The local backend: every attempt is a subprocess on this machine.

Local runs exercise the same supervisor, resume and preemption path as cloud
runs, and serve small experiments that need no cloud.

  backend = local.LocalBackend("/data/runs")  # python: default this machine's

A run's storage is <root>/<run_id>, a local directory holding
supervisor.json, the checkpoint dir ckpt/ and attempts/<n>/train.log.
provision always has capacity (one candidate, "local"); launch starts
the trainer (`python -m <TRAINER_MODULE> ...
--checkpoint_dir=<root>/<run_id>/ckpt`) in its own process group with stdout and
stderr in attempts/<n>/train.log; poll reads its exit code through
backend.classify_exit; teardown kills the process group if it still runs.
preempt(handle) sends SIGTERM, which is what a preemption notice does to a
trainer: it checkpoints and exits PREEMPTED_EXIT. A process started by another
supervisor process is not adopted, and teardown does not signal it (after a
reboot its pid may belong to an unrelated process); it is logged instead.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any, Optional, Sequence

from absl import logging

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import storage as storage_lib


class LocalBackend(backend_lib.Backend):
  """The local backend (module doc).

  Attributes:
    root: the local directory runs live under (absolute).
    python: the interpreter that runs the trainer (a path or a command);
      None is subprocess_env.python_command(), resolved at each attempt.
    env: extra environment of every trainer.
    storage: the run's storage (local files).
  """

  name = "local"

  def __init__(
      self,
      root: str,
      *,
      python: str | Sequence[str] | None = None,
      env: Optional[dict[str, str]] = None,
  ):
    """Initializes the backend.

    Args:
      root: the local directory runs live under.
      python: the interpreter that runs the trainer (a path or a command);
        None is subprocess_env.python_command(), resolved at each attempt.
      env: extra environment of every trainer.

    Raises:
      ValueError: root is empty or a gs:// path.
    """
    if not root or str(root).startswith("gs://"):
      raise ValueError(f"LocalBackend needs a local directory, got {root!r}")
    self.root = os.path.abspath(str(root))
    self.python = python
    self.env = dict(env or {})
    self.storage = storage_lib.LocalStorage()
    self._procs: dict[int, subprocess.Popen[bytes]] = {}

  def run_uri(self, run_id: str) -> str:
    """The run's dir, <root>/<run_id>.

    Args:
      run_id: the run.

    Returns:
      the dir.
    """
    return os.path.join(self.root, run_id)

  def plan(self, run: backend_lib.RunSpec, attempt: int) -> str:
    """The trainer's command line on this machine.

    Args:
      run: the run.
      attempt: unused.

    Returns:
      the command line.
    """
    del attempt
    return " ".join(run.trainer_argv(self.run_uri(run.run_id), self.python))

  def prepare(self, run: backend_lib.RunSpec) -> None:
    """Creates the run's dir.

    Args:
      run: the run.
    """
    os.makedirs(self.run_uri(run.run_id), exist_ok=True)

  def provision(self, run: backend_lib.RunSpec, attempt: int) -> dict[str, Any]:
    """The local machine: always has capacity.

    Args:
      run: the run.
      attempt: the attempt number.

    Returns:
      the handle.
    """
    return {
        "backend": self.name,
        "attempt": int(attempt),
        "candidate": "local",
        "run_id": run.run_id,
    }

  def launch(
      self, handle: dict[str, Any], run: backend_lib.RunSpec, attempt: int
  ) -> dict[str, Any]:
    """Starts the trainer as a subprocess, logging to the attempt's dir.

    Args:
      handle: the attempt's handle.
      run: the run.
      attempt: the attempt number.

    Returns:
      the handle with the subprocess's pid.
    """
    d = os.path.join(self.run_uri(run.run_id), "attempts", str(int(attempt)))
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(d, "train.log")
    argv = run.trainer_argv(self.run_uri(run.run_id), self.python)
    env = {**os.environ, **self.env, **run.trainer_env()}
    with open(log_path, "ab") as log:
      proc = subprocess.Popen(
          argv,
          stdout=log,
          stderr=subprocess.STDOUT,
          env=env,
          start_new_session=True,
      )
    self._procs[proc.pid] = proc
    logging.info(
        "local backend: attempt %d of %s is pid %d (log %s)",
        attempt,
        run.run_id,
        proc.pid,
        log_path,
    )
    return dict(handle, pid=proc.pid, log=log_path, argv=argv)

  def poll(self, handle: dict[str, Any]) -> backend_lib.Status:
    """The trainer subprocess's Status.

    Args:
      handle: the attempt's handle.

    Returns:
      the Status.

    Raises:
      RuntimeError: this backend did not start the pid.
    """
    proc = self._procs.get(int(handle["pid"]))
    if proc is None:
      raise RuntimeError(
          f"local backend: pid {handle['pid']} was not started by this backend"
      )
    rc = proc.poll()
    if rc is None:
      return backend_lib.Status(backend_lib.RUNNING)
    return backend_lib.Status(
        backend_lib.classify_exit(rc),
        int(rc),
        f"exit {rc}; log {handle['log']}",
    )

  def preempt(self, handle: dict[str, Any]) -> None:
    """SIGTERM to the trainer: a preemption notice (it checkpoints, exits).

    Args:
      handle: the attempt's handle.

    Raises:
      RuntimeError: this backend did not start the pid.
    """
    if int(handle["pid"]) not in self._procs:
      raise RuntimeError(
          f"local backend: pid {handle['pid']} was not started by this backend"
      )
    os.killpg(int(handle["pid"]), signal.SIGTERM)

  def teardown(self, handle: dict[str, Any]) -> None:
    """Kills the trainer's process group if it still runs; idempotent.

    Args:
      handle: the attempt's handle.
    """
    pid = handle.get("pid")
    if pid is None:
      return
    proc = self._procs.pop(int(pid), None)
    if proc is not None and proc.poll() is None:
      os.killpg(int(pid), signal.SIGKILL)
      proc.wait()
    # not started here: the pid may since belong to another process
    elif proc is None:
      logging.warning(
          "local backend: pid %d was not started by this backend; not killing "
          "it",
          int(pid),
      )
