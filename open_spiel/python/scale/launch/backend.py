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

"""The Backend interface: where and how one run's attempts execute.

A run is one training run (one checkpoint dir) with an idempotent identity,
its run_id. It executes as one or more attempts: a fresh machine (or
process) runs the trainer, `python -m <TRAINER_MODULE> <train_args>
--checkpoint_dir=<run storage>/ckpt`, until it finishes, fails or is
preempted; a preempted attempt is replaced by the next one, which resumes
from the checkpoint dir (the trainer continues a dir holding a
train_state.json). launch.supervisor drives the attempts; the Backend knows
the machines.

  backend = local.LocalBackend("/data/runs")
  rc = supervisor.Supervisor(backend).run(RunSpec("kuhn-a",
  ("--config=mmd_kuhn_poker_jax",)))

Methods:
  storage                a Storage (launch.storage) every attempt reaches;
                         read_text / write_text / exists delegate to it
  run_uri(run_id)        the run's storage root (state, checkpoints, logs):
                         <storage>/<run_id>
  prepare(run)           stages what attempts need (code, inputs); idempotent
  plan(run, attempt)     what the attempt would run, as text, creating nothing
  provision(run, n)      a machine for attempt n, trying the configured
                         candidates in order; raises NoCapacityError (with every
                         candidate it tried) when none has capacity
  launch(handle, run, n) starts the trainer on it
  poll(handle)           Status: RUNNING, SUCCEEDED, FAILED (rc) or
                         PREEMPTED (the machine was taken away, or the trainer
                         checkpointed on a preemption notice and exited
                         preemption.PREEMPTED_EXIT). poll may add what it
                         learned to the handle in place (an id, the machine);
                         the supervisor persists a changed handle at once, so
                         a restarted supervisor knows it
  teardown(handle)       releases the machine; idempotent, also after a failure
  abort(handle, exc)     the supervisor raised exc (KeyboardInterrupt for a
                         Ctrl-C) during the attempt: stop it cleanly (default:
                         teardown)
  adopt(handle)          re-attaches to an attempt a previous supervisor
                         process started (None: it cannot, and the attempt is
                         torn down and counted as lost); raises RunInFlightError
                         when the attempt may still be running and can be
                         neither adopted nor safely replaced

A handle is a JSON object naming its backend and attempt; the supervisor
records it in the run's state. A handle may carry "started_t" (epoch
seconds), when the attempt really started after any queueing in provision;
the run's deadline window opens at the first attempt's start.

Implementations: launch.local.LocalBackend (subprocesses) and
launch.gcp.GcpBackend (Compute Engine through gcloud). Other
platforms implement the same methods and register a config kind with
launch.config.register_backend.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
import signal
from typing import Any, Mapping, Optional, Sequence

from open_spiel.python.scale import plugins as plugins_lib
from open_spiel.python.scale.launch import preemption
from open_spiel.python.scale.launch import storage as storage_lib
from open_spiel.python.scale.launch import subprocess_env

RUNNING, SUCCEEDED, FAILED, PREEMPTED = (
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "PREEMPTED",
)
STATES = (RUNNING, SUCCEEDED, FAILED, PREEMPTED)
RUN_ID = re.compile(r"^[a-z][a-z0-9-]{0,39}$")
CKPT_DIR = "ckpt"
#: The trainer every attempt runs (`python -m TRAINER_MODULE`): the package's
#: train.
TRAINER_MODULE = __name__.rsplit(".", 2)[0] + ".train"


class RunInFlightError(RuntimeError):
  """The run may still be running elsewhere.

  Another supervisor, or an attempt this one cannot adopt, may still be
  running the run: starting another attempt could write its checkpoint dir
  twice.
  """


class NoCapacityError(RuntimeError):
  """No configured candidate could provision a machine now.

  `tried` lists every candidate and zone tried, each a dict with "candidate"
  and "zone", and optionally "what" (what the user asked for, e.g. "L4"; the
  candidate's name when absent), "reason" ("capacity" or "quota"; "capacity"
  when absent) and "error" (the provider's raw error, kept for the record and
  never shown in the message). The message is one line per kind of machine,
  e.g. "no L4 capacity in us-central1-a, us-central1-b; no T4 quota in
  us-central1-b".

  Attributes:
    tried: every candidate and zone tried, as above.
  """

  def __init__(self, tried):
    self.tried = list(tried)
    super().__init__(capacity_summary(self.tried))


def capacity_summary(tried: Sequence[Mapping[str, Any]]) -> str:
  """The one-line NoCapacityError message: the zones tried, by what and why.

  Args:
    tried: the candidates tried, with what was tried and why it failed.

  Returns:
    the message.
  """
  groups = {}  # (what, reason) -> zones, in the order first tried
  for t in tried:
    key = (
        t.get("what") or t.get("candidate") or "machine",
        t.get("reason") or "capacity",
    )
    zones = groups.setdefault(key, [])
    if t.get("zone") and t["zone"] not in zones:
      zones.append(t["zone"])
  if not groups:
    return "no capacity: no candidate was tried"
  return "; ".join(
      f"no {what} {reason}" + (f" in {', '.join(zones)}" if zones else "")
      for (what, reason), zones in groups.items()
  )


@dataclasses.dataclass(frozen=True)
class RunSpec:
  """One run: its identity and what its trainer runs.

  Attributes:
    run_id: lower-case letters, digits and '-', starting with a letter, at
      most 40 characters (it names machines and storage).
    train_args: flags of the trainer, without --checkpoint_dir (the
      supervisor sets it to the run's storage).
    env: extra environment of the trainer.
    plugins: plugin modules of the trainer (passed in plugins.ENV).
  """

  run_id: str
  train_args: tuple[str, ...] = ()
  env: Mapping[str, str] = dataclasses.field(default_factory=dict)
  plugins: tuple[str, ...] = ()

  def __post_init__(self):
    if not isinstance(self.run_id, str) or not RUN_ID.match(self.run_id):
      raise ValueError(f"run_id {self.run_id!r}: want [a-z][a-z0-9-]{{0,39}}")
    object.__setattr__(
        self, "train_args", tuple(str(a) for a in self.train_args)
    )
    object.__setattr__(self, "plugins", tuple(str(p) for p in self.plugins))
    for a in self.train_args:
      if a.startswith("--checkpoint_dir"):
        raise ValueError(
            "train_args may not set --checkpoint_dir: the run's storage holds "
            "it"
        )
    for k, v in dict(self.env).items():
      if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(k)) or not isinstance(
          v, str
      ):
        raise ValueError(f"env {k!r}={v!r}: want NAME=string")

  def record(self) -> dict[str, Any]:
    """The JSON form the supervisor stores and compares.

    Returns:
      the record.
    """
    return {
        "run_id": self.run_id,
        "train_args": list(self.train_args),
        "env": dict(sorted(dict(self.env).items())),
        "plugins": list(self.plugins),
    }

  def trainer_argv(
      self, run_uri: str, python: str | Sequence[str] | None = None
  ) -> list[str]:
    """The trainer's argv, with --checkpoint_dir=<run_uri>/ckpt.

    Args:
      run_uri: the run's storage.
      python: the interpreter (a path or a command); None is this machine's
        (subprocess_env.python_command).

    Returns:
      The argv, interpreter first.
    """
    return [
        *subprocess_env.python_command(python),
        "-u",
        "-m",
        TRAINER_MODULE,
        *self.train_args,
        f"--checkpoint_dir={join(run_uri, CKPT_DIR)}",
    ]

  def trainer_env(self) -> dict[str, str]:
    """The trainer's extra environment, plugins.ENV included.

    Returns:
      the environment.
    """
    env = dict(self.env)
    if self.plugins:
      env[plugins_lib.ENV] = ",".join(self.plugins)
    return env

  def trainer_shell(
      self,
      run_uri: str,
      python: str,
      extra_env: Optional[Mapping[str, str]] = None,
  ) -> str:
    """The trainer as one shell command line, environment assignments first.

    extra_env: what the machine adds, for example the platform it must see.

    Args:
      run_uri: the run's storage root.
      python: the python interpreter.
      extra_env: what the machine adds, for example the platform it must see.

    Returns:
      the command line.
    """
    env = " ".join(
        f"{k}={shlex.quote(v)}"
        for k, v in sorted(
            {**self.trainer_env(), **dict(extra_env or {})}.items()
        )
    )
    cmd = " ".join(shlex.quote(a) for a in self.trainer_argv(run_uri, python))
    return f"{env} {cmd}".strip()


@dataclasses.dataclass(frozen=True)
class Status:
  """An attempt's state (STATES), its exit code when known, and a detail.

  Attributes:
    state: one of STATES.
    rc: the exit code, when known.
    detail: a human-readable detail.
  """

  state: str
  rc: Optional[int] = None
  detail: str = ""

  def __post_init__(self):
    assert self.state in STATES, self.state

  @property
  def terminal(self) -> bool:
    """Whether the attempt has ended.

    Returns:
      True unless RUNNING.
    """
    return self.state != RUNNING


def classify_exit(rc: int) -> str:
  """A trainer's exit code as a state.

  A trainer's exit code as a state, the same whoever reports it (a
  subprocess's negative signal code or a shell's 128 + signal).

  0 is SUCCEEDED. PREEMPTED_EXIT (the trainer checkpointed on a preemption
  notice) or death by SIGTERM (-15; 143 from a shell) is PREEMPTED. Anything
  else is FAILED, a SIGKILL (-9 / 137: the out-of-memory killer, or a kill
  that gave the trainer no chance to checkpoint) included, so it is not
  relaunched again and again.

  Args:
    rc: the exit code.

  Returns:
    SUCCEEDED, PREEMPTED or FAILED.
  """
  rc = int(rc)
  if rc == 0:
    return SUCCEEDED
  if rc in (preemption.PREEMPTED_EXIT, -int(signal.SIGTERM)):
    return PREEMPTED
  return FAILED


join = storage_lib.join


class Backend:
  """The interface (module doc).

  Subclasses set `storage` and implement run_uri, prepare, provision, launch,
  poll and teardown; plan, abort and adopt have defaults.

  Attributes:
    name: the backend's name (a handle's "backend").
    storage: where runs keep their state, checkpoints and logs.
  """

  name = "backend"
  storage: storage_lib.Storage

  def run_uri(self, run_id: str) -> str:
    """The run's storage root (state, checkpoints, logs).

    Args:
      run_id: the run.

    Returns:
      the run's storage root URI.
    """
    raise NotImplementedError

  def read_text(self, uri: str) -> Optional[str]:
    """The object's text, or None when it does not exist.

    Args:
      uri: the object.

    Returns:
      its text, or None.
    """
    return self.storage.read_text(uri)

  def write_text(self, uri: str, text: str) -> None:
    """Writes the object's text.

    Args:
      uri: the object.
      text: its text.
    """
    self.storage.write_text(uri, text)

  def exists(self, uri: str) -> bool:
    """Whether the object exists.

    Args:
      uri: the object.

    Returns:
      True when it exists.
    """
    return self.storage.exists(uri)

  def prepare(self, run: RunSpec) -> None:
    """Stages what attempts need (code, inputs); idempotent.

    Args:
      run: the run.
    """
    raise NotImplementedError

  def plan(self, run: RunSpec, attempt: int) -> str:
    """What attempt `attempt` of `run` would run, as text.

    Creates nothing. The default is the trainer's command line on this machine.

    Args:
      run: the run.
      attempt: the attempt number.

    Returns:
      the plan.
    """
    del attempt
    return " ".join(run.trainer_argv(self.run_uri(run.run_id)))

  def shipped_packages(self) -> Optional[tuple[str, ...]]:
    """The import names of the code packages this backend ships, or None.

    The import names of the code packages this backend ships to the
    machine, or None when the trainer runs in this machine's own environment
    (the default). The launcher checks the run's game against them before any
    machine exists (launch.preflight).

    Returns:
      the import names, or None.
    """
    return None

  def provision(self, run: RunSpec, attempt: int) -> dict[str, Any]:
    """A machine for an attempt (module doc).

    Args:
      run: the run.
      attempt: the attempt number.

    Returns:
      the attempt's handle.
    """
    raise NotImplementedError

  def launch(
      self, handle: dict[str, Any], run: RunSpec, attempt: int
  ) -> dict[str, Any]:
    """Starts the trainer on the provisioned machine (module doc).

    Args:
      handle: provision's handle.
      run: the run.
      attempt: the attempt number.

    Returns:
      the attempt's handle.
    """
    raise NotImplementedError

  def poll(self, handle: dict[str, Any]) -> Status:
    """The attempt's Status (module doc).

    Args:
      handle: the attempt's handle.

    Returns:
      the attempt's Status.
    """
    raise NotImplementedError

  def teardown(self, handle: dict[str, Any]) -> None:
    """Releases the machine; idempotent (module doc).

    Args:
      handle: the attempt's handle.
    """
    raise NotImplementedError

  def abort(self, handle: dict[str, Any], exc: BaseException) -> None:
    """The supervisor raised exc during the attempt (module doc).

    Args:
      handle: the attempt's handle.
      exc: the exception.
    """
    del exc
    self.teardown(handle)

  # The default returns None; Returns documents what overrides return.
  # pylint: disable-next=redundant-returns-doc
  def adopt(self, handle: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Re-attaches to a previous supervisor's attempt (module doc).

    The default cannot.

    Args:
      handle: the previous attempt's handle.

    Returns:
      the adopted attempt's handle, or None when it cannot be adopted.
    """
    del handle
    return None


def check_handle(handle: Any) -> dict[str, Any]:
  """A handle is a JSON object naming its backend and attempt.

  Args:
    handle: a handle.

  Returns:
    the handle as a dict.

  Raises:
    TypeError: it is not such an object.
  """
  if (
      not isinstance(handle, Mapping)
      or "backend" not in handle
      or "attempt" not in handle
  ):
    raise TypeError(
        f"a handle is a JSON object with backend and attempt, got {handle!r}"
    )
  return dict(handle)
