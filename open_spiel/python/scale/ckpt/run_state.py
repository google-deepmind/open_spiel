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

"""Run checkpoints: the commit record and the learner state of a training run.

A preempted run must resume exactly where it committed, and a half-written
step must never be mistaken for a committed one. A run directory therefore
holds its per-step files first and the commit record last:

  learner_state/<step>    the learner's state_dict, written by its framework's
                          StateCodec (ckpt.codecs)
  train_state.json        the commit record, written last and atomically: its
                          record_version, the learner kind, the step,
                          counters, resume_hash, config_hash and the config

A run resumes from the step train_state.json names; files of a later step
without a record naming them were never committed.

  run_state.save(ckpt_dir, step, learner, state)
  run_state.write_train_state(ckpt_dir, record)
  step = run_state.check_resume(ckpt_dir, run_state.read_train_state(ckpt_dir),
                                kind, resume_hash, run_length)
  state = run_state.restore(ckpt_dir, step, learner)

A record carries `record_version` (RECORD_VERSION). A reader refuses any
other version unless a migration for it is registered
(`register_record_migration`), which rewrites the record into the current
version before anything reads it.

`verify` compares learner state already on disk with a live one without
restoring the file into the learner, so a host learner whose state is a live
object is never overwritten by the comparison.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional

from etils import epath

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.ckpt import layout
from open_spiel.python.scale.ckpt import records
from open_spiel.python.scale.ckpt import snapshot
from open_spiel.python.scale.core import x64

TRAIN_STATE = layout.TRAIN_STATE
LEARNER_DIR = "learner_state"
METRICS_DIR = "metrics"
RECORD_KEYS = ("kind", "step", "resume_hash", "config_hash", "config")
RECORD_VERSION = 1
VERSION_KEY = "record_version"

_MIGRATIONS: dict[Optional[int], Callable[[dict[str, Any]], dict[str, Any]]] = (
    {}
)


def register_record_migration(
    version: Optional[int], fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
  """Makes `read_train_state` accept records of `version`.

  None stands for records with no `record_version`. Each is rewritten
  through fn(record) -> record, which must return a record of RECORD_VERSION.
  One migration per version; registering the same function again is a no-op.

  Args:
    version: the record version migrated from, or None.
    fn: the migration.

  Raises:
    TypeError: fn is not callable.
    ValueError: `version` is the current version, or already has another
      migration.
  """
  if not callable(fn):
    raise TypeError(f"a record migration is callable, got {fn!r}")
  if version == RECORD_VERSION:
    raise ValueError(
        f"record_version {version} is current and needs no migration"
    )
  if _MIGRATIONS.get(version, fn) is not fn:
    raise ValueError(f"record_version {version!r} already has a migration")
  _MIGRATIONS[version] = fn


def read_train_state(ckpt_dir: str) -> Optional[dict[str, Any]]:
  """The commit record of `ckpt_dir`, or None when nothing is committed.

  Args:
    ckpt_dir: the run's checkpoint dir.

  Returns:
    The record at RECORD_VERSION, or None.

  Raises:
    ValueError: the record's version is neither RECORD_VERSION nor one with a
      registered migration, or a migration did not produce RECORD_VERSION.
  """
  path = epath.Path(ckpt_dir) / TRAIN_STATE
  if not path.exists():
    return None
  record = records.decode(json.loads(path.read_text()))
  version = record.get(VERSION_KEY)
  if version != RECORD_VERSION:
    if version not in _MIGRATIONS:
      raise ValueError(
          f"{path}: record_version {version!r}; this reader reads "
          f"{RECORD_VERSION}"
      )
    record = _MIGRATIONS[version](dict(record))
    if record.get(VERSION_KEY) != RECORD_VERSION:
      raise ValueError(
          f"{path}: the migration of record_version {version!r} returned "
          f"record_version {record.get(VERSION_KEY)!r}, not {RECORD_VERSION}"
      )
  return record


def write_train_state(ckpt_dir: str, record: Mapping[str, Any]) -> None:
  """Commits `record` (atomically), stamped with RECORD_VERSION.

  Args:
    ckpt_dir: the run's checkpoint dir.
    record: the commit record.

  Raises:
    ValueError: a required key is missing, or the record names another version.
  """
  missing = [k for k in RECORD_KEYS if k not in record]
  if missing:
    raise ValueError(f"a train_state record needs {missing}")
  if record.get(VERSION_KEY, RECORD_VERSION) != RECORD_VERSION:
    raise ValueError(
        f"a train_state record of record_version {record[VERSION_KEY]!r} "
        f"cannot be written; this writer writes {RECORD_VERSION}"
    )
  record = dict(record, **{VERSION_KEY: RECORD_VERSION})
  artifact.atomic_write(
      epath.Path(ckpt_dir) / TRAIN_STATE,
      json.dumps(records.encode(record), sort_keys=True, indent=1).encode(),
  )


def check_resume(
    ckpt_dir: str,
    ts: Mapping[str, Any],
    kind: str,
    resume_hash: str,
    run_length: int,
) -> int:
  """The committed step to resume from.

  Args:
    ckpt_dir: the run's checkpoint dir.
    ts: the train_state record.
    kind: this run's learner kind.
    resume_hash: this run's resume hash.
    run_length: this run's length.

  Returns:
    The committed step.

  Raises:
    ValueError: unless `ts` is a record of learner kind `kind` with this
      resume_hash and a step no later than `run_length`.
  """
  if "kind" not in ts:
    raise ValueError(
        f"{ckpt_dir}: the train_state record names no learner kind"
    )
  if ts["kind"] != kind:
    raise ValueError(f"{ckpt_dir} holds a {ts['kind']!r} run, not {kind!r}")
  if ts["resume_hash"] != resume_hash:
    raise ValueError(
        f"{ckpt_dir} holds a run of a different config (resume_hash "
        f"{ts['resume_hash'][:12]} != {resume_hash[:12]})"
    )
  step = int(ts["step"])
  if step > run_length:
    raise ValueError(f"checkpoint step {step} > the run's length {run_length}")
  return step


def learner_framework(learner: Any) -> str:
  """The framework whose StateCodec saves `learner`'s state.

  The learner's `framework` attribute when it declares one, else its policy's
  PolicySpec.framework (core.types.Learner).

  Args:
    learner: the learner.

  Returns:
    The framework name.

  Raises:
    TypeError: the learner declares neither.
  """
  fw = getattr(learner, "framework", None)
  if fw is None:
    policy = getattr(learner, "policy", None)
    if policy is None:
      raise TypeError(
          f"{type(learner).__name__} declares no framework: give it a "
          "`framework` attribute or a policy() whose spec names one"
      )
    fw = policy().spec.framework
  return str(fw)


def _codec(learner: Any) -> codecs.StateCodec:
  """The state codec of `learner`'s framework.

  Args:
    learner: The learner.

  Returns:
    The codec.
  """
  return codecs.state_codec(learner_framework(learner))


def learner_state_path(ckpt_dir: str, step: int, learner: Any) -> epath.Path:
  """The path of the learner state saved at `step`.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    learner: the learner; its codec sets the suffix.

  Returns:
    The path.
  """
  return (
      epath.Path(ckpt_dir)
      / LEARNER_DIR
      / f"{int(step)}{_codec(learner).suffix}"
  )


def save(ckpt_dir: str, step: int, learner: Any, state: Any) -> None:
  """Writes the learner state of `step` (write-once).

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    learner: the learner.
    state: the learner state.

  Raises:
    FileExistsError: a state is already saved at `step`.
  """
  path = learner_state_path(ckpt_dir, step, learner)
  if path.exists():
    raise FileExistsError(f"{path} exists (write-once)")
  _codec(learner).save(path, learner, state)


def restore(ckpt_dir: str, step: int, learner: Any) -> Any:
  """The learner state saved at `step`; stored dtypes must equal the learner's.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    learner: the learner.

  Returns:
    The learner state.

  Raises:
    FileNotFoundError: no state was saved at `step`.
    ArtifactError: a stored leaf's shape or dtype differs from the learner's.
  """
  x64.assert_x64("run_state.restore")
  path = learner_state_path(ckpt_dir, step, learner)
  if not path.exists():
    raise FileNotFoundError(f"no learner state at {path}")
  return _codec(learner).restore(path, learner)


def verify(ckpt_dir: str, step: int, learner: Any, state: Any) -> bool:
  """Whether the state on disk at `step` is bit for bit the given state.

  Compares with learner.state_dict(state). Reads the file on its own; never
  restores it into `learner`.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    learner: the learner.
    state: the learner state.

  Returns:
    True when the states are equal.

  Raises:
    FileNotFoundError: no state was saved at `step`.
  """
  x64.assert_x64("run_state.verify")
  path = learner_state_path(ckpt_dir, step, learner)
  if not path.exists():
    raise FileNotFoundError(f"no learner state at {path}")
  stored = _codec(learner).read_host(path, learner, state)
  return snapshot.equal(stored, snapshot.to_host(learner.state_dict(state)))
