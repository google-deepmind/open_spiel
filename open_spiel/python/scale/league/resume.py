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

"""The overwritten resume checkpoint of a league run.

A league keeps immutable checkpoints (keeps) only every `keep_every`
iterations, which can be hours apart. A preempted run resumes from this
record instead: a `League.save` taken between keeps (on a timer, or when the
preemption notice is up), recorded in one file that each new resume
checkpoint overwrites atomically, so a run loses at most the work since the
last one. No artifact is published for it.

    record = resume.read(ckpt_dir)                    # None if there is none
    start = resume.choose(train_state, record, resume_hash, num_iterations)
    resume.write(ckpt_dir, record)                    # after League.save
    resume.drop(ckpt_dir, keep_refs)                  # a keep supersedes it

Files:
  <ckpt>/league/<step>-<attempt>/   League.save(release_retired=False): the
                                    same state a keep holds (league.commit)
  <ckpt>/resume_state.json          the record: kind "league-resume", step,
                                    league_ref (the dir above), keep_step (the
                                    keep it was written over, None before the
                                    first), reason ("timer" or "preemption"),
                                    resume_hash, config_hash, the run's
                                    identity and counters, written_at.

Invariants: after a newer record is written the previous one's dir is
removed (`prune`); a keep clears the record, then its dir (`drop`), so the
record never names a removed dir. A process starts from the record only when
it is past the dir's keep (`choose`); a record at or before the keep is stale
and ignored; one of another config, past the run's length, or written over
another keep raises.
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Iterable, Mapping, Optional

from absl import logging
from etils import epath

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import records

RESUME_STATE = "resume_state.json"
KIND = "league-resume"
REASONS = ("timer", "preemption")
COMMITS_DIR = "league"  # league.commit.COMMITS_DIR (not imported: no cycle)
REQUIRED = (
    "kind",
    "step",
    "league_ref",
    "keep_step",
    "reason",
    "resume_hash",
    "config_hash",
    "attempts",
)


def utc_now() -> str:
  """The current UTC time, as the record's written_at.

  Returns:
    An ISO-8601 UTC timestamp.
  """
  return datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ"
  )


def _ref_step(ref: Any) -> int:
  """The step of a league/<step>-<attempt> ref; ValueError for anything else.

  Args:
    ref: The ref.

  Returns:
    The step.

  Raises:
    ValueError: `ref` is not a league commit ref.
  """
  if not isinstance(ref, str):
    raise ValueError(f"not a league commit ref: {ref!r}")
  head, sep, tail = ref.partition("/")
  step, dash, attempt = tail.partition("-")
  if (
      head != COMMITS_DIR
      or not sep
      or not dash
      or not step.isdigit()
      or not attempt
      or "/" in attempt
      or ".." in attempt
  ):
    raise ValueError(f"not a league commit ref: {ref!r}")
  return int(step)


def check(record: Mapping[str, Any]) -> None:
  """Validates a resume record.

  Args:
    record: the resume record.

  Raises:
    ValueError: a required key is missing or a field is invalid.
  """
  missing = [k for k in REQUIRED if k not in record]
  if missing:
    raise ValueError(f"a resume record needs {missing}")
  if record["kind"] != KIND:
    raise ValueError(f"resume record kind {record['kind']!r} != {KIND!r}")
  if record["reason"] not in REASONS:
    raise ValueError(
        f"resume record reason {record['reason']!r} not in {REASONS}"
    )
  step = record["step"]
  if isinstance(step, bool) or not isinstance(step, int) or step < 1:
    raise ValueError(f"resume record step {step!r}: an iteration count >= 1")
  if _ref_step(record["league_ref"]) != step:
    raise ValueError(
        f"resume record league_ref {record['league_ref']!r} is not a "
        "checkpoint "
        f"of step {step}"
    )
  ks = record["keep_step"]
  if ks is not None and (
      isinstance(ks, bool) or not isinstance(ks, int) or ks >= step
  ):
    raise ValueError(
        f"resume record keep_step {ks!r}: None or a keep before step {step}"
    )


def read(ckpt_dir: str) -> Optional[dict[str, Any]]:
  """The resume record of `ckpt_dir`, validated, or None.

  Args:
    ckpt_dir: the run's checkpoint dir.

  Returns:
    The record, or None when there is none.
  """
  p = epath.Path(ckpt_dir) / RESUME_STATE
  if not p.exists():
    return None
  rec = records.decode(json.loads(p.read_text()))
  check(rec)
  return rec


def write(ckpt_dir: str, record: Mapping[str, Any]) -> None:
  """Writes the resume record of `ckpt_dir` atomically, after validating it.

  Args:
    ckpt_dir: the run's checkpoint dir.
    record: the resume record.
  """
  check(record)
  artifact.atomic_write(
      epath.Path(ckpt_dir) / RESUME_STATE,
      json.dumps(records.encode(record), sort_keys=True, indent=1).encode(),
  )


def clear(ckpt_dir: str) -> None:
  """Removes the resume record of `ckpt_dir`, if any.

  Args:
    ckpt_dir: the run's checkpoint dir.
  """
  p = epath.Path(ckpt_dir) / RESUME_STATE
  if p.exists():
    p.unlink()


def choose(
    ts: Optional[Mapping[str, Any]],
    rs: Optional[Mapping[str, Any]],
    resume_hash: str,
    run_length: int,
) -> Optional[Mapping[str, Any]]:
  """The resume record a process starts from, or None.

  None means start from the keep ts, or fresh. The module doc lists the cases.

  Args:
    ts: the train_state.json record, or None.
    rs: the resume_state.json record, or None.
    resume_hash: this run's resume hash.
    run_length: this run's length.

  Returns:
    rs when the process resumes from it, else None.

  Raises:
    ValueError: rs is invalid or belongs to another run.
  """
  if rs is None:
    return None
  check(rs)
  keep = None if ts is None else int(ts["step"])
  if keep is not None and int(rs["step"]) <= keep:
    logging.info(
        "league.resume: resume record at step %d is not past the keep at %d: "
        "stale, "
        "ignored",
        int(rs["step"]),
        keep,
    )
    return None
  if rs["resume_hash"] != resume_hash:
    raise ValueError(
        f"the resume record (step {rs['step']}) is of a different config "
        f"(resume_hash {str(rs['resume_hash'])[:12]} != {resume_hash[:12]})"
    )
  if int(rs["step"]) > int(run_length):
    raise ValueError(
        f"resume record step {rs['step']} > the run's length {run_length}"
    )
  if rs["keep_step"] != keep:
    raise ValueError(
        f"the resume record (step {rs['step']}) was written over the keep at "
        f"{rs['keep_step']}, but the dir's keep is {keep}"
    )
  return rs


def prune(ckpt_dir: str, ref: str, keep_refs: Iterable[str]) -> bool:
  """Removes the resume checkpoint dir <ckpt>/<ref> unless a keep commits it.

  Args:
    ckpt_dir: the run's checkpoint dir.
    ref: the resume checkpoint's ref.
    keep_refs: the refs the keeps commit.

  Returns:
    True if something was removed.
  """
  _ref_step(ref)
  if ref in set(keep_refs):
    return False
  p = epath.Path(ckpt_dir) / ref
  if not p.exists():
    return False
  p.rmtree()
  logging.info(
      "league.resume: removed the superseded resume checkpoint %s/%s",
      ckpt_dir,
      ref,
  )
  return True


def drop(ckpt_dir: str, keep_refs: Iterable[str]) -> Optional[str]:
  """After a keep commit: clears the resume record, then prunes its dir.

  In that order, so the record never names a removed dir.

  Args:
    ckpt_dir: the run's checkpoint dir.
    keep_refs: the refs the keeps commit.

  Returns:
    The dropped ref, or None.
  """
  rs = read(ckpt_dir)
  if rs is None:
    return None
  clear(ckpt_dir)
  prune(ckpt_dir, rs["league_ref"], keep_refs)
  return rs["league_ref"]
