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

"""league.resume: the resume checkpoint's record.

league.resume: the overwritten resume checkpoint's record: its validation,
which record a run resumes from (known answers over every case of
train_state.json x resume_state.json), and the prune that never removes a
keep.
"""

import json
import os
import sys

import pytest

from open_spiel.python.scale.league import resume


def _rec(step, keep_step=None, ref=None, resume_hash="h", **kw):
  """A resume record.

  Args:
    step: The step.
    keep_step: The keep step.
    ref: The league ref (default from step).
    resume_hash: The resume hash.
    **kw: Overrides.

  Returns:
    The record.
  """
  rec = dict(
      kind=resume.KIND,
      step=step,
      league_ref=ref or f"league/{step}-a1",
      keep_step=keep_step,
      reason="timer",
      resume_hash=resume_hash,
      config_hash="c",
      attempts=[],
      updates=step,
      written_at=resume.utc_now(),
  )
  rec.update(kw)
  return rec


def test_write_read_clear_round_trip(tmp_path):
  """Write read clear round trip."""
  d = str(tmp_path)
  assert resume.read(d) is None
  resume.write(d, _rec(3))
  got = resume.read(d)
  assert got["step"] == 3 and got["league_ref"] == "league/3-a1"
  resume.write(d, _rec(5, ref="league/5-a1"))  # overwritten, one file
  assert resume.read(d)["step"] == 5 and os.listdir(d) == [resume.RESUME_STATE]
  resume.clear(d)
  assert resume.read(d) is None
  resume.clear(d)  # nothing to clear: a no-op


def test_write_validates_the_record(tmp_path):
  """Write validates the record."""
  d = str(tmp_path)
  for bad, msg in (
      (dict(kind="league"), "kind"),
      (dict(reason="whim"), "reason"),
      (dict(league_ref="params/3"), "ref"),
      (dict(league_ref="league/4-a1"), "step"),
      (dict(step=0), "step"),
  ):
    with pytest.raises(ValueError, match=msg):
      resume.write(
          d,
          _rec(3, **{k: v for k, v in bad.items() if k != "step"})
          if "step" not in bad
          else _rec(bad["step"]),
      )
  rec = _rec(3)
  del rec["resume_hash"]
  with pytest.raises(ValueError, match="resume_hash"):
    resume.write(d, rec)


def test_choose_known_answers():
  """Choose known answers."""
  ts4 = {"step": 4, "resume_hash": "h"}
  # No resume record: the keep (or a fresh start).
  assert resume.choose(None, None, "h", 8) is None
  assert resume.choose(ts4, None, "h", 8) is None
  # A resume record past the keep it was written over: resume from it.
  assert resume.choose(ts4, _rec(6, keep_step=4), "h", 8)["step"] == 6
  # Before any keep (a run preempted before its first keep commit).
  assert resume.choose(None, _rec(2), "h", 8)["step"] == 2
  # Stale: a keep at or after it was committed (the clear never ran): ignored.
  assert resume.choose(ts4, _rec(3), "h", 8) is None
  assert (
      resume.choose(
          {"step": 8, "resume_hash": "h"}, _rec(6, keep_step=4), "h", 8
      )
      is None
  )
  # Another config: refused, never silently ignored.
  with pytest.raises(ValueError, match="config"):
    resume.choose(ts4, _rec(6, keep_step=4, resume_hash="other"), "h", 8)
  # Past the run's length.
  with pytest.raises(ValueError, match="length"):
    resume.choose(None, _rec(9), "h", 8)
  # Written over another keep than the dir's: train_state.json moved under it.
  with pytest.raises(ValueError, match="keep"):
    resume.choose(ts4, _rec(6, keep_step=2), "h", 8)
  with pytest.raises(ValueError, match="keep"):
    resume.choose(None, _rec(6, keep_step=4), "h", 8)


def test_prune_removes_a_resume_dir_and_never_a_keep(tmp_path):
  """Prune removes a resume dir and never a keep."""
  d = tmp_path
  for r in ("league/3-a1", "league/4-a1"):
    (d / r).mkdir(parents=True)
    (d / r / "state.pkl").write_bytes(b"x")
  assert resume.prune(str(d), "league/3-a1", keep_refs=["league/4-a1"])
  assert not (d / "league/3-a1").exists() and (d / "league/4-a1").exists()
  # a keep: kept
  assert not resume.prune(str(d), "league/4-a1", keep_refs=["league/4-a1"])
  assert (d / "league/4-a1" / "state.pkl").exists()
  # already gone
  assert not resume.prune(str(d), "league/3-a1", keep_refs=[])
  with pytest.raises(ValueError, match="league commit ref"):
    resume.prune(str(d), "params/3", keep_refs=[])
  with pytest.raises(ValueError, match="league commit ref"):
    resume.prune(str(d), "league/../x", keep_refs=[])


def test_drop_clears_then_prunes(tmp_path):
  """Drop clears then prunes."""
  d = tmp_path
  (d / "league/6-a1").mkdir(parents=True)
  resume.write(str(d), _rec(6, keep_step=4, ref="league/6-a1"))
  assert resume.drop(str(d), keep_refs=["league/4-a0"]) == "league/6-a1"
  assert resume.read(str(d)) is None and not (d / "league/6-a1").exists()
  assert resume.drop(str(d), keep_refs=[]) is None  # nothing to drop
  assert json.loads(json.dumps(resume.REASONS)) == ["timer", "preemption"]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
