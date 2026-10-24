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

"""league.runner's resume checkpoint and preemption stop.

league.runner's resume checkpoint and preemption stop, on a kuhn
snapshot-pool league.

  * a preemption notice ends the run after the iteration in flight with a
    resume checkpoint (no keep, no artifact), the result says preempted, and
    the next process continues it: every later update's metrics and the
    final params equal the uninterrupted run's bitwise (the checkpoint
    is a full League.save, ArenaSource hands included);
  * the timer: with resume_every_s and an injected clock, the iterations a
    resume checkpoint is written at are a known answer; each replaces the
    one before (one resume dir on disk), a keep commit drops it; keeps and
    their artifacts are exactly what they were without the timer;
  * evals are skipped once the notice is up (the checkpoint goes first);
  * a resume record of another config is refused at start.
"""

import dataclasses
import glob
import json
import os
import sys

import pytest

from open_spiel.python.scale import metrics as metrics_lib
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import resume as resume_ckpt
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.tests import league_fixtures


def _snapshot_pool_cfg(updates=8, keep_every=8):
  """A kuhn snapshot-pool run config.

  Rounds of 4 updates (iterations 5..7 inside a round: no admission, the
  hands carry on), one keep at the end unless keep_every says otherwise.

  Args:
    updates: the run's updates.
    keep_every: iterations between keeps.

  Returns:
    The run config.
  """
  pol = league_fixtures.kuhn_policy(16)
  learner = dataclasses.replace(
      league_fixtures.br_learner(pol).cfg, num_updates=updates
  )
  return runner.snapshot_pool(
      "snapshot_pool_kuhn_resume",
      "kuhn_poker",
      kinds.PG(learner),
      dict(pol.spec.arch_kwargs),
      num_envs=32,
      num_steps=8,
      num_updates=updates,
      snapshot_every=4,
      harvest_every=4,
      max_pool=2,
      pool_slots=1,
      self_weight=0.75,
      keep_every=keep_every,
  )


class _NoticeAt:
  """A preemption notice that goes up once the League ran `at` iterations.

  Attributes:
    up: whether the notice is up.
  """

  def __init__(self, monkeypatch, at):
    self.up = False
    real = loop.League.run

    def run(lg, n, *a, **kw):
      """Runs the league, raising the notice once past `at`.

      Args:
        lg: the league.
        n: the number of iterations.
        *a: passed to the real run.
        **kw: passed to the real run.

      Returns:
        The real run's result.
      """
      out = real(lg, n, *a, **kw)
      if lg.iteration >= at:
        self.up = True
      return out

    monkeypatch.setattr(loop.League, "run", run)

  def __call__(self):
    return self.up


def _league_dirs(d):
  """The league commit folders of run `d`.

  Args:
    d: The run folder.

  Returns:
    The sorted relative paths.
  """
  return sorted(
      os.path.relpath(p, d) for p in glob.glob(f"{d}/{commit.COMMITS_DIR}/*-*")
  )


def test_preemption_writes_a_resume_checkpoint_and_the_resume_is_bit_for_bit(
    tmp_path, monkeypatch
):
  """Preemption writes a resume checkpoint and the resume is bit for bit."""
  full = runner.run(_snapshot_pool_cfg(), str(tmp_path / "full"))
  assert not full.preempted
  d = str(tmp_path / "cut")
  notice = _NoticeAt(monkeypatch, at=6)
  part = runner.run(_snapshot_pool_cfg(), d, preempted=notice)
  monkeypatch.undo()
  assert (
      part.preempted and part.league.iteration == 6 and len(part.metrics) == 6
  )
  assert run_state.read_train_state(d) is None  # no keep yet (keep_every 8)
  rs = resume_ckpt.read(d)
  assert (
      rs["step"] == 6
      and rs["reason"] == "preemption"
      and rs["keep_step"] is None
  )
  assert rs["league_ref"] == commit.ref(6, part.league.attempt)
  assert not artifact.exists(str(tmp_path / "cut"))  # nothing published for it
  res = runner.run(_snapshot_pool_cfg(), d)
  assert not res.preempted and len(res.metrics) == 2  # updates 7, 8
  for got, want in zip(res.metrics, full.metrics[6:]):
    assert got == want, (got, want)
  assert league_fixtures.leaves_equal(
      res.league.learner_params("main"), full.league.learner_params("main")
  )
  ts = run_state.read_train_state(d)
  assert ts["step"] == 8 and ts["artifact_step"] == 8
  assert resume_ckpt.read(d) is None  # the keep dropped the record ...
  assert _league_dirs(d) == [ts["league_ref"]]  # ... and its dir


def test_timer_resume_checkpoints_known_answer(tmp_path, monkeypatch):
  """Timer-driven resume checkpoints land at known iterations.

  clock() is read once at the start and once after each iteration (t =
  100 s per iteration); every 150 s, keep_every 4: resume saves after
  iterations 2 and 6 (1, 3, 5, 7 are < 150 s after the last save; 4 and 8 are
  keeps, which reset the clock).

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  ticks = iter(range(0, 10_000, 100))
  saves = []
  real_write = resume_ckpt.write

  def spy(ckpt, rec):
    files = _league_dirs(ckpt)
    saves.append((int(rec["step"]), rec["reason"], rec["keep_step"], files))
    real_write(ckpt, rec)

  monkeypatch.setattr(resume_ckpt, "write", spy)
  d = str(tmp_path / "t")
  res = runner.run(
      _snapshot_pool_cfg(keep_every=4),
      d,
      resume_every_s=150.0,
      clock=lambda: next(ticks),
  )
  assert [(s, r, k) for s, r, k, _ in saves] == [
      (2, "timer", None),
      (6, "timer", 4),
  ]
  ts = run_state.read_train_state(d)
  assert ts["step"] == 8 and sorted(ts["league_commits"]) == ["4", "8"]
  # At the second save the first resume dir was already gone (dropped by keep
  # 4).
  assert not [f for f in saves[1][3] if f.startswith("league/2-")]
  assert resume_ckpt.read(d) is None
  assert _league_dirs(d) == sorted(ts["league_commits"].values())  # keeps only
  # The keeps are what they are without the timer: same artifacts, same params.
  plain = runner.run(_snapshot_pool_cfg(keep_every=4), str(tmp_path / "plain"))
  assert league_fixtures.leaves_equal(
      res.league.learner_params("main"), plain.league.learner_params("main")
  )
  for step in (4, 8):
    assert artifact.exists(str(tmp_path / "t"), step)
  assert not artifact.exists(str(tmp_path / "t"), 2)
  assert not artifact.exists(str(tmp_path / "t"), 6)


def test_a_newer_resume_checkpoint_replaces_the_older(tmp_path, monkeypatch):
  """A newer resume checkpoint replaces the older."""
  ticks = iter(range(0, 10_000, 100))
  d = str(tmp_path / "r")
  notice = _NoticeAt(monkeypatch, at=7)
  part = runner.run(
      _snapshot_pool_cfg(),
      d,
      resume_every_s=150.0,
      clock=lambda: next(ticks),
      preempted=notice,
  )
  monkeypatch.undo()
  # saves at 2 (timer), 4 (timer), 6 (timer), 7 (preemption): one dir left, 7's.
  assert part.preempted and resume_ckpt.read(d)["step"] == 7
  assert _league_dirs(d) == [commit.ref(7, part.league.attempt)]


def test_evals_are_skipped_once_the_notice_is_up(tmp_path, monkeypatch):
  """Evals are skipped once the preemption notice is up.

  The notice goes up during update 2 (at its train row, before its eval):
  the evals of updates 0 and 1 ran, update 2's is skipped, and the run ends
  after iteration 2 with a resume checkpoint.

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  calls, up = [], {"v": False}
  monkeypatch.setattr(
      runner, "build_evals", lambda cfg, env, mesh, run_dir: [(_Every1(), None)]
  )
  monkeypatch.setattr(
      runner,
      "_eval_row",
      lambda ev, evaluator, lg, ls, u, **kw: calls.append(u) or {},
  )
  real_write = metrics_lib.MetricsWriter.write

  def write(self, step, row, *a, **kw):
    """Raises the notice at update 2's train row, then writes.

    Args:
      self: the MetricsWriter.
      step: the metrics step.
      row: the metrics row.
      *a: passed to the real write.
      **kw: passed to the real write.

    Returns:
      The real write's result.
    """
    if kw.get("prefix") == "train" and int(step) == 2:
      up["v"] = True
    return real_write(self, step, row, *a, **kw)

  monkeypatch.setattr(metrics_lib.MetricsWriter, "write", write)
  d = str(tmp_path / "e")
  res = runner.run(_snapshot_pool_cfg(), d, preempted=lambda: up["v"])
  assert calls == [0, 1], calls
  assert res.preempted and resume_ckpt.read(d)["step"] == 2


class _Every1:
  every = 1
  key = "stub"


def test_a_resume_record_of_another_config_is_refused(tmp_path, monkeypatch):
  """A resume record of another config is refused."""
  d = str(tmp_path / "x")
  notice = _NoticeAt(monkeypatch, at=2)
  runner.run(_snapshot_pool_cfg(), d, preempted=notice)
  monkeypatch.undo()
  rec = resume_ckpt.read(d)
  resume_ckpt.write(d, dict(rec, resume_hash="0" * 64))
  with pytest.raises(ValueError, match="different config"):
    runner.run(_snapshot_pool_cfg(), d)
  assert (
      json.loads((tmp_path / "x" / resume_ckpt.RESUME_STATE).read_text())[
          "step"
      ]
      == 2
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
