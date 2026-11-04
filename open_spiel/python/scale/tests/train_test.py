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

"""The trainer (train).

a run writes per-update metrics and checkpoints, counts frames in int64, resumes
only an unchanged config, stops early with --stop_after as a prefix of the full
run, and records its run id; the command-line flags of the entry point, a
learner kind's own flags and the preemption sources.
"""

import dataclasses
import json
import sys

from absl import app
from absl import flags
from absl.testing import flagsaver
import jax
import numpy as np
import pytest

from open_spiel.python.scale import metrics
from open_spiel.python.scale import train
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.examples import play_checkpoints
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.learners.tests import pg_fixtures as fx

PAIR = "train/main/"  # the one learner slot's metrics


def _tiny(name="mmd_kuhn_poker_jax", updates=3):
  """A tiny run config.

  Args:
    name: The config.
    updates: The updates.

  Returns:
    The config.
  """
  cfg = config_common.with_overrides(
      config_common.load_config(name),
      num_updates=updates,
      num_envs=8,
      num_steps=8,
  )
  return dataclasses.replace(cfg, keep_every=2)


def _rows(path):
  """The metric rows of the learner pair.

  Args:
    path: The metrics file.

  Returns:
    The rows.
  """
  return [r for r in metrics.read_jsonl(str(path)) if PAIR + "frames" in r]


def _learner_state(ck, step, learner):
  """The learner state committed at `step`.

  Args:
    ck: The run folder.
    step: The step.
    learner: The learner.

  Returns:
    The restored state.
  """
  return run_state.restore(commit.learner_dir(ck, "main", step), step, learner)


def test_run_writes_metrics_and_checkpoints_and_resumes(tmp_path):
  """train.run writes metrics and checkpoints, and resumes."""
  ck = str(tmp_path / "ck")
  res = train.run(_tiny(updates=3), checkpoint_dir=ck)
  assert int(res.state.update_step) == 3
  rows = _rows(tmp_path / "ck" / "metrics" / metrics.JSONL_NAME)
  assert [r["step"] for r in rows] == [1, 2, 3]
  assert rows[-1][PAIR + "frames"] == 3 * 8 * 8
  assert isinstance(rows[-1][PAIR + "frames"], int)
  for step in (2, 3):  # keep_every=2, and the final update
    art = artifact.load_artifact(ck, step)
    assert art.manifest.step == step and art.manifest.arch_name == "network"
  art = artifact.load_artifact(ck, 3)
  assert fx.leaves_equal(art.params, res.state.params)
  ts = run_state.read_train_state(ck)
  assert (
      ts["kind"] == "league" and ts["learner_kind"] == "pg" and ts["step"] == 3
  )
  assert ts["artifact_step"] == 3
  # Resume to 5 updates: continues from step 3 with the restored learner state.
  res2 = train.run(_tiny(updates=5), checkpoint_dir=ck)
  assert int(res2.state.update_step) == 5
  rows = _rows(tmp_path / "ck" / "metrics" / metrics.JSONL_NAME)
  assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
  assert rows[-1][PAIR + "frames"] == 5 * 8 * 8
  blob = _learner_state(ck, 3, res2.learner)
  assert int(blob.update_step) == 3 and blob.update_step.dtype == np.int64
  assert fx.leaves_equal(blob.params, res.state.params)


def test_resume_is_refused_for_a_different_config(tmp_path):
  """Resuming a checkpoint dir with another config is refused."""
  ck = str(tmp_path / "ck")
  train.run(_tiny(updates=2), checkpoint_dir=ck)
  other = config_common.with_overrides(_tiny(updates=4), seed=7)
  with pytest.raises(ValueError, match="config"):
    train.run(other, checkpoint_dir=ck)


def test_ppo_vs_uniform_runs():
  """A PPO run against a uniform opponent runs."""
  cfg = _tiny("ppo_tic_tac_toe", updates=2)
  assert [o.name for o in cfg.fixed_opponents] == ["uniform"]
  assert cfg.controller_kwargs == {
      "learner": "main",
      "self_play": 0.0,
      "weights": {"uniform": 1.0},
  }
  res = train.run(cfg, checkpoint_dir=None)
  assert int(res.state.update_step) == 2
  assert np.isfinite(res.metrics[-1]["loss_total"])
  assert res.league.learners["main"].draw_counts[-1] == {"uniform": 8}


def test_train_runs_league_configs_only():
  """train.run refuses anything but a league run config."""
  with pytest.raises(TypeError, match="league"):
    train.run(object())


def test_an_example_that_is_also_a_library_imports_beside_the_trainer():
  """examples.play_checkpoints imports beside the trainer.

  It is also a library (examples_test reads saved games with it): importing it
  where the trainer's flags exist must not define a flag twice (both take
  --seed); its flags exist only when it runs as a script.
  """
  assert "seed" in flags.FLAGS and "verify_pyspiel" not in flags.FLAGS
  assert callable(play_checkpoints.read_trajectories)


def test_main_flags(tmp_path):
  """train.main trains the config the flags name and records its hardware."""
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
  ck = str(tmp_path / "ck")
  with flagsaver.flagsaver(
      config="mmd_kuhn_poker_jax",
      checkpoint_dir=ck,
      num_updates=1,
      num_envs=4,
      num_steps=4,
  ):
    train.main(["train"])
  assert artifact.load_artifact(ck, 1).manifest.step == 1
  # The hardware this process got, one row per start.
  hw = json.loads((tmp_path / "ck" / "hardware.json").read_text())["attempts"]
  assert (
      len(hw) == 1 and hw[0]["num_devices"] >= 1 and hw[0]["platform"] == "cpu"
  )
  # num_envs is global and must split over the devices the run shards over.
  if len(jax.devices()) > 1:
    with flagsaver.flagsaver(
        config="mmd_kuhn_poker_jax",
        checkpoint_dir=str(tmp_path / "ck2"),
        num_updates=1,
        num_envs=len(jax.devices()) + 1,
        num_steps=4,
    ):
      with pytest.raises(ValueError, match="do not split over"):
        train.main(["train"])


def test_provenance_and_learner_decisions(tmp_path):
  """Artifacts carry the run id and the commit.

  Metrics and train_state carry the learner-decision count beside frames.

  Args:
    tmp_path: pytest's temporary directory.
  """
  ck = str(tmp_path / "ck")
  res = train.run(
      _tiny(updates=2), checkpoint_dir=ck, run_id=777, git_commit="abc123"
  )
  art = artifact.load_artifact(ck, 2)
  assert art.manifest.run_id == "777" and art.manifest.git_commit == "abc123"
  rows = _rows(tmp_path / "ck" / "metrics" / metrics.JSONL_NAME)
  d = [r[PAIR + "learner_decisions"] for r in rows]
  assert all(isinstance(x, int) for x in d) and 0 < d[0] < d[1] <= 2 * 8 * 8 * 2
  ts = run_state.read_train_state(str(tmp_path / "ck"))
  assert (
      ts["decisions"] == d[-1]
      and ts["frames"] == 2 * 8 * 8
      and str(ts["run_id"]) == "777"
  )
  assert res.metrics[-1]["learner_decisions"] == d[-1]


def test_resume_after_preemption_does_not_duplicate_metrics(tmp_path):
  """Resuming after a preemption does not duplicate metric rows."""
  ck = str(tmp_path / "ck")
  train.run(_tiny(updates=3), checkpoint_dir=ck)  # checkpoints 2 and 3
  # Simulate a preemption after update 4 whose checkpoint never landed: the
  # metrics file has a row for 4, train_state says 3.
  path = tmp_path / "ck" / "metrics" / metrics.JSONL_NAME
  with open(path, "a", encoding="utf-8") as f:
    f.write('{"step": 4, "time": 0.0, "train/main/frames": -1}\n')
  train.run(_tiny(updates=5), checkpoint_dir=ck)
  rows = _rows(path)
  assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
  assert (
      rows[3][PAIR + "frames"] == 4 * 8 * 8
  )  # the resumed run's row, not the stale one


def test_stop_after_is_a_prefix_and_resumes_to_the_full_run(tmp_path):
  """--stop_after=N ends the run at N; resuming finishes the same run.

  The run's length and LR horizon are unchanged (a checkpoint at N): its updates
  are the uninterrupted run's; resuming finishes the run on the same LR schedule
  (a resume deals fresh hands, so later batches differ).

  Args:
    tmp_path: pytest's temporary directory.
  """
  cfg = _tiny(updates=4)
  pgc = dataclasses.replace(cfg.learner.learner, anneal_lr=True, lr_end=0.0)
  cfg = dataclasses.replace(
      cfg, learner=dataclasses.replace(cfg.learner, learner=pgc)
  )
  assert (
      not cfg.learner.constant_lr()
  )  # an annealed LR: its horizon must not move
  full = train.run(cfg, checkpoint_dir=str(tmp_path / "full"))
  ck = str(tmp_path / "burn")
  part = train.run(cfg, checkpoint_dir=ck, stop_after=1)
  assert int(part.state.update_step) == 1 and len(part.metrics) == 1
  assert (
      artifact.load_artifact(ck, 1).manifest.step == 1
  )  # checkpointed at the stop
  assert (
      part.metrics[0] == full.metrics[0]
  )  # a prefix: same update, same LR horizon
  rest = train.run(cfg, checkpoint_dir=ck)
  assert int(rest.state.update_step) == 4
  assert [m["lr"] for m in rest.metrics] == [m["lr"] for m in full.metrics[1:]]
  with pytest.raises(ValueError, match="stop_after"):
    train.run(cfg, stop_after=0)


def test_run_logs_telemetry_and_keeps_settings_out_of_the_series(tmp_path):
  """Measurements are series, settings are params.

  A kuhn run's rows carry the norms, ratios, RL stats and throughput; its
  settings (a flat lr and coefficients, static sizes, the league's shape) are
  in params.json and in no row; kuhn tracks exact exploitability (the
  kuhn_poker game of the JAX games package).

  Args:
    tmp_path: pytest's temporary directory.
  """
  ck = tmp_path / "ck"
  train.run(_tiny("mmd_kuhn_poker", updates=3), checkpoint_dir=str(ck))
  rows = metrics.read_jsonl(str(ck / "metrics" / metrics.JSONL_NAME))
  upd = _rows(ck / "metrics" / metrics.JSONL_NAME)
  for k in (
      "loss_total",
      "grad_norm",
      "grad_norm_max",
      "grad_norm_clipped",
      "grad_clipped",
      "update_norm",
      "update_ratio",
      "param_norm",
      "param_change_ratio",
      "approx_kl",
      "entropy",
      "update_seconds",
      "frames_per_s",
      "decisions_per_s",
  ):
    assert all(PAIR + k in r for r in upd), k
  settings = (
      "train/main/lr",
      "train/main/ent_coef_effective",
      "train/main/kl_coef_effective",
      "train/main/train_rows",
      "league/learners",
      "league/retired",
      "league/stopped",
  )
  for k in settings:
    assert not [r for r in rows if k in r], k
  params = json.loads((ck / "metrics" / metrics.PARAMS_NAME).read_text())
  for k in settings:
    assert k in params, k
  expl = [
      (r["step"], r["eval/exploitability"])
      for r in rows
      if "eval/exploitability" in r
  ]
  assert [s for s, _ in expl] == [
      0,
      1,
      2,
      3,
  ]  # every update of a 3-update run (and update 0)
  assert all(np.isfinite(v) and v >= -1e-9 for _, v in expl)


def test_a_flag_of_another_learner_kind_is_refused(tmp_path):
  """A flag of another learner kind is refused."""
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
  # Reaches private _KIND_FLAGS of the code under test.
  assert "data_dir" in train._KIND_FLAGS["bc"]  # pylint: disable=protected-access
  with flagsaver.flagsaver(
      config="mmd_kuhn_poker_jax",
      data_dir=str(tmp_path),
      num_updates=1,
      num_envs=4,
      num_steps=4,
  ):
    flags.FLAGS["data_dir"].present = 1
    try:
      with pytest.raises(app.UsageError, match="'bc' learner kind"):
        train.main(["train"])
    finally:
      flags.FLAGS["data_dir"].present = 0


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
