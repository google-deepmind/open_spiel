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

"""Training telemetry and the PG learner's telemetry metrics.

Training telemetry (learners/telemetry.py) and the PG learner's
telemetry metrics: known answers, and the metrics' relations to each other.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners import telemetry
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss

F32 = jnp.float32


def test_episode_sums_known_answer():
  # Row 0: a hand ends at ply 1 (return 1, started before the segment: no
  # length), a hand of plies 2..4 ends at 4 (return -2 + 0.5, length 3).
  # Row 1: no hand ends.
  """Episode sums known answer."""
  reward = jnp.asarray(
      [[0.0, 1.0, 0.0, -2.0, 0.5, 9.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], F32
  )
  done = jnp.asarray([[False, True, False, False, True, False], [False] * 6])
  s = jax.device_get(telemetry.episode_sums(reward, done))
  assert int(s["n"]) == 2 and int(s["n_len"]) == 1 and int(s["length_sum"]) == 3
  np.testing.assert_allclose(float(s["return_sum"]), 1.0 - 1.5)
  np.testing.assert_allclose(float(s["return_sumsq"]), 1.0 + 2.25)
  out = telemetry.episode_summary(s)
  np.testing.assert_allclose(out["episode_return_mean"], -0.25)
  np.testing.assert_allclose(
      out["episode_return_std"], np.std([1.0, -1.5], ddof=1)
  )
  assert out["episode_length_mean"] == 3.0
  assert not telemetry.episode_summary(
      jax.device_get(telemetry.episode_sums(reward, jnp.zeros_like(done)))
  )


def test_explained_variance_known_answer():
  """Explained variance known answer."""
  target = jnp.asarray([1.0, 2.0, 3.0, 4.0, 99.0], F32)
  pred = jnp.asarray([1.0, 2.0, 3.0, 5.0, 0.0], F32)
  mask = jnp.asarray([True, True, True, True, False])
  ev = telemetry.explained_variance(
      jax.device_get(telemetry.variance_parts(pred, target, mask))
  )
  t, r = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 0.0, -1.0])
  np.testing.assert_allclose(
      ev["value_explained_var"], 1 - r.var() / t.var(), rtol=1e-6
  )
  flat = telemetry.variance_parts(pred, jnp.ones_like(target), mask)
  # Undefined: left out.
  assert not telemetry.explained_variance(jax.device_get(flat))


def test_module_groups_look_through_the_params_collection():
  """Module groups look through the params collection."""
  tree = {
      "params": {
          "torso": {"w": jnp.ones((2,))},
          "value_head": {"w": jnp.ones((3,))},
      }
  }
  assert sorted(telemetry.module_groups(tree)) == ["torso", "value_head"]
  n = jax.device_get(telemetry.norms(tree, "param_norm"))
  np.testing.assert_allclose(n["param_norm"], np.sqrt(5.0), rtol=1e-6)
  np.testing.assert_allclose(n["param_norm/torso"], np.sqrt(2.0), rtol=1e-6)
  assert not telemetry.module_groups({"params": {"only": {"w": jnp.ones(1)}}})


def _cfg(**kw):
  """A small PGConfig; KL off for losses other than MMD unless given.

  Args:
    **kw: PGConfig overrides.

  Returns:
    The PGConfig.
  """
  base = dict(
      loss="mmd",
      lr=1e-3,
      lr_end=0.0,
      anneal_lr=False,
      num_updates=10,
      update_epochs=2,
      num_minibatches=2,
      shuffle="flat",
      max_grad_norm=0.5,
      gamma=0.99,
      gae_lambda=0.95,
      pg=loss_lib.PGLossConfig(clip_eps=0.2, vf_coef=0.5),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.2),
  )
  base.update(kw)
  if base["loss"] != "mmd" and "kl_coef" not in kw:
    base["kl_coef"] = schedules.Schedule("constant", 0.0)
  return pg.PGConfig(**base)


def _setup(game="kuhn_poker", b=8, t=8, **cfg_kw):
  """A learner, its state and a self-play batch.

  Args:
    game: The game.
    b: The rows.
    t: The plies.
    **cfg_kw: PGConfig overrides.

  Returns:
    (learner, state, batch).
  """
  pol = fx.make_policy(game, None)
  learner = pg.PGLearner(pol, _cfg(**cfg_kw))
  s = learner.init(jax.random.key(0), fx.make_env(game))
  batch = fx.self_play_batch(pol, s.params, game=game, b=b, t=t, seed=1)
  return learner, s, batch


@pytest.mark.parametrize("loss", pg.PG_LOSSES + (toy_loss.NAME,))
def test_pg_update_reports_norms_ratios_and_rl_stats(loss):
  """PG update reports norms ratios and RL stats."""
  learner, s, batch = _setup(loss=loss, max_grad_norm=0.05)
  s2, m = learner.update(s, batch, jax.random.key(3))
  for k in (
      "grad_norm",
      "grad_norm_max",
      "grad_norm_clipped",
      "grad_clipped",
      "update_norm",
      "update_ratio",
      "param_norm",
      "param_change_norm",
      "param_change_ratio",
      "approx_kl",
      "loss_total",
  ):
    assert k in m and np.isfinite(m[k]), k
  assert m["grad_norm_max"] >= m["grad_norm"] >= m["grad_norm_clipped"]
  assert m["grad_norm_clipped"] <= 0.05 + 1e-7
  assert 0.0 <= m["grad_clipped"] <= 1.0
  assert m["update_ratio"] > 0.0 and m["param_change_ratio"] > 0.0
  np.testing.assert_allclose(
      m["param_norm"], float(optax.tree.norm(s2.params)), rtol=1e-5
  )
  delta = jax.tree_util.tree_map(lambda a, b: a - b, s2.params, s.params)
  np.testing.assert_allclose(
      m["param_change_norm"], float(optax.tree.norm(delta)), rtol=1e-4
  )
  mods = [k for k in m if k.startswith("param_norm/")]
  if mods:
    np.testing.assert_allclose(
        sum(m[k] ** 2 for k in mods), m["param_norm"] ** 2, rtol=1e-4
    )
  if int(np.sum(np.asarray(batch.done))) >= 2:
    assert (
        np.isfinite(m["episode_return_std"])
        and "episode_return_mean/seat0" in m
    )
  for k in learner.held_metrics():
    if k == "kl_coef_effective" and loss != "mmd":
      continue
    if loss == toy_loss.NAME and k in (
        "ent_coef_effective",
        "kl_coef_effective",
    ):
      continue
    assert k in m, k


def test_pg_per_minibatch_norms_add_up():
  """Per-module grad norms square-sum to the global one, per minibatch."""
  learner, s, batch = _setup()
  prep = learner.prepare(s, batch)
  _, per_mb = learner.update_prepared(s, batch, prep, jax.random.key(0))
  mods = [k for k in per_mb if k.startswith("grad_norm/")]
  assert mods
  total = sum(np.asarray(per_mb[k], np.float64) ** 2 for k in mods)
  np.testing.assert_allclose(
      total, np.asarray(per_mb["grad_norm"], np.float64) ** 2, rtol=1e-4
  )
  np.testing.assert_array_equal(
      np.asarray(per_mb["grad_clipped"]),
      (np.asarray(per_mb["grad_norm"]) >= 0.5).astype(np.float32),
  )


def test_terms_that_are_off_are_not_reported():
  """Terms that are off are not reported."""
  learner, s, batch = _setup(loss="ppo")
  _, m = learner.update(s, batch, jax.random.key(0))
  for k in (
      "clip_frac_value",
      "advantage_filter_keep_fraction",
      "advantage_filter_fallback",
  ):
    assert k not in m
  learner, s, batch = _setup(
      loss="ppo", pg=loss_lib.PGLossConfig(clip_value=True)
  )
  _, m = learner.update(s, batch, jax.random.key(0))
  assert "clip_frac_value" in m


def test_episode_returns_per_seat_and_every_seat_batches():
  """Episode returns per seat and every seat batches."""
  reward = jnp.asarray([[0.0, 2.0], [0.0, -2.0], [0.0, 1.0], [0.0, -1.0]], F32)
  done = jnp.asarray([[False, True]] * 4)
  seat = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]], jnp.int32)
  out = telemetry.episode_summary(
      jax.device_get(telemetry.episode_sums(reward, done, seat, 2))
  )
  assert (
      out[
          # zero-sum: the mean over seats says nothing
          "episode_return_mean"
      ]
      == 0.0
  )
  assert (
      out["episode_return_mean/seat0"] == 1.5
      and out["episode_return_mean/seat1"] == -1.5
  )
  assert telemetry.every_seat_rows(seat, 2)
  assert not telemetry.every_seat_rows(jnp.zeros((4, 2), jnp.int32), 2)
  assert not telemetry.every_seat_rows(seat[:3], 2)


def test_pg_every_seat_batch_reports_per_seat_returns_not_the_zero_sum_mean():
  """PG every seat batch reports per seat returns not the zero sum mean."""
  learner, s, batch = _setup(b=8, t=16)  # self-play, every seat a row
  _, m = learner.update(s, batch, jax.random.key(0))
  assert "episode_return_mean" not in m and "reward_per_row" not in m
  assert "episode_return_mean/seat0" in m and "episode_return_std" in m


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
