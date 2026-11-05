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

"""Tests of learners.ppo_magnet and its recipe.

The loss terms, the temperature and step-size clocks (Schedule rescale_step)
and the PGConfig the recipe builds.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners import ppo_magnet as pm


def _inputs(n=64, a=7, seed=0):
  """Random loss inputs.

  Args:
    n: rows.
    a: actions.
    seed: the key seed.

  Returns:
    The loss's keyword arguments.
  """
  k = jax.random.split(jax.random.key(seed), 6)
  legal = jax.random.bernoulli(k[0], 0.6, (n, a)).at[:, 0].set(True)
  logits = jax.random.normal(k[1], (n, a), jnp.float32)
  beh = loss_lib.masked_log_softmax(
      logits + 0.3 * jax.random.normal(k[2], (n, a), jnp.float32), legal
  )
  action = jnp.argmax(
      jnp.where(legal, jax.random.uniform(k[3], (n, a), jnp.float32), -1.0), -1
  )
  return dict(
      logits=logits,
      value=jax.random.normal(k[4], (n,), jnp.float32),
      legal=legal,
      action=action.astype(jnp.int32),
      behavior_log_probs=beh,
      advantages=jax.random.normal(k[5], (n,), jnp.float32),
      returns=jnp.zeros((n,), jnp.float32),
      policy_mask=jnp.ones((n,), bool),
      value_mask=jnp.ones((n,), bool),
  )


def test_loss_terms_by_hand():
  cfg = pm.PPOMagnetConfig()
  inp = _inputs()
  total, terms = pm.ppo_magnet_loss(cfg, temperature=0.05, **inp)
  lp = np.asarray(loss_lib.masked_log_softmax(inp["logits"], inp["legal"]))
  beh = np.asarray(inp["behavior_log_probs"])
  legal = np.asarray(inp["legal"])
  p = np.where(legal, np.exp(lp), 0.0)
  idx = np.arange(64), np.asarray(inp["action"])
  ratio = np.exp(lp[idx] - beh[idx])
  adv = np.asarray(inp["advantages"])
  surr = np.minimum(adv * ratio, adv * np.clip(ratio, 0.8, 1.2))
  kl_beh = np.where(legal, p * (lp - np.where(legal, beh, 0.0)), 0.0).sum(-1)
  n_legal = legal.sum(-1)
  ukl = np.where(legal, p * lp, 0.0).sum(-1) + np.log(n_legal)
  vl = np.mean(np.asarray(inp["value"]) ** 2)
  want = -surr.mean() + 0.1 * kl_beh.mean() + 0.05 * ukl.mean()
  np.testing.assert_allclose(
      float(terms["policy_loss"]), -surr.mean(), rtol=1e-5
  )
  np.testing.assert_allclose(
      float(terms["kl_loss"]), kl_beh.mean(), rtol=1e-4, atol=1e-6
  )
  np.testing.assert_allclose(
      float(terms["magnet_kl"]), ukl.mean(), rtol=1e-4, atol=1e-6
  )
  np.testing.assert_allclose(float(terms["value_loss"]), vl, rtol=1e-5)
  np.testing.assert_allclose(float(total), want + vl, rtol=1e-4, atol=1e-6)


def test_registered_and_chunked_terms_add_up():
  assert pg.extension(pm.LOSS) is pm.SPEC
  cfg = pm.PPOMagnetConfig()
  inp = _inputs(n=32)
  full, terms = pm.ppo_magnet_loss(cfg, temperature=0.02, **inp)
  counts = pg.LossCounts(policy=jnp.float32(32.0), value=jnp.float32(32.0))
  parts = [
      pm.ppo_magnet_loss(
          cfg,
          temperature=0.02,
          counts=counts,
          **{k: v[sl] for k, v in inp.items()}
      )
      for sl in (slice(0, 10), slice(10, 32))
  ]
  np.testing.assert_allclose(
      float(parts[0][0] + parts[1][0]), float(full), rtol=1e-5
  )
  for k in pm.CHUNK_SUMS:
    np.testing.assert_allclose(
        float(parts[0][1][k] + parts[1][1][k]),
        float(terms[k]),
        rtol=1e-4,
        atol=1e-6,
    )


def test_temperature_and_lr_rescale_clock():
  x64.enable_x64()
  cfg = pm.PPOMagnetConfig(rescale_step=3000, rescale_time_scale=14.0)

  def t(n):
    """The temperature at update n.

    Args:
      n: the update.

    Returns:
      The temperature.
    """
    return float(cfg.temperature(np.int64(n)))

  assert t(0) == pytest.approx(0.05)
  assert t(2999) == pytest.approx(0.05 / 3000**0.3, rel=1e-6)
  assert t(3000) == pytest.approx(0.05 / (3000 * 14 + 1) ** 0.3, rel=1e-6)
  assert t(6000) == pytest.approx(0.05 / (6000 * 14 + 1) ** 0.3, rel=1e-6)
  lc = recipes.ppo_magnet_config(
      num_updates=10,
      sgd_steps=4,
      value_scale=13.0,
      rescale_step=3000,
      rescale_time_scale=14.0,
  )
  lr = lc.lr_schedule()
  assert float(lr.value(np.int64(0))) == pytest.approx(1e-4)
  assert float(lr.value(np.int64(2999))) == pytest.approx(
      min(1e-4, 0.5 / 3000**1.1)
  )
  for n in (3000, 3001, 6000, 9000):
    assert float(lr.value(np.int64(n))) == pytest.approx(5e-6, rel=1e-12)
  with pytest.raises(ValueError, match="rescale"):
    recipes.ppo_magnet_config(
        num_updates=10,
        sgd_steps=4,
        value_scale=13.0,
        rescale_step=3000,
        rescale_time_scale=14.0,
        loss_config=pm.PPOMagnetConfig(),
    )
  with pytest.raises(ValueError, match="floor"):
    recipes.ppo_magnet_config(
        num_updates=10,
        sgd_steps=4,
        value_scale=13.0,
        rescale_step=10,
        rescale_time_scale=2.0,
    )


def test_lr_without_rescale_is_the_plain_schedule():
  lc = recipes.ppo_magnet_config(num_updates=10, sgd_steps=4, value_scale=1.0)
  assert lc.lr_sched == schedules.Schedule(
      "inverse_power", 0.5, power=1.1, floor=5e-6, ceil=1e-4
  )


def test_field_rewards():
  r = jnp.asarray(
      [
          [0.0, 26.0],
          [0.0, 25.0],
          [0.0, 13.0],
          [0.0, 14.0],
          [26.0, 0.0],
          [0.0, 0.0],
          [0.0, 0.0],
          [0.0, 0.0],
      ],
      jnp.float32,
  )
  out = np.asarray(pm.field_rewards(r, 4))
  np.testing.assert_allclose(out.reshape(2, 4, 2).sum(1), 0.0, atol=1e-5)
  np.testing.assert_allclose(out[0, 1], 26.0 - (25 + 13 + 14) / 3.0, rtol=1e-6)
  np.testing.assert_allclose(
      out[4:, 0], [26.0, -26 / 3, -26 / 3, -26 / 3], rtol=1e-6
  )
  with pytest.raises(ValueError, match="groups"):
    pm.field_rewards(r[:6], 4)
  with pytest.raises(ValueError):
    pm.PPOMagnetConfig(field_players=1)
  lc = recipes.ppo_magnet_config(
      num_updates=10, sgd_steps=4, value_scale=13.0, field_players=4
  )
  adv, _ = pm.SPEC.targets(
      lc.loss_config,
      r,
      jnp.zeros_like(r),
      jnp.ones(r.shape, bool),
      jnp.zeros(r.shape, bool),
      jnp.zeros((8,), jnp.float32),
  )
  assert np.asarray(adv).shape == (8, 2)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
