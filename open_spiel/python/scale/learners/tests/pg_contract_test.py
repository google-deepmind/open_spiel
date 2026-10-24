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

"""PGLearner contract tests and known answers.

PGLearner: config validation, shapes and dtypes, determinism, masking, x64
counters, and known answers (GAE wiring, the first Adam step, the LR
schedule, the magnet refresh, a registered loss's composition).
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss

tree_map = jax.tree_util.tree_map
F32 = jnp.float32


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


def _setup(game="kuhn_poker", network=None, b=8, t=8, seed=0, **cfg_kw):
  """A policy, its learner and state, and a self-play batch.

  Args:
    game: The game.
    network: The network config, or None for the default.
    b: The rows.
    t: The plies.
    seed: The seed.
    **cfg_kw: PGConfig overrides.

  Returns:
    (policy, learner, state, batch).
  """
  pol = fx.make_policy(game, network)
  learner = pg.PGLearner(pol, _cfg(**cfg_kw))
  s = learner.init(jax.random.key(seed), fx.make_env(game))
  batch = fx.self_play_batch(pol, s.params, game=game, b=b, t=t, seed=seed + 1)
  return pol, learner, s, batch


# -- config validation -------------------------------------------------------


def test_config_rejects_bad_settings():
  """Config rejects bad settings."""
  with pytest.raises(AssertionError):
    _cfg(loss="a2c")
  with pytest.raises(ValueError, match="ppo"):
    _cfg(loss="ppo", kl_coef=schedules.Schedule("constant", 0.1))
  with pytest.raises(ValueError, match="value_ema"):
    _cfg(value_ema=True, pg=loss_lib.PGLossConfig(value_head="hl_gauss"))
  with pytest.raises(ValueError, match="value_ema"):
    _cfg(loss=toy_loss.NAME, value_ema=True)
  with pytest.raises(AssertionError):
    _cfg(shuffle="time")
  with pytest.raises(AssertionError):
    _cfg(update_epochs=0)


def test_stateful_policy_needs_env_axis_shuffle():
  """Stateful policy needs env axis shuffle."""
  pol = fx.make_policy("kuhn_poker", fx.rnn())
  with pytest.raises(ValueError, match="env_axis"):
    pg.PGLearner(pol, _cfg(shuffle="flat"))
  pg.PGLearner(pol, _cfg(shuffle="env_axis"))


def test_minibatch_divisibility_fails_loud():
  """Minibatch divisibility fails loud."""
  _, learner, s, batch = _setup(
      b=5, t=5, num_minibatches=4
  )  # 10 rows * 5 plies = 50
  with pytest.raises(ValueError, match="divisible"):
    learner.update(s, batch, jax.random.key(0))


def test_batch_dtypes_are_asserted():
  """Batch dtypes are asserted."""
  _, learner, s, batch = _setup()
  with pytest.raises(AssertionError, match="reward"):
    learner.update(
        s,
        batch._replace(reward=batch.reward.astype(jnp.float64)),
        jax.random.key(0),
    )
  with pytest.raises(AssertionError, match="valid"):
    learner.update(
        s,
        batch._replace(valid=batch.valid.astype(jnp.int32)),
        jax.random.key(0),
    )


# -- shapes, dtypes, determinism ---------------------------------------------


@pytest.mark.parametrize("loss", pg.PG_LOSSES + (toy_loss.NAME,))
def test_update_keeps_tree_shapes_and_dtypes(loss):
  """Update keeps tree shapes and dtypes."""
  _, learner, s, batch = _setup(loss=loss, network=fx.mlp(dtype="bfloat16"))
  s2, m = learner.update(s, batch, jax.random.key(3))
  assert jax.tree_util.tree_structure(
      s2.params
  ) == jax.tree_util.tree_structure(s.params)
  for a, b in zip(
      jax.tree_util.tree_leaves(s.params), jax.tree_util.tree_leaves(s2.params)
  ):
    # bf16 compute, f32 params
    assert a.shape == b.shape and a.dtype == b.dtype == F32
  assert s.update_step.dtype == jnp.int64 and int(s.update_step) == 0
  assert s2.update_step.dtype == jnp.int64 and int(s2.update_step) == 1
  assert not fx.leaves_equal(s.params, s2.params)
  for k, v in m.items():
    assert isinstance(v, float) and np.isfinite(v), (k, v)
  for k in ("loss_total", "grad_norm", "lr"):
    assert k in m
  if loss == "ppo":
    assert "kl_penalty" not in m


def test_update_is_deterministic_and_key_dependent():
  """Update is deterministic and key dependent."""
  _, learner, s, batch = _setup(num_minibatches=4)
  a, ma = learner.update(s, batch, jax.random.key(7))
  b, mb = learner.update(s, batch, jax.random.key(7))
  c, _ = learner.update(s, batch, jax.random.key(8))
  assert fx.leaves_equal(a, b) and ma == mb
  assert not fx.leaves_equal(a.params, c.params)  # a different shuffle


@pytest.mark.parametrize("loss", pg.PG_LOSSES + (toy_loss.NAME,))
def test_invalid_rows_and_illegal_logits_do_not_change_the_update(loss):
  """Invalid rows and illegal logits do not change the update."""
  _, learner, s, batch = _setup("leduc_poker", loss=loss, b=8, t=10)
  assert not bool(
      jnp.all(batch.steps.legal_mask | ~batch.valid[..., None])
  ), "need illegal"
  rng = np.random.default_rng(0)
  inv = ~batch.valid

  def junk(shape):
    """Large random values.

    Args:
      shape: The shape.

    Returns:
      f32 of `shape`.
    """
    return jnp.asarray(rng.normal(size=shape) * 1e3, F32)

  legal = batch.steps.legal_mask
  p = batch._replace(
      behavior_logits=jnp.where(
          legal & batch.valid[..., None],
          batch.behavior_logits,
          junk(batch.behavior_logits.shape),
      ),
      behavior_logp=jnp.where(
          inv, junk(batch.behavior_logp.shape), batch.behavior_logp
      ),
      value_pred=jnp.where(
          inv[..., None], junk(batch.value_pred.shape), batch.value_pred
      ),
      target_policy=junk(batch.target_policy.shape),
      target_value=junk(batch.target_value.shape),
      steps=batch.steps._replace(
          info=batch.steps.info._replace(
              card=jnp.where(
                  inv,
                  jnp.asarray(rng.integers(0, 6, inv.shape), jnp.int32),
                  batch.steps.info.card,
              )
          )
      ),
  )
  a, _ = learner.update(s, batch, jax.random.key(1))
  b, _ = learner.update(s, p, jax.random.key(1))
  assert fx.leaves_equal(a.params, b.params)


# -- known answers -------------------------------------------------------------


def _hand_batch(pol, s, value_ema):
  """A kuhn batch row rewritten by hand.

  A kuhn batch row rewritten by hand: valid t0, t2, t4; the hand ends at
  t1 (opponent ply) with reward 1; no done after t4, so t4 bootstraps.

  Args:
    pol: the kuhn policy.
    s: the learner state.
    value_ema: unused (the row is the same with and without the EMA).

  Returns:
    (batch, the row's perspective seat).
  """
  del value_ema  # Unused: the row is the same with and without the EMA.
  batch = fx.self_play_batch(pol, s.params, b=1, t=5, all_seats=False)
  seat = int(batch.steps.seat[0, 0])
  valid = jnp.asarray([[True, False, True, False, True]])
  done = jnp.asarray([[False, True, False, False, False]])
  reward = jnp.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]], F32)
  v = np.zeros((1, 5, 2), np.float32)
  v[0, [0, 2, 4], seat] = [0.5, 0.2, 0.4]
  return (
      batch._replace(
          valid=valid, done=done, reward=reward, value_pred=jnp.asarray(v)
      ),
      seat,
  )


@pytest.mark.parametrize("value_ema", [False, True])
def test_prepare_known_answer(value_ema):
  """Prepare known answer."""
  pol = fx.make_policy()
  learner = pg.PGLearner(
      pol,
      _cfg(gamma=0.9, gae_lambda=0.8, value_ema=value_ema, num_minibatches=1),
  )
  s = learner.init(jax.random.key(0), fx.make_env())
  batch, seat = _hand_batch(pol, s, value_ema)
  out = pol.evaluate(s.params, (), batch.bootstrap_step, None)
  valb = float(out.value[0, seat])
  if value_ema:  # the initial EMA (zeros) denormalizes every value to 0
    val0, val2, val4, valb = 0.0, 0.0, 0.0, 0.0
  else:
    val0, val2, val4 = 0.5, 0.2, 0.4
  a4 = 0.9 * valb - val4
  a2 = (0.9 * val4 - val2) + 0.72 * a4
  a0 = 1.0 - val0
  prep = learner.prepare(s, batch)
  np.testing.assert_allclose(
      np.asarray(prep.advantages[0]), [a0, 0, a2, 0, a4], rtol=1e-6, atol=1e-7
  )
  np.testing.assert_allclose(
      np.asarray(prep.targets[0]),
      [a0 + val0, 0, a2 + val2, 0, a4 + val4],
      rtol=1e-6,
      atol=1e-7,
  )
  np.testing.assert_allclose(
      np.asarray(prep.value_old[0])[[0, 2, 4]], [val0, val2, val4], atol=1e-7
  )
  for x in prep:
    assert x.dtype == F32


def test_first_step_is_the_adam_sign_step_on_the_pg_loss():
  """First step is the adam sign step on the pg loss.

  One epoch, one minibatch, no clipping: Adam's first step is -lr * g / (|g| +
  eps), g = grad of losses.pg_loss on every row.
  """
  lr, eps = 1e-3, 1e-5
  pol, learner, s, batch = _setup(
      "leduc_poker",
      loss="ppo",
      lr=lr,
      update_epochs=1,
      num_minibatches=1,
      max_grad_norm=1e9,
      b=4,
      t=10,
  )
  prep = learner.prepare(s, batch)
  n = batch.valid.size

  def flat(x):
    """Merges the row and time axes.

    Args:
      x: A leaf, [B, T, ...].

    Returns:
      The leaf, [n, ...].
    """
    return x.reshape((n,) + x.shape[2:])

  def _loss(p):
    """The pg loss of params `p` on the flattened batch.

    Args:
      p: The params.

    Returns:
      The total loss.
    """
    out = pol.evaluate(p, (), tree_map(flat, batch.steps), None)
    seat = flat(batch.steps.seat)
    v = out.value[jnp.arange(n), seat]
    total, _ = loss_lib.pg_loss(
        learner.cfg.pg,
        logits=out.logits,
        value=v,
        legal=flat(batch.steps.legal_mask),
        action=flat(batch.action),
        behavior_logp=flat(batch.behavior_logp),
        mask=flat(batch.valid),
        advantages=flat(prep.advantages),
        targets=flat(prep.targets),
        value_old=flat(prep.value_old),
        ent_coef=0.01,
    )
    return total

  g = jax.grad(_loss)(s.params)
  s2, _ = learner.update(s, batch, jax.random.key(0))
  want = tree_map(lambda gg: -lr * gg / (jnp.abs(gg) + eps), g)
  got = tree_map(lambda a, b: b - a, s.params, s2.params)
  # Steps are ~lr in size; where |g| ~ eps the sign step amplifies the f32
  # summation-order difference of the shuffled minibatch, hence atol 2e-5 * lr.
  for w, x in zip(
      jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
  ):
    np.testing.assert_allclose(
        np.asarray(x), np.asarray(w), rtol=1e-4, atol=2e-5 * lr
    )
  assert (
      max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(got))
      > 0.5 * lr
  )


def test_lr_schedule_known_answers_and_int64():
  """LR schedule known answers and int64."""
  pol = fx.make_policy()
  lrn = pg.PGLearner(
      pol, _cfg(lr=1e-3, lr_end=0.0, anneal_lr=True, num_updates=4)
  )
  got = [float(lrn.lr_at(jnp.int64(u))) for u in range(6)]
  np.testing.assert_allclose(
      got, [1e-3, 7.5e-4, 5e-4, 2.5e-4, 0.0, 0.0], atol=1e-12
  )
  big = pg.PGLearner(
      pol, _cfg(lr=1.0, lr_end=0.0, anneal_lr=True, num_updates=2**32)
  )
  u = 2**31 + 5
  np.testing.assert_allclose(
      float(big.lr_at(jnp.int64(u))), 1.0 - u / 2**32, rtol=1e-12
  )
  const = pg.PGLearner(pol, _cfg(lr=3e-4, anneal_lr=False))
  assert float(const.lr_at(jnp.int64(10**12))) == pytest.approx(3e-4)


def test_update_step_counter_does_not_wrap_at_2_31():
  """Update step counter does not wrap at 2 31."""
  _, learner, s, batch = _setup(
      anneal_lr=True, num_updates=2**33, reference_update_every=7
  )
  s = s._replace(update_step=jnp.int64(2**31 - 1))
  s2, m = learner.update(s, batch, jax.random.key(0))
  assert s2.update_step.dtype == jnp.int64 and int(s2.update_step) == 2**31
  assert m["update_step"] == float(2**31)
  assert 0.0 < m["lr"] < 1e-3


def test_magnet_refresh_every_n():
  """Magnet refresh every n."""
  _, learner, s0, batch = _setup(reference_update_every=2)
  assert fx.leaves_equal(s0.magnet, s0.params)
  s1, _ = learner.update(s0, batch, jax.random.key(1))
  assert fx.leaves_equal(s1.magnet, s0.params)  # after update 1: not due
  s2, _ = learner.update(s1, batch, jax.random.key(2))
  assert fx.leaves_equal(s2.magnet, s2.params)  # after update 2: re-taken
  assert not fx.leaves_equal(s2.magnet, s0.params)


def test_ppo_has_no_magnet():
  """PPO has no magnet."""
  _, _, s, _ = _setup(loss="ppo")
  assert s.magnet is None


def test_registered_loss_composition_known_answer():
  """Registered loss composition known answer.

  A registered loss (toy_loss): its own targets (GAE(lam) advantages, TD(lam)
  returns), and the per-minibatch terms equal its loss on every row at its
  temperature of update 0.
  """
  tcfg = toy_loss.ToyLossConfig()
  pol, learner, s, batch = _setup(
      "leduc_poker",
      loss=toy_loss.NAME,
      loss_config=tcfg,
      value_ema=False,
      update_epochs=1,
      num_minibatches=1,
      b=4,
      t=10,
  )
  prep = learner.prepare(s, batch)
  v = adv_lib.seat_value(batch.value_pred, batch.steps.seat)
  out_b = pol.evaluate(s.params, (), batch.bootstrap_step, None)
  boot = out_b.value[
      jnp.arange(out_b.value.shape[0]), batch.bootstrap_step.seat
  ]
  g = adv_lib.gae(
      batch.reward,
      v,
      batch.valid,
      batch.done,
      boot,
      gamma=tcfg.gamma,
      lam=tcfg.lam,
  )
  np.testing.assert_allclose(
      np.asarray(prep.advantages), np.asarray(g.advantages), atol=1e-6
  )
  np.testing.assert_allclose(
      np.asarray(prep.targets), np.asarray(g.returns), atol=1e-6
  )

  n = batch.valid.size

  def flat(x):
    """Merges the row and time axes.

    Args:
      x: A leaf, [B, T, ...].

    Returns:
      The leaf, [n, ...].
    """
    return x.reshape((n,) + x.shape[2:])

  out = pol.evaluate(s.params, (), tree_map(flat, batch.steps), None)
  legal = flat(batch.steps.legal_mask)
  _, terms = toy_loss.loss(
      tcfg,
      logits=out.logits,
      value=out.value[jnp.arange(n), flat(batch.steps.seat)],
      legal=legal,
      action=flat(batch.action),
      behavior_log_probs=loss_lib.masked_log_softmax(
          flat(batch.behavior_logits), legal
      ),
      advantages=flat(prep.advantages),
      returns=flat(prep.targets),
      policy_mask=flat(batch.valid),
      value_mask=flat(batch.valid),
      temperature=toy_loss.temperature_schedule(tcfg, jnp.int64(0), 0, 1.0),
  )
  _, per_mb = learner.update_prepared(s, batch, prep, jax.random.key(0))
  for k in ("policy_loss", "magnet_kl", "value_loss", "temperature"):
    np.testing.assert_allclose(
        float(per_mb[k][0, 0]), float(terms[k]), rtol=1e-5, atol=1e-7
    )
  np.testing.assert_allclose(
      float(learner.temperature_at(jnp.int64(4))), 0.1 / 5, rtol=1e-6
  )


def test_state_dict_round_trip():
  """State dict round trip."""
  _, learner, s, batch = _setup(value_ema=True)
  s, _ = learner.update(s, batch, jax.random.key(0))
  blob = learner.state_dict(s)
  assert all(
      x is not None
      for x in jax.tree_util.tree_leaves(blob, is_leaf=lambda x: x is None)
  )
  s2 = learner.load_state(blob)
  assert fx.leaves_equal(s, s2)
  assert s2.update_step.dtype == jnp.int64
  a, _ = learner.update(s, batch, jax.random.key(5))
  b, _ = learner.update(s2, batch, jax.random.key(5))
  assert fx.leaves_equal(a, b)


def test_learner_protocol_surface():
  """Learner protocol surface."""
  pol, learner, s, _ = _setup()
  assert learner.policy() is pol
  assert learner.params(s) is s.params
  assert dataclasses.is_dataclass(learner.cfg)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
