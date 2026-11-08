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

"""PGLearner teacher distillation (pg.py module doc "Teacher distillation").

* PGConfig.teacher_coef at its default is off and left out of the config
  hash; a registered loss refuses it;
* set_teacher refuses a teacher that does not fit (game, actions, players,
  a stateful policy), a config without teacher_coef and a second teacher;
  update refuses teacher_coef without a teacher;
* the update's teacher_kl is KL(teacher || learner) over the trained rows,
  computed by hand from the teacher's own forward (its own params and spec);
  0 with top-1 agreement 1 when the teacher is the learner;
* the teacher's params are unchanged by an update; the learner's move;
* c(t) = 0 leaves the update the one without a teacher;
* the teacher's probabilities are the same in chunks or in one call, and an
  accumulated (chunked) minibatch gives the one-call update.
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss

COEF = schedules.Schedule("linear", 1.0, 0.0, decay_steps=10)


def _cfg(**kw):
  """A small PPO PGConfig with the teacher schedule COEF.

  Args:
    **kw: PGConfig fields to override.

  Returns:
    The config.
  """
  base = dict(
      loss="ppo",
      lr=1e-3,
      anneal_lr=False,
      update_epochs=1,
      num_minibatches=1,
      teacher_coef=COEF,
  )
  base.update(kw)
  return pg.PGConfig(**base)


def _setup(cfg, teacher_net=None, game="kuhn_poker"):
  """A PG learner, its state, a teacher and a self-play batch.

  Args:
    cfg: the PGConfig.
    teacher_net: the teacher's network (default a width-8 MLP).
    game: the game.

  Returns:
    (learner, state, teacher policy, teacher params, batch).
  """
  pol = fx.make_policy(game)
  lrn = pg.PGLearner(pol, cfg)
  s = lrn.init(jax.random.key(0), fx.make_env(game))
  tpol = fx.make_policy(game, teacher_net or fx.mlp(widths=(8,)))
  tparams = tpol.init_params(jax.random.key(7))
  batch = fx.self_play_batch(pol, s.params, game, b=8, t=8, seed=3)
  return lrn, s, tpol, tparams, batch


def _masked_probs(logits, legal):
  """Legal-masked softmax in f64 (NaN on a row with no legal action).

  Args:
    logits: [..., A] logits.
    legal: [..., A] bool legal mask.

  Returns:
    The [..., A] probabilities.
  """
  with np.errstate(invalid="ignore"):
    x = np.where(legal, np.asarray(logits, np.float64), -np.inf)
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.where(legal, np.exp(x), 0.0)
    return p / p.sum(axis=-1, keepdims=True)


def test_default_is_off_and_not_hashed():
  c = pg.PGConfig(num_updates=10)
  assert c.teacher_coef == pg.TEACHER_COEF_OFF and not c.teacher_on()
  assert "teacher_coef" not in common.config_dict(c)
  on = dataclasses.replace(c, teacher_coef=COEF)
  assert on.teacher_on()
  assert common.config_dict(on)["teacher_coef"]["kind"] == "linear"
  assert common.config_hash(on) != common.config_hash(c)
  with pytest.raises(ValueError, match="Schedule"):
    pg.PGConfig(num_updates=10, teacher_coef=0.5)


def test_refusals():
  lrn, s, tpol, tparams, batch = _setup(_cfg())
  with pytest.raises(ValueError, match="call set_teacher"):
    lrn.update(s, batch, jax.random.key(1))
  off = pg.PGLearner(fx.make_policy(), _cfg(teacher_coef=pg.TEACHER_COEF_OFF))
  with pytest.raises(ValueError, match="off default"):
    off.set_teacher(tpol, tparams, name="t")
  leduc = fx.make_policy("leduc_poker")
  with pytest.raises(ValueError, match="leduc_poker"):
    lrn.set_teacher(leduc, leduc.init_params(jax.random.key(0)), name="leduc")
  rnn = fx.make_policy("kuhn_poker", fx.rnn())
  with pytest.raises(ValueError, match="stateless"):
    lrn.set_teacher(rnn, rnn.init_params(jax.random.key(0)), name="rnn")
  lrn.set_teacher(tpol, tparams, name="t")
  with pytest.raises(ValueError, match="already"):
    lrn.set_teacher(tpol, tparams, name="t2")
  with pytest.raises(NotImplementedError, match="teacher"):
    lrn.update_prepared(s, batch, lrn.prepare(s, batch), jax.random.key(1))


def test_teacher_kl_by_hand_and_teacher_frozen():
  """At lr 0 (params fixed) the teacher metrics equal the hand values.

  teacher_kl and teacher_top1 are the hand values over the trained rows, from
  the teacher's own forward; c(0) = 1.
  """
  lrn, s, tpol, tparams, batch = _setup(_cfg(lr=0.0))
  before = jax.tree_util.tree_map(np.asarray, tparams)
  lrn.set_teacher(tpol, tparams, name="t")
  _, m = lrn.update(s, batch, jax.random.key(1))
  b, t = batch.valid.shape
  flat = jax.tree_util.tree_map(
      lambda x: x.reshape((b * t,) + x.shape[2:]), batch.steps
  )
  legal = np.asarray(flat.legal_mask)
  valid = np.asarray(batch.valid).reshape(-1)
  s_logits, _, _ = lrn.forward(s.params, None, flat)
  seat = jnp.asarray(flat.seat)
  t_logits = tpol.evaluate(
      tparams, tpol.init_carry(tparams, b * t, seat, None), flat, None
  ).logits
  pt, ps = _masked_probs(t_logits, legal), _masked_probs(s_logits, legal)
  with np.errstate(divide="ignore", invalid="ignore"):
    per = np.where(legal & (pt > 0), pt * (np.log(pt) - np.log(ps)), 0.0).sum(
        -1
    )
  want = per[valid].mean()
  assert want > 0
  np.testing.assert_allclose(m["teacher_kl"], want, rtol=1e-4, atol=1e-6)
  top = np.argmax(np.where(legal, pt, -1), -1) == np.argmax(
      np.where(legal, ps, -1), -1
  )
  np.testing.assert_allclose(m["teacher_top1"], top[valid].mean(), atol=1e-6)
  assert m["teacher_coef"] == 1.0
  for x, y in zip(
      jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(tparams)
  ):
    assert np.array_equal(x, np.asarray(y))
  assert "teacher_coef" in lrn.held_metrics()


def test_teacher_equal_to_learner_has_zero_kl():
  pol = fx.make_policy()
  lrn = pg.PGLearner(pol, _cfg(lr=0.0))
  s = lrn.init(jax.random.key(0), fx.make_env())
  lrn.set_teacher(pol, s.params, name="self")
  batch = fx.self_play_batch(pol, s.params, b=8, t=8, seed=3)
  _, m = lrn.update(s, batch, jax.random.key(1))
  assert abs(m["teacher_kl"]) < 1e-6, m["teacher_kl"]
  assert m["teacher_top1"] == 1.0


def test_update_trains_toward_the_teacher_and_leaves_it_frozen():
  lrn, s, tpol, tparams, batch = _setup(
      _cfg(
          lr=3e-2,
          update_epochs=4,
          teacher_coef=schedules.Schedule("constant", 50.0),
      )
  )
  before = jax.tree_util.tree_map(np.asarray, tparams)
  lrn.set_teacher(tpol, tparams, name="t")
  kls = []
  for i in range(5):
    s2, m = lrn.update(s, batch, jax.random.key(i))
    kls.append(m["teacher_kl"])
    assert not fx.leaves_equal(s2.params, s.params)
    s = s2
  assert kls[-1] < kls[0], kls
  for x, y in zip(
      jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(tparams)
  ):
    assert np.array_equal(x, np.asarray(y))


def test_zero_coefficient_is_the_update_without_a_teacher():
  zero = schedules.Schedule("linear", 0.0, 0.0, decay_steps=1)
  lrn, s, tpol, tparams, batch = _setup(_cfg(teacher_coef=zero))
  lrn.set_teacher(tpol, tparams, name="t")
  plain = pg.PGLearner(lrn.policy(), _cfg(teacher_coef=pg.TEACHER_COEF_OFF))
  s_t, m_t = lrn.update(s, batch, jax.random.key(1))
  s_p, m_p = plain.update(
      plain.init_from_params(s.params), batch, jax.random.key(1)
  )
  assert fx.leaves_equal(s_t.params, s_p.params)
  assert m_t["loss_total"] == m_p["loss_total"]
  assert {"teacher_kl", "teacher_top1", "teacher_coef"} <= set(m_t)
  assert not {"teacher_kl", "teacher_top1", "teacher_coef"} & set(m_p)


def test_teacher_probs_chunked_equal_unchunked():
  lrn, _, tpol, tparams, batch = _setup(_cfg())
  lrn.set_teacher(tpol, tparams, name="t")
  rows = {"steps": batch.steps, "valid": batch.valid}
  whole = np.asarray(lrn._teacher_probs(tparams, rows))  # pylint: disable=protected-access
  small = pg.PGLearner(lrn.policy(), _cfg())
  small.set_teacher(tpol, tparams, name="t", chunk_rows=5)
  parts = np.asarray(small._teacher_probs(tparams, rows))  # pylint: disable=protected-access
  assert whole.shape == batch.valid.shape + (fx.make_env().num_actions,)
  np.testing.assert_allclose(parts, whole, atol=1e-6, rtol=0)
  legal = np.asarray(batch.steps.legal_mask)
  assert (whole[~legal] == 0).all()
  some = legal.any(-1)  # a ply with no legal action (no decision) gets all 0
  assert some.any() and (~some).any()
  np.testing.assert_allclose(whole.sum(-1)[some], 1.0, atol=1e-5)
  assert (whole.sum(-1)[~some] == 0).all()


def test_accumulated_minibatch_matches_one_call():
  """Chunked teacher KL (forward_chunk_rows < minibatch) equals one call.

  The teacher KL's chunk sums give the one-call metrics and update.
  """
  _, s, tpol, tparams, batch = _setup(_cfg())
  out = {}
  for chunk in (10**6, 8):
    lrn = pg.PGLearner(fx.make_policy(), _cfg(forward_chunk_rows=chunk))
    lrn.set_teacher(tpol, tparams, name="t")
    out[chunk] = lrn.update(
        lrn.init_from_params(s.params), batch, jax.random.key(1)
    )
  (s_a, m_a), (s_b, m_b) = out[10**6], out[8]
  for k in ("teacher_kl", "teacher_top1", "loss_total"):
    np.testing.assert_allclose(m_b[k], m_a[k], rtol=1e-5, atol=1e-6)
  for x, y in zip(
      jax.tree_util.tree_leaves(s_a.params),
      jax.tree_util.tree_leaves(s_b.params),
  ):
    np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-5, rtol=0)


def test_fixed_minibatch_path():
  lrn, s, tpol, tparams, batch = _setup(
      _cfg(minibatch_size=16, max_decision_share=None)
  )
  lrn.set_teacher(tpol, tparams, name="t")
  _, m = lrn.update(s, batch, jax.random.key(1))
  assert np.isfinite(m["teacher_kl"]) and m["teacher_kl"] > 0


def test_registered_loss_refuses_teacher_coef():
  with pytest.raises(ValueError, match="pg_loss"):
    pg.PGConfig(
        loss=toy_loss.NAME,
        anneal_lr=False,
        teacher_coef=COEF,
        loss_config=toy_loss.ToyLossConfig(),
    )


if __name__ == "__main__":
  from open_spiel.python.scale.core import x64  # pylint: disable=g-import-not-at-top

  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
