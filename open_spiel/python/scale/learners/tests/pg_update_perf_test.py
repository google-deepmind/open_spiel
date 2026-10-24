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

"""The PG update's two speed paths give the update they replace.

The PG update's two speed paths give the update they replace:

* gradient accumulation (pg.py module doc "Forward chunking"): a minibatch
above forward_chunk_rows takes ppo / mmd's loss chunk by chunk under the
minibatch's pg_stats; its loss, metrics and gradient equal the one-pass
loss's (the rematerialized chunk scan), for flat plies and for [B, T]
segments with a carry, with every pg_loss option that reads the whole
minibatch (advantage normalization, the advantage filter, value clipping,
HL-Gauss);
* PGConfig.magnet_from_behavior: mmd with the magnet re-taken every update
reads the magnet's logits from the batch's behaviour logits; on a batch
the starting params played, the update equals the one that runs the
magnet's forward. An option at its default does not enter the config
hash.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss

tree_map = jax.tree_util.tree_map
F32 = jnp.float32


def _cfg(**kw):
  """A small PGConfig; KL off for PPO.

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
      pg=loss_lib.PGLossConfig(clip_eps=0.2, vf_coef=0.5),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.2),
  )
  base.update(kw)
  if base["loss"] == "ppo":
    base["kl_coef"] = schedules.Schedule("constant", 0.0)
  return pg.PGConfig(**base)


def _flat_minibatch(learner, s, batch):
  """Every ply of the batch as one flat minibatch (time-major).

  Every ply of the batch as one flat minibatch (time-major), with the
  magnet's logits as the update computes them.

  Args:
    learner: the PGLearner.
    s: its state.
    batch: the TrajectoryBatch.

  Returns:
    The flat minibatch rows.
  """
  # Reaches private _prepare of the code under test.
  prep, masks = learner._prepare(s, batch)  # pylint: disable=protected-access
  # Reaches private _rows of the code under test.
  rows = learner._rows(batch, prep, masks)  # pylint: disable=protected-access
  if s.magnet is not None and "magnet_logits" not in rows:
    # Reaches private _magnet_logits of the code under test.
    rows["magnet_logits"] = learner._magnet_logits(s.magnet, rows, None)  # pylint: disable=protected-access
  # Reaches private _time_major of the code under test.
  return tree_map(learner._time_major, rows)  # pylint: disable=protected-access


def _both(learner, s, mb, carry):
  """The loss and gradient computed in one pass and accumulated.

  Args:
    learner: The learner.
    s: Its state.
    mb: The minibatch.
    carry: The carry.

  Returns:
    (one-pass result, accumulated result).
  """
  kl, ent = learner.cfg.kl_coef.f32(0), learner.cfg.ent_coef.f32(0)
  args = (
      s.params,
      s.magnet,
      mb,
      carry,
      mb["targets"],
      mb["value_old"],
      kl,
      ent,
      F32(0.0),
      jnp.asarray(False),
  )
  with jax.default_matmul_precision(
      "highest"
  ):  # a GPU's default f32 matmul is TF32
    one = jax.jit(lambda: learner.loss_and_grad(*args, accumulate=False))()
    acc = jax.jit(lambda: learner.loss_and_grad(*args, accumulate=True))()
  return one, acc


def _assert_same(one, acc, tol=2e-5):
  (m1, g1), (m2, g2) = one, acc
  assert set(m1) == set(m2), set(m1) ^ set(m2)
  for k in m1:
    np.testing.assert_allclose(
        np.asarray(m2[k]), np.asarray(m1[k]), rtol=tol, atol=tol, err_msg=k
    )
  for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2)):
    scale = max(float(jnp.max(jnp.abs(a))), 1e-6)
    np.testing.assert_allclose(
        np.asarray(b) / scale, np.asarray(a) / scale, atol=tol
    )


PG_OPTIONS = {
    "mmd": dict(loss="mmd"),
    "ppo": dict(loss="ppo"),
    "mmd_filter_clip": dict(
        loss="mmd",
        pg=loss_lib.PGLossConfig(
            clip_eps=0.2,
            vf_coef=0.5,
            clip_value=True,
            use_advantage_filter=True,
            advantage_filter_quantile=0.5,
            advantage_filter_floor=0.0,
        ),
    ),
    "mmd_hl_gauss": dict(
        loss="mmd",
        pg=loss_lib.PGLossConfig(
            clip_eps=0.1, vf_coef=0.5, value_head="hl_gauss"
        ),
    ),
    "ppo_unnormalized": dict(
        loss="ppo", pg=loss_lib.PGLossConfig(normalize_advantages=False)
    ),
}


@pytest.mark.parametrize("name", sorted(PG_OPTIONS))
def test_accumulated_gradient_equals_one_pass_on_flat_plies(name):
  """Accumulated gradient equals one pass on flat plies."""
  kw = dict(PG_OPTIONS[name])
  value = "hl_gauss" if name == "mmd_hl_gauss" else "scalar"
  pol = fx.make_policy("leduc_poker", fx.mlp(value=value))
  learner = pg.PGLearner(pol, _cfg(forward_chunk_rows=7, **kw))
  s = learner.init(jax.random.key(0), fx.make_env("leduc_poker"))
  batch = fx.self_play_batch(
      pol, s.params, game="leduc_poker", b=8, t=10, seed=3
  )
  mb = _flat_minibatch(learner, s, batch)
  n = mb["valid"].shape[
      0
  ]  # 8 tables x 2 seats x 10 plies: 23 chunks of 7, one pad row
  assert n == 160 and learner.accumulates(n)
  assert 0 < int(mb["valid"].sum()) < n
  _assert_same(*_both(learner, s, mb, jnp.zeros((), F32)))


def test_accumulated_gradient_equals_one_pass_on_segments_with_a_carry():
  """Accumulated gradient equals one pass on segments with a carry."""
  # Reaches private _magnet_logits of the code under test.
  # pylint: disable=protected-access
  pol = fx.make_policy("leduc_poker", fx.rnn())
  learner = pg.PGLearner(pol, _cfg(shuffle="env_axis", forward_chunk_rows=3))
  s = learner.init(jax.random.key(0), fx.make_env("leduc_poker"))
  src = fx.source(pol, s.params, "leduc_poker", b=8, t=5)
  src.next(jax.random.key(1))
  batch = src.next(jax.random.key(2))  # starts mid-hand: a non-zero init carry
  assert np.abs(np.asarray(batch.init_carry["h"])).max() > 0
  # Reaches private _prepare of the code under test.
  prep, masks = learner._prepare(s, batch)  # pylint: disable=protected-access
  # Reaches private _rows of the code under test.
  rows = learner._rows(batch, prep, masks)  # pylint: disable=protected-access
  rows["magnet_logits"] = learner._magnet_logits(
      s.magnet,
      rows,
      # Test access to private members.
      batch.init_carry,
  )  # pylint: disable=protected-access
  assert rows["valid"].shape == (16, 5) and learner.accumulates(
      16
  )  # 6 chunks of 3, 2 pads
  _assert_same(*_both(learner, s, rows, batch.init_carry))


def test_registered_losses_keep_the_one_pass_gradient():
  """Registered losses keep the one pass gradient."""
  pol = fx.make_policy("kuhn_poker")
  learner = pg.PGLearner(
      pol,
      _cfg(
          loss=toy_loss.NAME,
          kl_coef=schedules.Schedule("constant", 0.0),
          forward_chunk_rows=4,
      ),
  )
  assert not learner.accumulates(10**6)
  assert pg.PGLearner(pol, _cfg(forward_chunk_rows=4)).accumulates(5)
  assert not pg.PGLearner(pol, _cfg(forward_chunk_rows=4)).accumulates(4)


@pytest.mark.parametrize("loss", ["mmd", "ppo"])
def test_update_with_accumulation_equals_the_unchunked_update(loss):
  """Update with accumulation equals the unchunked update."""
  pol = fx.make_policy("leduc_poker")
  big = pg.PGLearner(pol, _cfg(loss=loss, forward_chunk_rows=10**9))
  small = pg.PGLearner(pol, _cfg(loss=loss, forward_chunk_rows=9))
  s = big.init(jax.random.key(0), fx.make_env("leduc_poker"))
  batch = fx.self_play_batch(
      pol, s.params, game="leduc_poker", b=8, t=10, seed=5
  )
  with jax.default_matmul_precision(
      "highest"
  ):  # a GPU's default f32 matmul is TF32
    s2b, mb = big.update(s, batch, jax.random.key(4))
    s2s, ms = small.update(
        small.init_from_params(s.params), batch, jax.random.key(4)
    )
  assert fx.max_abs_diff(s2b.params, s2s.params) < 1e-5
  for k in (
      "loss_total",
      "loss_policy",
      "loss_value",
      "entropy",
      "approx_kl",
      "grad_norm",
  ):
    np.testing.assert_allclose(ms[k], mb[k], rtol=1e-4, atol=1e-6, err_msg=k)


# -- magnet_from_behavior ------------------------------------------------------


def test_magnet_from_behavior_config():
  """Magnet from behavior config."""
  assert not pg.PGConfig(num_updates=1).magnet_from_behavior
  _cfg(magnet_from_behavior=True)
  with pytest.raises(ValueError, match="magnet_from_behavior"):
    _cfg(loss="ppo", magnet_from_behavior=True)
  with pytest.raises(ValueError, match="magnet_from_behavior"):
    _cfg(magnet_from_behavior=True, reference_update_every=2)
  with pytest.raises(ValueError, match="bool"):
    _cfg(magnet_from_behavior=1)
  off = common.config_dict(_cfg())["magnet_from_behavior"]
  assert isinstance(off, bool) and not off
  on = common.config_dict(_cfg(magnet_from_behavior=True))[
      "magnet_from_behavior"
  ]
  assert isinstance(on, bool) and on


@pytest.mark.parametrize("minibatch_size", [0, 40])
def test_magnet_from_behavior_update_equals_the_magnet_forward(minibatch_size):
  """Magnet from behavior update equals the magnet forward."""
  pol = fx.make_policy("leduc_poker")
  kw = dict(minibatch_size=minibatch_size)
  fwd = pg.PGLearner(pol, _cfg(**kw))
  beh = pg.PGLearner(pol, _cfg(magnet_from_behavior=True, **kw))
  s = fwd.init(jax.random.key(0), fx.make_env("leduc_poker"))
  batch = fx.self_play_batch(
      pol, s.params, game="leduc_poker", b=8, t=10, seed=7
  )
  # the stored behaviour logits are the starting params' logits where a seat
  # acted
  logits, _, _ = fwd.forward(
      s.params,
      None,
      tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batch.steps),
  )
  v = np.asarray(batch.valid).reshape(-1)
  np.testing.assert_allclose(
      np.asarray(batch.behavior_logits).reshape(v.size, -1)[v],
      np.asarray(logits)[v],
      atol=1e-5,
  )
  key = jax.random.key(4)
  with jax.default_matmul_precision(
      "highest"
  ):  # a GPU's default f32 matmul is TF32
    s2f, mf = fwd.update(s, batch, key)
    s2b, mbh = beh.update(beh.init_from_params(s.params), batch, key)
  assert fx.max_abs_diff(s2f.params, s2b.params) < 1e-5
  for k in ("loss_total", "kl_penalty", "loss_policy"):
    np.testing.assert_allclose(mbh[k], mf[k], rtol=1e-4, atol=1e-6, err_msg=k)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
