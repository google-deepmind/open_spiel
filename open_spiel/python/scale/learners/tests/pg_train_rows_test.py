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

"""PGLearner's decisions-only flat shuffle and once-per-update magnet forward.

PGLearner's decisions-only flat shuffle (train_rows) and the once-per-update
magnet forward: the static size, masks (every decision is trained exactly
once per epoch), determinism, and equality of the hoisted magnet logits with
the per-row forward.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx

tree_map = jax.tree_util.tree_map


def _cfg(**kw):
  """A small PGConfig.

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
  return pg.PGConfig(**base)


def _batch(b, t, all_seats=True, network=None, seed=1, game="kuhn_poker"):
  """A policy, its params and a self-play batch.

  Args:
    b: The rows.
    t: The plies.
    all_seats: Whether every seat is a row.
    network: The network config, or None for the default.
    seed: The seed.
    game: The game.

  Returns:
    (policy, params, batch).
  """
  pol = fx.make_policy(game, network)
  params = pol.init_params(jax.random.key(0))
  return (
      pol,
      params,
      fx.self_play_batch(
          pol, params, game=game, b=b, t=t, seed=seed, all_seats=all_seats
      ),
  )


def test_self_play_size_known_answer():
  """Self play size known answer.

  64 Arena rows x 2 seats x 32 plies = 4096 plies; one seat acts per ply: at
  most 64 x 32 = 2048 decisions. Kuhn max_len 3: every 4 plies of a row hold a
  terminal ply, so at most 64 x (32 - 8) = 1536.
  """
  pol, _, batch = _batch(64, 32)
  assert batch.valid.shape == (128, 32)
  n_valid = int(np.asarray(batch.valid).sum())
  assert pg.PGLearner(pol, _cfg()).train_rows(batch) == 2048
  learner = pg.PGLearner(pol, _cfg(), env_max_len=fx.make_env().max_len)
  assert fx.make_env().max_len == 3
  assert learner.train_rows(batch) == 1536 >= n_valid, n_valid


def test_seat_layout_size_known_answer():
  """One perspective seat per row (rows alternate seats): the seat layout.

  Provable bound: 128 rows x (32 - floor(32 / 4)) = 3072 plies (a multiple of 2
  minibatches x 128); a declared share 0.5 tightens it to 2048.
  """
  pol, _, batch = _batch(128, 32, all_seats=False)
  assert batch.valid.shape == (128, 32)
  n_valid = int(np.asarray(batch.valid).sum())
  learner = pg.PGLearner(pol, _cfg(), env_max_len=3)
  assert learner.train_rows(batch) == 3072
  # Reaches private _train_rows of the code under test.
  assert learner._train_rows[(128, 32, 1)][0] == "seat"  # pylint: disable=protected-access
  declared = pg.PGLearner(pol, _cfg(max_decision_share=0.5), env_max_len=3)
  assert declared.train_rows(batch) == 2048 >= n_valid, n_valid
  # Without env_max_len nothing bounds a seat row below T: every ply,
  # time-major.
  assert pg.PGLearner(pol, _cfg()).train_rows(batch) is None


def test_full_batch_when_the_bound_covers_every_ply_or_the_batch_is_small():
  """Full batch when the bound covers every ply or the batch is small."""
  pol, _, batch = _batch(64, 32)
  every = batch._replace(
      valid=jnp.ones_like(batch.valid)
  )  # every ply a decision
  assert (
      pg.PGLearner(pol, _cfg()).train_rows(every) is None
  )  # no env_max_len: bound = B x T
  _, _, small = _batch(4, 8)  # 64 plies: the 256-row quantum is the whole batch
  assert pg.PGLearner(pol, _cfg(), env_max_len=3).train_rows(small) is None


def test_decisions_over_the_bound_raise_instead_of_dropping():
  """Decisions over the bound raise instead of dropping.

  Every ply valid contradicts env_max_len 3 (every 4th ply of a row is a
  terminal ply): 4096 decisions > the seat bound 3072 raise, nothing is cut.
  """
  pol, _, batch = _batch(64, 32)
  every = batch._replace(valid=jnp.ones_like(batch.valid))
  learner = pg.PGLearner(pol, _cfg(), env_max_len=3)
  with pytest.raises(pg.TrainRowsOverflowError, match="by 1024"):
    learner.train_rows(every)


def test_declared_share_overflow_fails_loud():
  """Declared share overflow fails loud."""
  pol, params, batch = _batch(128, 32, all_seats=False)
  n_valid = int(np.asarray(batch.valid).sum())
  learner = pg.PGLearner(pol, _cfg(max_decision_share=0.125), env_max_len=3)
  assert n_valid > 512, n_valid  # 0.125 x 4096 = 512 plies (2 x 256)
  s = learner.init_from_params(params)
  with pytest.raises(
      pg.TrainRowsOverflowError, match="max_decision_share=0.125"
  ):
    learner.update(s, batch, jax.random.key(0))
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == 0  # pylint: disable=protected-access


def test_static_size_is_pinned_per_shape_one_compile():
  """Static size is pinned per shape one compile.

  The size is fixed by the first batch of a shape: a later batch of that shape
  with fewer decisions trains at the same size (one compiled update), and one
  with more decisions than the size raises (no second program).
  """
  pol, params, batch = _batch(64, 32)
  learner = pg.PGLearner(pol, _cfg(), env_max_len=3)
  s = learner.init_from_params(params)
  _, m1 = learner.update(s, batch, jax.random.key(1))
  assert m1["train_rows"] == 1536.0
  assert m1["train_rows_headroom"] == 1536.0 - m1["decisions"] >= 0.0
  _, _, other = _batch(64, 32, seed=5)
  fewer = other._replace(valid=other.valid & (jnp.arange(32)[None, :] < 16))
  _, m2 = learner.update(s, fewer, jax.random.key(1))
  assert m2["train_rows"] == 1536.0 and m2["decisions"] < m1["decisions"]
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == 1  # pylint: disable=protected-access
  seat_like = batch._replace(
      valid=jnp.ones_like(batch.valid) & (batch.steps.to_move >= 0)
  )
  with pytest.raises(pg.TrainRowsOverflowError):
    learner.update(s, seat_like, jax.random.key(1))
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == 1  # pylint: disable=protected-access


def test_grouped_layout_needs_one_game_per_row_group():
  """Grouped layout needs one game per row group.

  A self-play batch whose rows are regrouped (row pairs from different Arena
  rows) is the seat layout, not grouped.
  """
  pol, _, batch = _batch(64, 32)
  learner = pg.PGLearner(pol, _cfg(), env_max_len=3)
  valid = np.asarray(batch.valid)
  # Reaches private _grouped of the code under test.
  assert learner._grouped(batch, valid)  # pylint: disable=protected-access
  perm = np.arange(128).reshape(64, 2)
  perm[:, 1] = np.roll(
      perm[:, 1], 2
  )  # seat-1 rows of other Arena rows (seat order kept)
  mixed = tree_map(lambda x: x[perm.reshape(-1)], batch)
  # Reaches private _grouped of the code under test.
  assert not learner._grouped(mixed, np.asarray(mixed.valid))  # pylint: disable=protected-access


def test_env_axis_keeps_whole_segments():
  """Env axis keeps whole segments."""
  pol = fx.make_policy("kuhn_poker", fx.rnn())
  batch = fx.self_play_batch(
      pol, pol.init_params(jax.random.key(0)), b=64, t=32
  )
  assert (
      pg.PGLearner(pol, _cfg(shuffle="env_axis"), env_max_len=3).train_rows(
          batch
      )
      is None
  )


def test_every_decision_is_trained_exactly_once_per_epoch():
  """Every decision is trained exactly once per epoch.

  Known answer for the compacted minibatches: tag each ply with its flat index;
  across one epoch's minibatches every valid ply appears once and every other
  row is invalid padding.
  """
  pol, _, batch = _batch(64, 32)
  learner = pg.PGLearner(pol, _cfg(num_minibatches=4), env_max_len=3)
  cap = learner.train_rows(batch)
  b, t = batch.valid.shape
  rows = {
      "valid": batch.valid,
      "tag": jnp.arange(b * t, dtype=jnp.int32).reshape(b, t),
  }
  # Reaches private _compact of the code under test.
  flat = learner._compact(rows, cap)  # pylint: disable=protected-access
  assert flat["valid"].shape == (cap,)
  # Reaches private _minibatches of the code under test.
  mbs, carries = learner._minibatches(jax.random.key(4), flat, ())  # pylint: disable=protected-access
  assert carries is None and mbs["valid"].shape == (4, cap // 4)
  valid = np.asarray(mbs["valid"]).ravel()
  tags = np.asarray(mbs["tag"]).ravel()
  want = np.nonzero(np.asarray(batch.valid).ravel())[0]
  assert sorted(tags[valid].tolist()) == sorted(want.tolist())
  assert valid.sum() == len(want) and len(set(tags.tolist())) == cap


def test_compacted_update_is_deterministic_and_trains_the_same_decisions():
  """Compacted update is deterministic and trains the same decisions."""
  pol, params, batch = _batch(64, 32)
  learner = pg.PGLearner(pol, _cfg(), env_max_len=3)
  s = learner.init_from_params(params)
  s1, m1 = learner.update(s, batch, jax.random.key(7))
  s2, m2 = learner.update(s, batch, jax.random.key(7))
  assert fx.leaves_equal(s1.params, s2.params) and m1 == m2
  assert m1["decisions"] == float(np.asarray(batch.valid).sum())
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == 1  # pylint: disable=protected-access


@pytest.mark.parametrize("shuffle", ["flat_compacted", "flat_full", "env_axis"])
def test_hoisted_magnet_logits_equal_the_per_row_forward(shuffle):
  """Hoisted magnet logits equal the per row forward."""
  network = fx.rnn() if shuffle == "env_axis" else None
  pol, params, batch = _batch(64, 32, network=network)
  learner = pg.PGLearner(
      pol, _cfg(shuffle="env_axis" if shuffle == "env_axis" else "flat")
  )
  magnet = tree_map(lambda x: x * 0.5, params)
  rows = {"steps": batch.steps, "valid": batch.valid}
  if shuffle == "flat_compacted":
    # Reaches private _compact of the code under test.
    rows = learner._compact(rows, learner.train_rows(batch))  # pylint: disable=protected-access
  # Reaches private _magnet_logits of the code under test.
  got = learner._magnet_logits(magnet, rows, batch.init_carry)  # pylint: disable=protected-access
  if shuffle == "env_axis":
    want, _, _ = learner.forward(magnet, batch.init_carry, batch.steps)
  elif shuffle == "flat_full":
    b, t = batch.valid.shape
    flat = tree_map(lambda x: x.reshape((b * t,) + x.shape[2:]), batch.steps)
    want = learner.forward(magnet, None, flat)[0].reshape(b, t, -1)
  else:
    want, _, _ = learner.forward(magnet, None, rows["steps"])
  assert got.shape == want.shape and got.dtype == jnp.float32
  np.testing.assert_allclose(
      np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6
  )


# -- bucketed sizes (PGConfig.train_rows_buckets) ------------------------------
def _seat_batch_with(n_keep, seed=1):
  """The 128 x 32 seat-layout batch with only n_keep decisions valid.

  The 128 x 32 seat-layout batch with only its first n_keep decisions
  (time-major order) kept valid.

  Args:
    n_keep: the decisions kept valid.
    seed: the batch seed.

  Returns:
    (policy, params, batch).
  """
  pol, params, batch = _batch(128, 32, all_seats=False, seed=seed)
  v = np.asarray(batch.valid)
  flat = np.swapaxes(v, 0, 1).ravel()
  keep = np.cumsum(flat) <= n_keep
  valid2 = np.swapaxes((flat & keep).reshape(32, 128), 0, 1)
  return pol, params, batch._replace(valid=jnp.asarray(valid2))


def test_bucket_sizes_known_answer():
  """Bucket sizes known answer.

  128 x 32 = 4096 plies, seat bound 3072 (env_max_len 3), quantum 2 x 128:
  shares 0.2 / 0.25 / 0.5 / 0.9 give ceil(819.2) -> 1024, 1024 (merged), 2048,
  and 3687 -> 3840 >= 3072 (dropped: above the bound). A batch trains at the
  smallest bucket holding its decisions, the bound when none does.
  """
  pol, _, batch = _seat_batch_with(10_000)
  learner = pg.PGLearner(
      pol, _cfg(train_rows_buckets=(0.2, 0.25, 0.5, 0.9)), env_max_len=3
  )
  n_valid = int(np.asarray(batch.valid).sum())
  # Reaches private _bucket_rows of the code under test.
  assert learner._bucket_rows(128, 32, 3072) == (1024, 2048)  # pylint: disable=protected-access
  want = 1024 if n_valid <= 1024 else (2048 if n_valid <= 2048 else 3072)
  assert learner.train_rows(batch) == want
  assert learner.train_rows_bound(batch) == 3072
  for n_keep, size in ((1, 1024), (1024, 1024), (1025, 2048), (2048, 2048)):
    _, _, b2 = _seat_batch_with(n_keep)
    if int(np.asarray(b2.valid).sum()) == n_keep:
      assert learner.train_rows(b2) == size, (n_keep, size)
  # every ply trains (no env_max_len, c = None): buckets still compact below B x
  # T
  free = pg.PGLearner(pol, _cfg(train_rows_buckets=(0.25,)))
  _, _, small = _seat_batch_with(500)
  assert free.train_rows(small) == 1024
  _, _, big = _seat_batch_with(1100)
  assert int(np.asarray(big.valid).sum()) > 1024
  assert free.train_rows(big) is None


def test_buckets_bound_the_compiled_programs_and_drop_nothing():
  """Buckets bound the compiled programs and drop nothing.

  Batches of varying decision counts compile one update per bucket used, at most
  len(buckets) + 1; each trains every decision exactly once per epoch; the
  metrics report the size trained and the bound.
  """
  pol, params, _ = _seat_batch_with(10_000)
  learner = pg.PGLearner(
      pol, _cfg(train_rows_buckets=(0.25, 0.5)), env_max_len=3
  )
  s = learner.init_from_params(params)
  seen = set()
  for i, n_keep in enumerate((300, 900, 1500, 1000, 2000, 400)):
    _, _, b = _seat_batch_with(n_keep, seed=1 + i)
    n_valid = int(np.asarray(b.valid).sum())
    cap = learner.train_rows(b)
    assert cap in (1024, 2048, 3072) and cap >= n_valid
    seen.add(cap)
    _, m = learner.update(s, b, jax.random.key(i))
    assert m["train_rows"] == float(cap) and m["decisions"] == float(n_valid)
    assert m["train_rows_bound"] == 3072.0 and m["train_rows_headroom"] >= 0.0
    # no silent drop: the compacted rows hold every decision once
    rows = {
        "valid": b.valid,
        "tag": jnp.arange(128 * 32, dtype=jnp.int32).reshape(128, 32),
    }
    # Reaches private _compact of the code under test.
    flat = learner._compact(rows, cap)  # pylint: disable=protected-access
    want = np.nonzero(np.asarray(b.valid).ravel())[0]
    got = np.asarray(flat["tag"])[np.asarray(flat["valid"])]
    assert sorted(got.tolist()) == sorted(want.tolist())
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == len(seen) <= 3  # pylint: disable=protected-access


def test_buckets_still_raise_over_the_bound():
  """Buckets still raise over the bound."""
  pol, params, batch = _batch(64, 32)
  every = batch._replace(valid=jnp.ones_like(batch.valid))
  learner = pg.PGLearner(
      pol, _cfg(train_rows_buckets=(0.25, 0.5)), env_max_len=3
  )
  with pytest.raises(pg.TrainRowsOverflowError, match="by 1024"):
    learner.update(learner.init_from_params(params), every, jax.random.key(0))
  # Reaches private _cache_size, _update_jit of the code under test.
  assert learner._update_jit._cache_size() == 0  # pylint: disable=protected-access


def test_no_buckets_is_the_one_static_size():
  """The default (no buckets) trains at the single static size."""
  pol, params, batch = _batch(64, 32)
  a = pg.PGLearner(pol, _cfg(), env_max_len=3)
  b = pg.PGLearner(pol, _cfg(train_rows_buckets=()), env_max_len=3)
  assert a.train_rows(batch) == b.train_rows(batch) == 1536
  sa, ma = a.update(a.init_from_params(params), batch, jax.random.key(3))
  sb, mb = b.update(b.init_from_params(params), batch, jax.random.key(3))
  assert (
      fx.leaves_equal(sa.params, sb.params)
      and ma == mb
      and "train_rows_bound" not in ma
  )


@pytest.mark.parametrize(
    "bad",
    [
        (0.5, 0.5),
        (0.6, 0.5),
        (0.0,),
        (1.5,),
        tuple(0.1 * i for i in range(1, 10)),
        [0.5],
    ],
)
def test_bucket_config_refusals(bad):
  """Bucket config refusals."""
  with pytest.raises((AssertionError, ValueError)):
    _cfg(train_rows_buckets=bad)


def test_buckets_need_the_flat_shuffle():
  """Buckets need the flat shuffle."""
  with pytest.raises(ValueError, match="flat"):
    _cfg(shuffle="env_axis", train_rows_buckets=(0.5,))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
