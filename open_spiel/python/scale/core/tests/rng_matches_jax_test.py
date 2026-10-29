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

"""core.rng's numpy path equals jax.random bit for bit.

With raw uint32[2] keys and no jax import, every draw the control plane makes
(key, split, fold_in, fold_in64, permutation, choice without replacement,
choice with p, uniform f64) equals jax.random under the threefry2x32
implementation with jax_threefry_partitionable on. A jax key goes to
jax.random itself and comes back as the same kind, so control-plane draws do
not depend on the kind of key.
"""

import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import slots
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import sampler


def _data(k):
  """The key data of `k`.

  Args:
    k: A typed key.

  Returns:
    The numpy key data.
  """
  return np.asarray(jax.random.key_data(k))


SEEDS = (0, 1, 7, 42, 12345, 2**31 - 1, 2**32 + 9, 2**62 + 3)


def test_partitionable_threefry_is_the_modelled_config():
  """The jax config is partitionable threefry, the config rng models."""
  assert jax.config.jax_threefry_partitionable
  assert str(jax.random.key_impl(jax.random.key(0))) in ("threefry2x32", "fry")


def test_key_split_fold_in_equal_jax():
  """key, split and fold_in equal jax's."""
  for s in SEEDS:
    jk, nk = jax.random.key(s), rng.key(s)
    np.testing.assert_array_equal(nk, _data(jk))
    for num in (2, 3, 4, 7, (2, 3)):
      np.testing.assert_array_equal(
          rng.split(nk, num), _data(jax.random.split(jk, num))
      )
    for d in (0, 1, 2, 977, 2**31, 2**32 - 1):
      np.testing.assert_array_equal(
          rng.fold_in(nk, d), _data(jax.random.fold_in(jk, d))
      )
    for n in (0, 5, 2**32 - 1, 2**32, 2**32 + 1, 2**40 + 17, 2**62):
      np.testing.assert_array_equal(
          rng.fold_in64(nk, n), _data(x64.fold_in64(jk, jnp.int64(n)))
      )


def test_permutation_equals_jax():
  """Permutation equals jax.random.permutation."""
  for s in SEEDS[:5]:
    jk, nk = jax.random.key(s), rng.key(s)
    for n in (0, 1, 2, 3, 10, 64, 513, 1625, 1626, 4096, 5000):
      np.testing.assert_array_equal(
          rng.permutation(nk, n), np.asarray(jax.random.permutation(jk, n))
      )


def test_choice_equals_jax():
  """Choice equals jax.random.choice in both implemented forms."""
  r = np.random.default_rng(0)
  for s in SEEDS[:5]:
    jk, nk = jax.random.key(s), rng.key(s)
    for n, m in ((5, 1), (5, 3), (40, 7), (4000, 2)):
      np.testing.assert_array_equal(
          rng.choice(nk, n, (m,), replace=False),
          np.asarray(jax.random.choice(jk, n, (m,), replace=False)),
      )
    for size in (1, 2, 3, 6, 33):
      p = r.random(size)
      p[r.random(size) < 0.3] = 0.0
      if p.sum() == 0:
        p[0] = 1.0
      p = p / p.sum()
      for shape in ((1,), (17,), (4096,), (3, 5)):
        np.testing.assert_array_equal(
            rng.choice(nk, size, shape, p=p),
            np.asarray(jax.random.choice(jk, size, shape, p=p)),
        )


def test_uniform64_equals_jax():
  """uniform64 equals jax.random.uniform at float64."""
  for s in SEEDS[:4]:
    jk, nk = jax.random.key(s), rng.key(s)
    for shape in ((), (1,), (9,), (4, 5)):
      np.testing.assert_array_equal(
          rng.uniform64(nk, shape),
          np.asarray(jax.random.uniform(jk, shape, jnp.float64)),
      )


def test_jax_keys_go_to_jax_random_and_stay_jax_keys():
  """A jax key goes to jax.random and stays a jax key."""
  jk = jax.random.key(3)
  s = rng.split(jk, 4)
  assert isinstance(s, jax.Array) and jax.dtypes.issubdtype(
      s.dtype, jax.dtypes.prng_key
  )
  np.testing.assert_array_equal(_data(s), _data(jax.random.split(jk, 4)))
  f = rng.fold_in(jk, 5)
  assert jax.dtypes.issubdtype(f.dtype, jax.dtypes.prng_key)
  np.testing.assert_array_equal(_data(f), _data(jax.random.fold_in(jk, 5)))
  g = rng.fold_in64(jk, 2**33 + 1)
  np.testing.assert_array_equal(
      _data(g), _data(x64.fold_in64(jk, jnp.int64(2**33 + 1)))
  )
  p = rng.permutation(jk, 9)
  assert isinstance(p, np.ndarray)
  np.testing.assert_array_equal(p, np.asarray(jax.random.permutation(jk, 9)))
  np.testing.assert_array_equal(
      rng.choice(jk, 4, (6,), p=np.array([0.1, 0.2, 0.3, 0.4])),
      np.asarray(
          jax.random.choice(
              jk, 4, (6,), p=jnp.asarray(np.array([0.1, 0.2, 0.3, 0.4]))
          )
      ),
  )
  assert rng.is_jax_key(jk) and not rng.is_jax_key(rng.key(3))


def test_to_jax_key_wraps_raw_data():
  """to_jax_key wraps raw key data."""
  nk = rng.split(rng.key(5), 3)[1]
  jk = rng.to_jax_key(nk)
  assert jax.dtypes.issubdtype(jk.dtype, jax.dtypes.prng_key)
  np.testing.assert_array_equal(_data(jk), nk)
  np.testing.assert_array_equal(_data(jax.random.split(jk)), rng.split(nk))
  assert rng.to_jax_key(jk) is jk


def test_arena_reexports_the_slot_records():
  """core.arena re-exports core.slots' records; a SlotTable pickles."""
  for name in ("Slot", "SlotTable", "TemperatureDrop", "IN_GRAPH"):
    assert getattr(arena, name) is getattr(slots, name), name
  t = arena.SlotTable(
      np.zeros((2, 2), np.int32),
      np.zeros((2, 2), np.int32),
      np.arange(2, dtype=np.int32),
  )
  # A round trip of bytes this test just pickled.
  # pylint: disable-next=g-unsafe-pickle-load
  assert pickle.loads(pickle.dumps(t)).__class__ is slots.SlotTable  # noqa: S301


@pytest.mark.parametrize("n", [3, 100])
def test_control_plane_draws_do_not_depend_on_key_kind(n):
  """Sampler and snapshot schedule draws do not depend on the key kind.

  Draws from rng.key(s) equal those from jax.random.key(s) (the League passes
  typed keys; jax-free callers raw).

  Args:
    n: the number of draws.
  """
  where = {"a": (1, 0), "b": (1, 1), "c": (2, 0)}
  for strat in (True, False):
    mix = (
        sampler.Mix(0.2, {"a": 0.5, "b": 0.3, "c": 0.0})
        if strat
        else sampler.Mix(0.1, {"a": 0.6, "c": 0.3})
    )
    for s in (0, 9):
      t1, c1 = sampler.table_from_mix(mix, where, 2 * n, 2, rng.key(s), strat)
      t2, c2 = sampler.table_from_mix(
          mix, where, 2 * n, 2, jax.random.key(s), strat
      )
      assert c1 == c2
      for x, y in zip(t1, t2):
        np.testing.assert_array_equal(x, y)
  mixs = controller.SnapshotSchedule(
      self_weight=0.9,
      snapshot_every=1,
      harvest_every=1,
      max_pool=4,
      pool_slots=2,
  )
  pool = tuple(range(1, n + 1))
  for s in range(20):
    assert mixs.active_pool(pool, rng.key(s)) == mixs.active_pool(
        pool, jax.random.key(s)
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
