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

"""core.rng without jax: known answers and contracts of the numpy path.

This file imports no jax, so it also runs with jax not importable; equality
with jax.random lives in rng_matches_jax_test.

Known answers: threefry2x32 (20 rounds, JAX's hash) against the Random123
known-answer vectors JAX's own tests pin; key(seed) is the seed's 64 bits as
two uint32 words (jax.random.key's key data). Contracts: keys are uint32[2]
numpy arrays, split adds a leading axis, every draw is a pure function of
(key, arguments), fold_in64 equals fold_in below 2**32 and does not wrap at
2**32, permutation returns a permutation of range(n) (int64), choice
without replacement draws distinct indices, choice with p never draws a
zero-mass index and matches p in frequency.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng


def _hex(a):
  """The words of `a` in hex.

  Args:
    a: uint32 words.

  Returns:
    A tuple of hex strings.
  """
  return tuple(f"{int(x):#010x}" for x in np.asarray(a).reshape(-1))


def test_threefry_known_answers():
  """Random123 threefry2x32_20 KATs (as pinned by jax's random_test)."""
  cases = [
      ((0, 0), (0, 0), ("0x6b200159", "0x99ba4efe")),
      (
          (0xFFFFFFFF, 0xFFFFFFFF),
          (0xFFFFFFFF, 0xFFFFFFFF),
          ("0x1cb996fc", "0xbb002be7"),
      ),
      (
          (0x13198A2E, 0x03707344),
          (0x243F6A88, 0x85A308D3),
          ("0xc4923a9c", "0x483df7a0"),
      ),
  ]
  for (k1, k2), (x1, x2), want in cases:
    y1, y2 = rng.threefry2x32(
        np.uint32(k1),
        np.uint32(k2),
        np.array([x1], np.uint32),
        np.array([x2], np.uint32),
    )
    assert y1.dtype == np.uint32 and y2.dtype == np.uint32
    assert _hex(np.concatenate([y1, y2])) == want


def test_threefry_is_elementwise_over_counts():
  """threefry2x32 is elementwise over its counts."""
  k1, k2 = np.uint32(0x13198A2E), np.uint32(0x03707344)
  x1 = np.array([0x243F6A88, 0, 7], np.uint32)
  x2 = np.array([0x85A308D3, 0, 9], np.uint32)
  y1, y2 = rng.threefry2x32(k1, k2, x1, x2)
  assert y1.shape == (3,) and y2.shape == (3,)
  assert _hex([y1[0], y2[0]]) == ("0xc4923a9c", "0x483df7a0")
  for i in range(3):
    z1, z2 = rng.threefry2x32(k1, k2, x1[i : i + 1], x2[i : i + 1])
    assert (z1[0], z2[0]) == (y1[i], y2[i])


def test_key_is_the_seed_as_two_words():
  """key(seed) is the seed's 64 bits as two uint32 words."""
  for seed, want in (
      (0, [0, 0]),
      (42, [0, 42]),
      (2**33 + 5, [2, 5]),
      (-1, [2**32 - 1] * 2),
  ):
    k = rng.key(seed)
    assert k.dtype == np.uint32 and k.shape == (2,)
    assert k.tolist() == want, (seed, k)
  with pytest.raises((ValueError, AssertionError)):
    rng.key(2**64)
  with pytest.raises((TypeError, AssertionError)):
    rng.key(1.5)


def test_split_shapes_dtypes_and_determinism():
  """Split's shapes, dtypes and determinism."""
  k = rng.key(7)
  s = rng.split(k)
  assert s.shape == (2, 2) and s.dtype == np.uint32
  assert rng.split(k, 5).shape == (5, 2)
  assert rng.split(k, (3, 4)).shape == (3, 4, 2)
  np.testing.assert_array_equal(rng.split(k, 5), rng.split(rng.key(7), 5))
  assert not np.array_equal(s[0], s[1])
  assert not np.array_equal(rng.split(rng.key(8)), s)
  # Partitionable split: the first n keys of split(k, m) are split(k, n) for n
  # <= m.
  np.testing.assert_array_equal(rng.split(k, 3), rng.split(k, 6)[:3])
  with pytest.raises((ValueError, AssertionError)):
    rng.split(np.zeros(3, np.uint32))


def test_fold_in_and_fold_in64():
  """fold_in and fold_in64 agree below 2**32; fold_in64 does not wrap above."""
  k = rng.key(3)
  a, b = rng.fold_in(k, 0), rng.fold_in(k, 1)
  assert a.shape == (2,) and a.dtype == np.uint32 and not np.array_equal(a, b)
  np.testing.assert_array_equal(rng.fold_in(k, 1), b)
  for n in (0, 1, 2**31, 2**32 - 1):
    np.testing.assert_array_equal(rng.fold_in64(k, n), rng.fold_in(k, n))
    np.testing.assert_array_equal(
        rng.fold_in64(k, np.int64(n)), rng.fold_in(k, n)
    )
  # Above 2**32 the high word is folded in too: no wrap.
  assert not np.array_equal(rng.fold_in64(k, 2**32), rng.fold_in64(k, 0))
  np.testing.assert_array_equal(
      rng.fold_in64(k, 2**32 + 3), rng.fold_in(rng.fold_in(k, 3), 1)
  )
  with pytest.raises((ValueError, AssertionError)):
    rng.fold_in(k, 2**32)
  with pytest.raises((ValueError, AssertionError)):
    rng.fold_in(k, -1)
  with pytest.raises((TypeError, AssertionError)):
    rng.fold_in64(k, 1.0)


def test_permutation_is_a_permutation():
  """Permutation returns a permutation of range(n)."""
  for n in (0, 1, 2, 3, 17, 1000, 5000):  # 5000: two sort rounds
    p = rng.permutation(rng.key(n), n)
    assert p.dtype == np.int64 and p.shape == (n,)
    np.testing.assert_array_equal(np.sort(p), np.arange(n))
  np.testing.assert_array_equal(
      rng.permutation(rng.key(1), 64), rng.permutation(rng.key(1), 64)
  )
  assert not np.array_equal(
      rng.permutation(rng.key(1), 64), rng.permutation(rng.key(2), 64)
  )


def test_permutation_is_uniform_on_three_elements():
  """Permutation is uniform over the 6 orders of 3 elements."""
  counts = {}
  for i in range(6000):
    t = tuple(rng.permutation(rng.fold_in(rng.key(0), i), 3).tolist())
    counts[t] = counts.get(t, 0) + 1
  assert len(counts) == 6
  sigma = np.sqrt(6000 * (1 / 6) * (5 / 6))
  assert all(abs(c - 1000) < 5 * sigma for c in counts.values()), counts


def test_choice_without_replacement():
  """Choice without replacement draws distinct items."""
  k = rng.key(11)
  d = rng.choice(k, 10, (4,), replace=False)
  assert (
      d.shape == (4,)
      and len(set(d.tolist())) == 4
      and set(d.tolist()) <= set(range(10))
  )
  np.testing.assert_array_equal(d, rng.permutation(k, 10)[:4])
  with pytest.raises(ValueError):
    rng.choice(k, 3, (4,), replace=False)


def test_choice_with_probabilities_known_masses():
  """Choice with p matches known masses."""
  p = np.array([0.1, 0.0, 0.6, 0.3])
  d = rng.choice(rng.key(5), 4, (20000,), p=p)
  assert d.shape == (20000,) and d.dtype == np.int64
  assert not np.any(d == 1)
  for i, w in enumerate(p):
    se = np.sqrt(w * (1 - w) / 20000)
    assert abs(np.mean(d == i) - w) <= 4 * se + 1e-12, (i, np.mean(d == i))
  np.testing.assert_array_equal(d, rng.choice(rng.key(5), 4, (20000,), p=p))
  one = rng.choice(rng.key(5), 4, (3,), p=np.array([0.0, 0.0, 1.0, 0.0]))
  assert one.tolist() == [2, 2, 2]
  with pytest.raises(ValueError):
    rng.choice(rng.key(5), 4, (3,), p=np.array([0.5, 0.5]))


def test_uniform64_is_in_unit_interval():
  """uniform64 is float64 in [0, 1) with mean 1/2."""
  u = rng.uniform64(rng.key(9), (10000,))
  assert u.dtype == np.float64 and u.shape == (10000,)
  assert np.all(u >= 0.0) and np.all(u < 1.0)
  assert abs(u.mean() - 0.5) < 5 * np.sqrt(1 / 12 / 10000)


def test_unsupported_draws_fail_loud():
  """The draws rng does not implement raise NotImplementedError."""
  with pytest.raises(NotImplementedError):
    rng.choice(rng.key(0), 4, (2,), replace=False, p=np.full(4, 0.25))
  with pytest.raises(NotImplementedError):
    rng.choice(rng.key(0), 4, (2,), replace=True)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
