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

"""Threefry-2x32 keys and draws in numpy, equal bit for bit to jax.random.

The population-training control plane (samplers, controllers, the
evaluator's mixture draws, the league key schedule) draws random numbers on
the host and must run without jax. This module gives it jax.random's
results without importing jax. Two kinds of key are accepted:

  * a jax key (typed, or raw uint32[2] as a jax array): the call goes to
    jax.random itself and returns the same kind of key;
  * a numpy uint32[2] key (`key(seed)`): computed here in numpy, bit for bit
    what jax.random computes for the same key data under the threefry2x32
    implementation with jax_threefry_partitionable on (jax's default);
    core/tests/rng_matches_jax_test pins every function against jax.random.

So a seeded draw is the same with either kind of key. Draws return numpy
arrays (permutation int64, choice int64, uniform64 float64); keys are
uint32[..., 2].

  k = rng.key(0)
  order = rng.permutation(rng.fold_in(k, 3), 10)

Implemented draws (what the control plane uses; anything else raises
NotImplementedError rather than computing something jax would not):
split, fold_in, fold_in64, random bits (32 and 64), uniform64,
permutation(n), choice(n, shape, replace=False) without p, choice(n, shape,
p=...) with replacement.
"""

import math
from typing import Any, Sequence, Union

import numpy as np

from open_spiel.python.scale.core import frameworks

_U32 = np.uint32
_U64 = np.uint64
_MASK32 = 0xFFFFFFFF
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
_KS_CONSTANT = np.uint32(
    0x1BD11BDA
)  # the Threefry key schedule constant, from Skein
_UINT32_MAX = 4294967295
Shape = Union[int, Sequence[int]]


# -- the hash
# ------------------------------------------------------------------------------
def _rotl(x: np.ndarray, d: int) -> np.ndarray:
  """Rotates uint32 words left by `d` bits.

  Args:
    x: uint32 words.
    d: The rotation in bits.

  Returns:
    The rotated words.
  """
  return (x << _U32(d)) | (x >> _U32(32 - d))


def threefry2x32(k1, k2, x1, x2) -> tuple[np.ndarray, np.ndarray]:
  """Threefry-2x32 with 20 rounds (jax's hash).

  key (k1, k2), counts (x1, x2) uint32 arrays of one shape -> two uint32
  arrays of that shape.

  Args:
    k1: the key's first word (uint32).
    k2: the key's second word (uint32).
    x1: the first counts (uint32).
    x2: the second counts (uint32).

  Returns:
    (y1, y2), two uint32 arrays of the counts' shape.
  """
  k1, k2 = _U32(k1), _U32(k2)
  x1 = np.array(x1, dtype=_U32, ndmin=1)
  x2 = np.array(x2, dtype=_U32, ndmin=1)
  assert x1.shape == x2.shape, (x1.shape, x2.shape)
  ks = (
      np.array([k1], _U32),
      np.array([k2], _U32),
      np.array([k1 ^ k2 ^ _KS_CONSTANT], _U32),
  )
  x1 = x1 + ks[0]
  x2 = x2 + ks[1]
  for i in range(5):
    for r in _ROTATIONS[i % 2]:
      x1 = x1 + x2
      x2 = _rotl(x2, r)
      x2 = x1 ^ x2
    x1 = x1 + ks[(i + 1) % 3]
    x2 = x2 + ks[(i + 2) % 3] + np.array([i + 1], _U32)
  return x1, x2


# -- keys ----------------------------------------------------------------------
def is_jax_key(k: Any) -> bool:
  """Whether k is a jax array (a typed jax key), not a numpy key.

  Args:
    k: a key.

  Returns:
    True for a jax key.
  """
  return frameworks.is_jax_array(k)


def key(seed: int) -> np.ndarray:
  """The numpy key of an integer seed.

  jax.random.key(seed)'s key data (the seed's 64 bits as [high word, low
  word]).

  Args:
    seed: an integer in the int64 range.

  Returns:
    uint32[2], [high word, low word].

  Raises:
    TypeError: seed is not an integer.
    ValueError: seed is outside the int64 range.
  """
  if isinstance(seed, (bool, float)) or not isinstance(seed, (int, np.integer)):
    raise TypeError(f"rng.key: an integer seed, got {seed!r}")
  s = int(seed)
  if not -(2**63) <= s < 2**63:
    raise ValueError(f"rng.key: seed {s} is not an int64")
  u = s & 0xFFFFFFFFFFFFFFFF
  return np.array([u >> 32, u & _MASK32], _U32)


def _raw(k: Any) -> np.ndarray:
  """Checks `k` is a raw threefry key.

  Args:
    k: The key.

  Returns:
    The key as uint32[2].

  Raises:
    ValueError: `k` is not uint32[2].
  """
  a = np.asarray(k)
  if a.dtype != _U32 or a.shape != (2,):
    raise ValueError(
        f"a numpy threefry key is uint32[2], got {a.dtype}{list(a.shape)}"
    )
  return a


def to_jax_key(k: Any):
  """A typed jax key with k's data (a jax key is returned as is).

  Args:
    k: a numpy key or a typed jax key.

  Returns:
    the typed jax key.
  """
  # Lazy: jax is optional here (the jax-free control plane imports this module).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  if is_jax_key(k):
    return k
  return jax.random.wrap_key_data(
      jax.numpy.asarray(_raw(k)), impl="threefry2x32"
  )


def _shape(num: Shape) -> tuple[int, ...]:
  """A sample count or shape as a shape tuple.

  Args:
    num: An int or a sequence of ints.

  Returns:
    The shape.
  """
  shape = (
      tuple(int(n) for n in num)
      if isinstance(num, (tuple, list))
      else (int(num),)
  )
  assert all(n >= 0 for n in shape), shape
  return shape


def _counts(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
  """The uint64 iota over shape as (hi, lo) words (jax's iota_2x32_shape).

  Args:
    shape: The shape.

  Returns:
    (hi, lo), each uint32 of `shape`.
  """
  idx = np.arange(math.prod(shape), dtype=_U64)  # prod(()) == 1: one count, 0
  hi = (idx >> _U64(32)).astype(_U32).reshape(shape)
  lo = (idx & _U64(_MASK32)).astype(_U32).reshape(shape)
  return hi, lo


def _hash_counts(
    k: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """Threefry of the iota counts over `shape` under key `k`.

  Args:
    k: The raw key, uint32[2].
    shape: The shape.

  Returns:
    (bits1, bits2), each of `shape`.
  """
  hi, lo = _counts(shape)
  b1, b2 = threefry2x32(k[0], k[1], hi.reshape(-1), lo.reshape(-1))
  return b1[: hi.size].reshape(shape), b2[: hi.size].reshape(shape)


def split(k: Any, num: Shape = 2):
  """jax.random.split: keys [*num, 2] (jax keys: jax's split).

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    num: the number (or shape) of keys.

  Returns:
    the keys, [*num, 2] for a numpy key.
  """
  if is_jax_key(k):
    # Lazy: jax is optional here (the jax-free control plane imports this
    # module).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return jax.random.split(k, num)
  b1, b2 = _hash_counts(_raw(k), _shape(num))
  return np.stack([b1, b2], axis=-1)


def fold_in(k: Any, data: int):
  """jax.random.fold_in with data in [0, 2**32).

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    data: an integer in [0, 2**32).

  Returns:
    the folded key, of k's kind.

  Raises:
    TypeError: data is not an integer.
    ValueError: data is not a uint32.
  """
  if is_jax_key(k):
    # Lazy: jax is optional here (the jax-free control plane imports this
    # module).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return jax.random.fold_in(k, data)
  if isinstance(data, (bool, float)) or not isinstance(data, (int, np.integer)):
    raise TypeError(f"fold_in: an integer, got {data!r}")
  d = int(data)
  if not 0 <= d <= _MASK32:
    raise ValueError(
        f"fold_in: data {d} is not a uint32 (fold_in64 takes int64 counters)"
    )
  r = _raw(k)
  y1, y2 = threefry2x32(r[0], r[1], [0], [d])
  return np.array([y1[0], y2[0]], _U32)


def fold_in64(k: Any, n: int):
  """core.x64.fold_in64.

  fold_in of an int64 counter without the 2**32 wrap (equal to fold_in below
  2**32; above, the high word is folded in too).

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    n: an int64 counter.

  Returns:
    the folded key, of k's kind.

  Raises:
    TypeError: n is not an integer.
    ValueError: n is outside the int64 range.
  """
  if isinstance(n, (bool, float)) or not isinstance(n, (int, np.integer)):
    raise TypeError(f"fold_in64: an integer counter, got {n!r}")
  n = int(n)
  if not -(2**63) <= n < 2**63:
    raise ValueError(f"fold_in64: counter {n} is not an int64")
  if is_jax_key(k):
    # Lazy: core.x64 imports jax, which is optional here (the jax-free control
    # plane).
    from open_spiel.python.scale.core import x64  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return x64.fold_in64(k, np.int64(n))
  lo, hi = n & _MASK32, (n >> 32) & _MASK32
  out = fold_in(k, lo)
  return fold_in(out, hi) if hi != 0 else out


# -- draws ---------------------------------------------------------------------
def bits32(k: Any, shape: Shape) -> np.ndarray:
  """jax.random.bits(k, shape, uint32) of a numpy key.

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    shape: the output shape.

  Returns:
    uint32 bits of `shape`.
  """
  b1, b2 = _hash_counts(_raw(k), _shape(shape))
  return b1 ^ b2


def bits64(k: Any, shape: Shape) -> np.ndarray:
  """jax.random.bits(k, shape, uint64) of a numpy key.

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    shape: the output shape.

  Returns:
    uint64 bits of `shape`.
  """
  b1, b2 = _hash_counts(_raw(k), _shape(shape))
  return (b1.astype(_U64) << _U64(32)) | b2.astype(_U64)


def uniform64(k: Any, shape: Shape = ()) -> np.ndarray:
  """jax.random.uniform(k, shape, float64) in [0, 1).

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    shape: the output shape.

  Returns:
    float64 uniforms of `shape`.
  """
  if is_jax_key(k):
    # Lazy: jax is optional here (the jax-free control plane imports this
    # module).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return np.asarray(jax.random.uniform(k, _shape(shape), jax.numpy.float64))
  bits = bits64(k, shape)
  one = np.array(1.0, np.float64).view(_U64)
  floats = ((bits >> _U64(12)) | one).view(np.float64) - np.float64(1.0)
  return np.maximum(np.float64(0.0), floats)


def permutation(k: Any, n: int) -> np.ndarray:
  """jax.random.permutation(k, n) as int64 (the sort-based shuffle).

  The shuffle is rounds of a stable sort by fresh 32-bit keys.

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    n: the number of items.

  Returns:
    int64[n], a permutation of range(n).
  """
  n = int(n)
  assert n >= 0, n
  if is_jax_key(k):
    # Lazy: jax is optional here (the jax-free control plane imports this
    # module).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return np.asarray(jax.random.permutation(k, n))
  x = np.arange(n, dtype=np.int64)
  rounds = int(np.ceil(3 * np.log(max(1, n)) / np.log(_UINT32_MAX)))
  cur = _raw(k)
  for _ in range(rounds):
    pair = split(cur)
    cur, sub = pair[0], pair[1]
    sort_keys = bits32(sub, (n,))
    x = x[np.argsort(sort_keys, kind="stable")]
  return x


def choice(
    k: Any,
    n: int,
    shape: Sequence[int] = (),
    replace: bool = True,
    p: Any = None,
) -> np.ndarray:
  """jax.random.choice(k, n, shape, replace, p) for two forms.

  The two forms the control plane draws: without replacement and no p (the first
  prod(shape) of a permutation), and with replacement under p (inverse CDF of a
  float64 uniform). Returns int64 indices.

  Args:
    k: a numpy key uint32[2] or a typed jax key.
    n: the number of items.
    shape: the output shape.
    replace: draw with replacement (requires p).
    p: the item probabilities, or None (requires replace=False).

  Returns:
    int64 indices of `shape`.

  Raises:
    ValueError: n <= 0 with samples to take, or more draws without replacement
      than items.
    NotImplementedError: a form jax draws other than the two above.
  """
  n = int(n)
  shape = tuple(int(s) for s in shape)
  if is_jax_key(k):
    # Lazy: jax is optional here (the jax-free control plane imports this
    # module).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    pj = None if p is None else jax.numpy.asarray(p)
    return np.asarray(jax.random.choice(k, n, shape, replace=replace, p=pj))
  draws = math.prod(shape)
  if n <= 0 and draws:
    raise ValueError("choice: n must be > 0 unless no samples are taken")
  if p is None:
    if replace:
      raise NotImplementedError(
          "rng.choice: uniform draws with replacement (jax's randint) "
          "are not implemented; pass p"
      )
    if draws > n:
      raise ValueError(f"choice: {draws} draws without replacement from {n}")
    return permutation(k, n)[:draws].reshape(shape)
  if not replace:
    raise NotImplementedError(
        "rng.choice: weighted draws without replacement (jax's Gumbel "
        "top-k) are not implemented"
    )
  pa = np.asarray(p, np.float64)
  if pa.shape != (n,):
    raise ValueError(f"choice: p has shape {pa.shape}, needs ({n},)")
  if draws == 0:
    return np.zeros(shape, np.int64)
  cum = np.cumsum(pa)
  r = cum[-1] * (np.float64(1.0) - uniform64(k, shape))
  return np.searchsorted(cum, r, side="left").astype(np.int64)
