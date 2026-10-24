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

"""64-bit discipline: the package runs with jax_enable_x64 on.

Importing the package leaves jax's config alone. The package's
`enable_x64()` (re-exported here) turns x64 on; the entry points (the
trainer, the league runner's run(), the launcher, the policy server) call
it, and so must code that uses the components directly, once, before it
creates arrays. The components (the Arena, decide, artifact reads and
writes, learners, serving) call `assert_x64()`, so a process that never
turned it on, or switched it off, fails instead of silently computing in 32
bits: an int32 frame counter wraps at 2**31, a float64 leaf restores as
float32 and decide's f64 becomes f32.

`fold_in64` is the int64 RNG fold-in: jax.random.fold_in takes its data mod
2**32, so counter n and n + 2**32 would give the same key. Every counter
(frames, updates, env steps, plies, hands, chance events) is int64 and is
folded in through `fold_in64`.
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import frameworks as _frameworks

enable_x64 = _frameworks.enable_x64


#: jax_enable_x64 is off in a process that needs it on.
NotEnabledError = _frameworks.X64NotEnabledError


def x64_enabled() -> bool:
  """Whether jax_enable_x64 is on.

  Returns:
    the flag.
  """
  return bool(jax.config.read("jax_enable_x64"))


def assert_x64(where: str = "") -> None:
  """Raises NotEnabledError unless jax_enable_x64 is on.

  Args:
    where: the caller, for the message.

  Raises:
    NotEnabledError: jax_enable_x64 is off.
  """
  if not x64_enabled():
    raise NotEnabledError(_frameworks.x64_off_message(where))


def _select_key(pred: Any, a: Any, b: Any) -> Any:
  """where(pred, a, b) for one typed or raw (uint32[2]) PRNG key.

  Args:
    pred: The condition.
    a: The key where it holds.
    b: The key elsewhere.

  Returns:
    The selected key.
  """
  if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
    data = jnp.where(pred, jax.random.key_data(a), jax.random.key_data(b))
    return jax.random.wrap_key_data(data, impl=jax.random.key_impl(a))
  return jnp.where(pred, a, b)


def fold_in64(key: Any, n: Any) -> Any:
  """jax.random.fold_in over an int64 counter, without the 2**32 wrap.

  Equals jax.random.fold_in(key, n) for 0 <= n < 2**32, so it is a drop-in
  replacement for fold_in on 32-bit counters; above that the high word is
  folded in too.

  Args:
    key: a typed jax key.
    n: an int64 counter.

  Returns:
    the folded key.
  """
  assert_x64("fold_in64")
  n = jnp.asarray(n)
  assert jnp.issubdtype(
      n.dtype, jnp.integer
  ), f"fold_in64 counter must be an integer, got {n.dtype}"
  assert (
      n.ndim == 0
  ), f"fold_in64 takes a scalar counter (vmap for batches), got {n.shape}"
  n = n.astype(jnp.int64)
  lo = jnp.bitwise_and(n, jnp.int64(0xFFFFFFFF)).astype(jnp.uint32)
  hi = jnp.right_shift(n, jnp.int64(32)).astype(jnp.uint32)
  k_lo = jax.random.fold_in(key, lo)
  return _select_key(hi != 0, jax.random.fold_in(k_lo, hi), k_lo)
