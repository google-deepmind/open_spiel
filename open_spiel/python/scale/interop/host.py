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

"""Host-side tree helpers for HOST_DRIVEN policies.

A HOST_DRIVEN policy runs outside XLA: it receives numpy SeatSteps and raw
uint32[..., 2] keys, and may answer with arrays of its own framework. The
host rollout loop needs to copy those answers back to numpy and to gather and
scatter rows of them without knowing the framework. Each helper here works
leaf by leaf on numpy arrays and on any array type whose backend a plugin
registers. A jax array leaf goes through the numpy ops, so `take`, `put` and
`alloc_rows` return it as a host numpy array.

  host.register_array_backend(lambda x: isinstance(x, MyTensor), MyOps())
  rows = host.take(carry, np.array([0, 2]))
  carry = host.put(carry, np.array([0, 2]), rows)
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import numpy as np

tree_map = jax.tree_util.tree_map


class ArrayOps(NamedTuple):
  """Row operations on one array type.

  Attributes:
    take: take(x, idx) is rows idx of x (idx an int numpy array).
    put: put(x, idx, y) is a copy of x with rows idx replaced by y (x
      unchanged).
    zeros_like_rows: zeros_like_rows(x, n) is zeros of shape
      (n,) + x.shape[1:], x's dtype and placement (the numpy ops place them
      on the host).
    to_numpy: to_numpy(x) is a fresh host numpy copy of x.
  """

  take: Callable[[Any, np.ndarray], Any]
  put: Callable[[Any, np.ndarray, Any], Any]
  zeros_like_rows: Callable[[Any, int], Any]
  to_numpy: Callable[[Any], np.ndarray]


def _np_put(x, idx, y):
  """A copy of `x` with `x[idx] = y`.

  Args:
    x: The array.
    idx: The indices.
    y: The values.

  Returns:
    The copy.
  """
  out = np.array(x, copy=True)
  out[idx] = y
  return out


NUMPY = ArrayOps(
    take=lambda x, idx: np.asarray(x)[idx],
    put=_np_put,
    zeros_like_rows=lambda x, n: np.zeros((n,) + tuple(x.shape[1:]), x.dtype),
    to_numpy=lambda x: np.array(x, copy=True),
)

_BACKENDS: list[tuple[Callable[[Any], bool], ArrayOps]] = []


def register_array_backend(
    predicate: Callable[[Any], bool], ops: ArrayOps
) -> None:
  """Handles leaves where `predicate(x)` is True with `ops`.

  Backends are tried in registration order, before numpy; registering the same
  pair again has no effect.

  Raises:
    TypeError: predicate is not callable or ops is not an ArrayOps.
    ValueError: predicate is already registered with other ops.

  Args:
    predicate: leaf -> whether ops handles it.
    ops: the backend's ArrayOps.
  """
  if not callable(predicate) or not isinstance(ops, ArrayOps):
    raise TypeError(
        f"want a callable predicate and an ArrayOps, got {predicate!r}, {ops!r}"
    )
  for got_pred, got_ops in _BACKENDS:
    if got_pred is predicate:
      if got_ops is not ops:
        raise ValueError(
            f"array backend predicate {predicate!r} is already registered "
            "with other ops"
        )
      return
  _BACKENDS.append((predicate, ops))


def _ops(x: Any) -> ArrayOps:
  """The array ops of `x`'s backend.

  Args:
    x: An array.

  Returns:
    The ArrayOps.
  """
  for predicate, ops in _BACKENDS:
    if predicate(x):
      return ops
  return NUMPY


def to_numpy(tree: Any) -> Any:
  """Every leaf (jax, numpy or a registered backend's array) as a numpy copy.

  A typed jax PRNG key becomes its raw key data.

  Args:
    tree: a pytree.

  Returns:
    the pytree with numpy leaves.
  """

  def conv(x):
    """One leaf as a numpy copy.

    Args:
      x: the leaf.

    Returns:
      the numpy copy.
    """
    if isinstance(x, jax.Array):
      x = jax.block_until_ready(x)
      if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
      return np.array(x, copy=True)
    return _ops(x).to_numpy(x)

  return tree_map(conv, tree)


def raw_keys(keys: Any) -> np.ndarray:
  """Typed jax keys [...] as raw uint32[..., 2] numpy (host policies' form).

  Args:
    keys: typed jax keys.

  Returns:
    uint32[..., 2].
  """
  return np.array(jax.random.key_data(keys), copy=True)


def take(tree: Any, idx: np.ndarray) -> Any:
  """Rows `idx` of every leaf.

  Args:
    tree: a pytree of row-major leaves.
    idx: int row indices.

  Returns:
    the pytree of those rows.
  """
  return tree_map(lambda x: _ops(x).take(x, idx), tree)


def put(full: Any, idx: np.ndarray, part: Any) -> Any:
  """A copy of `full` with rows `idx` replaced by `part` (`full` unchanged).

  Args:
    full: a pytree of row-major leaves.
    idx: int row indices.
    part: the new rows, a pytree like full.

  Returns:
    the copy.
  """
  return tree_map(lambda x, y: _ops(x).put(x, idx, y), full, part)


def alloc_rows(part: Any, n: int) -> Any:
  """Zeros with leading dim n, else each leaf's shape, dtype and placement.

  Args:
    part: a pytree of row-major leaves.
    n: the row count.

  Returns:
    the zeros pytree.
  """
  return tree_map(lambda x: _ops(x).zeros_like_rows(x, n), part)
