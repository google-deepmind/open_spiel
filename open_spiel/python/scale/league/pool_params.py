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

"""How the League stores the K member positions of a population pool.

The population pool has a fixed number K of positions so the Arena never
recompiles as members come and go. The loop reads and writes positions only
through the `PoolStorage` chosen for the pool's member policy (`for_policy`):

  JaxStacked  an in-graph JAX policy: one pytree of jax arrays stacked
              [K, ...], written with .at[m].set.
  HostList    a HOST_DRIVEN policy: a list of K host objects (the HostArena's
              params form).

A plugin adds storage for other policies with `register(predicate, factory)`;
registered predicates are tried before the built-ins.

    storage = pool_params.for_policy(member_policy)
    stacked = storage.set(storage.empty(template, k), m, params)

Every write returns a new container and never mutates the old one (an
ArenaSource may hold it until its next set_params). `check(params, template)`
refuses params whose tree, shapes or dtypes differ from the architecture's.
`to_host` is what a benched member keeps in host memory. This module imports
no array framework at import time.
"""

from typing import Any, Callable, Protocol

import numpy as np

from open_spiel.python.scale.core import slots
from open_spiel.python.scale.core import types


def _tree():
  """jax.tree_util, imported on first use.

  Returns:
    jax.tree_util.
  """
  # Lazy: pool_params imports without jax (jax_free_control_test).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return jax.tree_util


class PoolStorage(Protocol):
  """The storage of one pool's K positions (module doc)."""

  def empty(self, template: Any, n: int) -> Any:
    """K empty positions shaped like `template`.

    Args:
      template: a member's params.
      n: the number of positions.

    Returns:
      The stacked params.
    """
    ...

  def set(self, stacked: Any, m: int, params: Any) -> Any:
    """Stacked params with position m set to `params`.

    Args:
      stacked: the stacked params.
      m: the position.
      params: the member's params.

    Returns:
      The updated stacked params.
    """
    ...

  def get(self, stacked: Any, m: int) -> Any:
    """The params at position m.

    Args:
      stacked: the stacked params.
      m: the position.

    Returns:
      The member's params.
    """
    ...

  def check(self, params: Any, template: Any) -> None:
    """Raises unless `params` fit the pool's template.

    Args:
      params: a member's params.
      template: the pool's template.
    """
    ...

  def to_host(self, params: Any) -> Any:
    """Params in host memory.

    Args:
      params: a member's params.

    Returns:
      The host params.
    """
    ...


def check_tree(params: Any, template: Any) -> None:
  """Raises ValueError unless params has template's tree, shapes and dtypes.

  Args:
    params: a member's params.
    template: the pool's template.

  Raises:
    ValueError: the trees, shapes or dtypes differ.
  """
  tu = _tree()
  sa = [(np.shape(x), np.asarray(x).dtype) for x in tu.tree_leaves(params)]
  sb = [(np.shape(x), np.asarray(x).dtype) for x in tu.tree_leaves(template)]
  if tu.tree_structure(params) != tu.tree_structure(template) or sa != sb:
    raise ValueError(
        f"params do not match the league architecture: {sa} vs {sb}"
    )


class JaxStacked:
  """[K, ...] jax arrays (module doc)."""

  def empty(self, template: Any, n: int) -> Any:
    """K zero positions shaped like `template`.

    Args:
      template: a member's params.
      n: the number of positions.

    Returns:
      The stacked params.
    """
    # Lazy: pool_params imports without jax (jax_free_control_test).
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return _tree().tree_map(
        lambda x: jnp.zeros((n,) + jnp.shape(x), jnp.asarray(x).dtype), template
    )

  def set(self, stacked: Any, m: int, params: Any) -> Any:
    """Stacked params with position m set to `params`.

    Args:
      stacked: the stacked params.
      m: the position.
      params: the member's params.

    Returns:
      The updated stacked params.
    """
    # Lazy: pool_params imports without jax (jax_free_control_test).
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return _tree().tree_map(
        lambda big, x: big.at[m].set(jnp.asarray(x, big.dtype)), stacked, params
    )

  def get(self, stacked: Any, m: int) -> Any:
    """The params at position m.

    Args:
      stacked: the stacked params.
      m: the position.

    Returns:
      The member's params.
    """
    return _tree().tree_map(lambda x: x[m], stacked)

  def check(self, params: Any, template: Any) -> None:
    """Raises unless `params` have the template's tree, shapes and dtypes.

    Args:
      params: a member's params.
      template: the pool's template.
    """
    check_tree(params, template)

  def to_host(self, params: Any) -> Any:
    """Jax arrays as host numpy; typed PRNG keys and other leaves as they are.

    Args:
      params: a params tree.

    Returns:
      The host tree.
    """
    # Lazy: pool_params imports without jax (jax_free_control_test).
    import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    def one(x):
      """One leaf in host memory.

      Args:
        x: a leaf.

      Returns:
        The leaf as numpy if it is a non-key jax array, else x.
      """
      if isinstance(x, jax.Array) and not jnp.issubdtype(
          x.dtype, jax.dtypes.prng_key
      ):
        return np.asarray(x)
      return x

    return _tree().tree_map(one, params)


class HostList:
  """A list of K host objects (module doc)."""

  def empty(self, template: Any, n: int) -> Any:
    """K positions holding the template.

    Empty positions are never drawn: no table names them.

    Args:
      template: a member's params.
      n: the number of positions.

    Returns:
      The list of positions.
    """
    # empty positions are never drawn: no table names them
    return [template] * int(n)

  def set(self, stacked: Any, m: int, params: Any) -> Any:
    """A copy of the list with position m set to `params`.

    Args:
      stacked: the list of positions.
      m: the position.
      params: the member's params.

    Returns:
      The updated list.
    """
    out = list(stacked)
    out[m] = params
    return out

  def get(self, stacked: Any, m: int) -> Any:
    """The params at position m.

    Args:
      stacked: the list of positions.
      m: the position.

    Returns:
      The member's params.
    """
    return stacked[m]

  def check(self, params: Any, template: Any) -> None:
    """Raises unless `params` have the template's type.

    Args:
      params: a member's params.
      template: the pool's template.

    Raises:
      ValueError: the types differ.
    """
    if type(params) is not type(template):
      raise ValueError(
          "params do not match the league architecture: a "
          f"{type(params).__name__}, the template is a "
          f"{type(template).__name__}"
      )

  def to_host(self, params: Any) -> Any:
    """The params as they are (they live on the host).

    Args:
      params: a member's params.

    Returns:
      params.
    """
    return params


_REGISTERED: list[Any] = []  # (predicate, factory)


def register(
    predicate: Callable[[types.PolicySpec], bool],
    factory: Callable[[], PoolStorage],
) -> None:
  """Stores the pools of policies whose spec satisfies predicate in factory().

  Args:
    predicate: whether a PolicySpec's pools use this storage.
    factory: builds the storage.
  """
  entry = (predicate, factory)
  if entry not in _REGISTERED:
    _REGISTERED.append(entry)


def for_policy(policy: types.Policy) -> PoolStorage:
  """The storage of a member policy's pool (module doc).

  Args:
    policy: the member policy.

  Returns:
    The PoolStorage.

  Raises:
    ValueError: no storage takes the policy.
  """
  spec = policy.spec
  for predicate, factory in _REGISTERED:
    if predicate(spec):
      return factory()
  if spec.capability not in slots.IN_GRAPH:
    return HostList()
  if spec.framework == "jax":
    return JaxStacked()
  raise ValueError(
      f"no pool storage for {spec.arch_name} (framework {spec.framework}); "
      "register one with league.pool_params.register"
  )
