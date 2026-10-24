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

"""The belief of a perfect-information game: the observed state itself.

In a perfect-information game a seat's InfoState determines the full game
state, so the "belief" is a point mass and search needs no sampling. This
sampler returns `Env.world_from_info(info)` K times, which lets the same
SearchPolicy code run on perfect- and imperfect-information games.

  sampler = registry.build("perfect_information", env)
  # worlds: leaves [B, 1, ...]
  worlds, logw = sampler.sample_worlds({}, (), info, k=1, key=key)

Search over these worlds has no strategy-fusion bias
(`is_perfect_information` is True).
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.core import types


class PerfectInformationSampler:
  """K copies of `Env.world_from_info(info)`. Stateless; params {}.

  Attributes:
    env: the perfect-information Env.
    kwargs: the sampler's constructor kwargs, as recorded (none).
  """

  capability = types.Capability.NATIVE_JAX
  stateful = False
  is_perfect_information = True
  params_source = "none"

  def __init__(self, env: types.Env):
    """Builds the sampler.

    Args:
      env: a perfect-information Env.

    Raises:
      ValueError: the env's `world_from_info` returns None (the env has hidden
        information).
    """
    probe = env.info_state(env.init(jax.random.key(0)), jnp.int32(0))
    if env.world_from_info(probe) is None:
      raise ValueError(
          f"{env.name}: world_from_info returned None, so the game has hidden "
          "information; use an imperfect-information sampler (enumeration, "
          "particle or a game-specific one)"
      )
    self.env = env
    self.kwargs: dict[str, Any] = {}

  def init_params(self, key: Any = None) -> dict[str, Any]:
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  def sample_worlds(self, params, carry, info, k: int, key):
    """Each row's own state, K times.

    Args:
      params: unused.
      carry: unused.
      info: the InfoStates (the full states), leaves [B, ...].
      k: worlds per row.
      key: unused.

    Returns:
      (env states with leaves [B, K, ...], log-weights f32[B, K] of 0).

    Raises:
      ValueError: `k` < 1.
    """
    del params, carry, key
    if k < 1:
      raise ValueError(f"k must be >= 1, got {k}")
    worlds = jax.vmap(self.env.world_from_info)(info)
    b = jax.tree_util.tree_leaves(worlds)[0].shape[0]
    worlds = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[:, None], (x.shape[0], k) + x.shape[1:]),
        worlds,
    )
    return worlds, jnp.zeros((b, k), jnp.float32)


registry.register("perfect_information", PerfectInformationSampler)
