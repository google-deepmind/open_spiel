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

"""The uniform random policy: zero logits, uniform over the legal actions.

  policy = UniformPolicy("kuhn_poker", num_actions=2, num_players=2)

The baseline opponent of many recipes and the simplest Policy to test
against.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry

ARCH = "uniform"


def _batch_of(tree) -> int:
  """The batch size of an InfoState.

  Args:
    tree: The InfoState, leaves [B, ...].

  Returns:
    The leading size.

  Raises:
    ValueError: The InfoState has no leaves.
  """
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    raise ValueError("empty InfoState")
  return leaves[0].shape[0]


class UniformPolicy(types.Policy):
  """Zero logits and zero values. Stateless; params {}; carry ().

  Attributes:
    spec: the PolicySpec.
  """

  decider = "scripted"

  def __init__(self, game: str, num_actions: int, num_players: int):
    self.spec = types.PolicySpec(
        game=game,
        obs_format="none",
        num_actions=num_actions,
        num_players=num_players,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            game=game, num_actions=num_actions, num_players=num_players
        ),
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """No features.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      f32[B, 0].
    """
    return jnp.zeros((_batch_of(info), 0), jnp.float32)

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

  def evaluate(self, params, carry, step, key):
    """Zero logits and zero values.

    Args:
      params: unused.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    del params, carry, key
    b = step.legal_mask.shape[0]
    return types.PolicyOutput(
        logits=jnp.zeros((b, self.spec.num_actions), jnp.float32),
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


registry.register(ARCH, UniformPolicy)
