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

"""A prefer-centre Connect Four policy.

The teacher of the behaviour-cloning example.

It plays the first open column in the order 3, 2, 4, 1, 5, 0, 6 (centre
outwards, ties to the left). Deterministic and cheap, it gives behaviour
cloning a known target and evaluation a weak fixed opponent.

  pol = policies.registry.build("connect_four.prefer_center", {})
  out = pol.evaluate({}, (), step, key)      # one-hot logits on the chosen
  column

A column is open when its top cell (board row 0) is empty. Rows outside the
policy's domain (no open column, or not the row's seat to move) get NaN
logits (fixed_action.one_hot_output).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.policies import fixed_action
from open_spiel.python.scale.policies import registry

ARCH = "connect_four.prefer_center"
PRIORITY = (3, 2, 4, 1, 5, 0, 6)
NUM_ACTIONS = connect_four.COLS


def legal_columns(info) -> jax.Array:
  """bool[B, 7]: the column's top cell is empty.

  Args:
    info: A batched InfoState.

  Returns:
    bool[B, 7] open columns.
  """
  return info.board[..., :NUM_ACTIONS] == -1


def actions(info) -> jax.Array:
  """i32[B]: the first open column in PRIORITY (column 3 if none is open).

  Args:
    info: A batched InfoState.

  Returns:
    i32[B] actions.
  """
  prio = jnp.asarray(PRIORITY, jnp.int32)
  legal_prio = legal_columns(info)[
      ..., prio
  ]  # [B, 7] legality in priority order
  first = jnp.argmax(legal_prio.astype(jnp.int32), axis=-1)
  return prio[first].astype(jnp.int32)


def in_domain(info) -> jax.Array:
  """bool[B]: some column is open and it is the row's seat to move.

  Args:
    info: A batched InfoState.

  Returns:
    bool[B], per row.
  """
  board = info.board
  n0 = jnp.sum(board == 0, axis=-1, dtype=jnp.int32)
  n1 = jnp.sum(board == 1, axis=-1, dtype=jnp.int32)
  to_move = jnp.where(n0 > n1, jnp.int32(1), jnp.int32(0))
  return jnp.any(legal_columns(info), axis=-1) & (
      to_move == info.seat.astype(jnp.int32)
  )


class PreferCenterPolicy(types.Policy):
  """Stateless; params {}; carry ().

  Attributes:
    spec: The policy's PolicySpec.
  """

  ARCH = ARCH
  decider = "scripted"

  def __init__(self):
    self.spec = types.PolicySpec(
        game="connect_four",
        obs_format="connect_four_info",
        num_actions=NUM_ACTIONS,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """The policy's params (it has none).

    Args:
      key: Unused PRNG key.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """Empty features (the policy reads the InfoState directly).

    Args:
      info: The InfoState.

    Returns:
      The empty features.
    """
    return fixed_action.empty_features(info)

  def init_carry(self, params, batch, seat, key):
    """The policy's carry (it keeps none).

    Args:
      params: Unused.
      batch: Unused.
      seat: Unused.
      key: Unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Passes the carry through unchanged.

    Args:
      params: Unused.
      carry: The carry.
      step: Unused.

    Returns:
      The carry.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """One-hot fixed actions on the rows in domain.

    Args:
      params: Unused.
      carry: Unused.
      step: The step; its info is the batched InfoState.
      key: Unused.

    Returns:
      The policy output.
    """
    del params, carry, key
    info = step.info
    return fixed_action.one_hot_output(
        actions(info), in_domain(info), NUM_ACTIONS, 2
    )


registry.register(ARCH, PreferCenterPolicy)
