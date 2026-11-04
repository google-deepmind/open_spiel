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

"""A gin rummy rule bot as a Policy over GinInfo.

The bot is the JAX gin_rummy game's `simple_bot_action_opt`, which plays every
gin decision
by fixed rules: take the upcard when it improves the hand, discard the card
that leaves the least deadwood, knock as soon as it may. It is the reference
opponent of the gin evaluation (envs.gin.reference_eval) and a teacher for
behaviour cloning.

  pol = policies.registry.build("gin_rummy.rule_bot", {})

The bot reads an engine state dict. Its decisions depend only on the
mover's own hand, the upcard, the phase, the first-upcard pass count, the
knock card, the throwback fields, the endgame flags and the stock size, so
`bot_view` rebuilds exactly those keys from the seat's InfoState: the hand in
its own seat's slot, a synthetic deck holding `stock_count` cards, and
knocker_deadwood 0 for a gin knock, -1 otherwise. A key the bot starts
reading is missing from the view and raises at trace time, so the policy
cannot see hidden state.
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.jax.games.gin_rummy.simple_bot import (
    simple_bot_action_opt,
)
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import fixed_action
from open_spiel.python.scale.policies import registry

ARCH = "gin_rummy.rule_bot"


def bot_view(info) -> dict[str, Any]:
  """The engine-state keys simple_bot reads, from ONE GinInfo (the mover's).

  Args:
    info: One unbatched GinInfo.

  Returns:
    The engine-state dict.
  """
  seat = info.seat.astype(jnp.int32)
  hand = info.hand.astype(jnp.int8)
  zeros = jnp.zeros(cm.NUM_CARDS, jnp.int8)
  deck = (
      jnp.arange(cm.NUM_CARDS, dtype=jnp.int32)
      < info.stock_count.astype(jnp.int32)
  ).astype(jnp.int8)
  return {
      "phase": info.phase.astype(jnp.int8),
      "current_player": seat.astype(jnp.int8),
      "player0_hand": jnp.where(seat == 0, hand, zeros),
      "player1_hand": jnp.where(seat == 1, hand, zeros),
      "upcard": info.upcard.astype(jnp.int8),
      "deck": deck,
      "knock_card": info.knock_card.astype(jnp.int8),
      "pass_count": info.pass_count.astype(jnp.int8),
      "allow_upcard_throwback": info.allow_upcard_throwback.astype(jnp.bool_),
      "upcard_taken": info.upcard_taken.astype(jnp.int8),
      "finished_layoffs": info.finished_layoffs.astype(jnp.bool_),
      "knocker_deadwood": jnp.where(
          info.knocker_gin, jnp.int32(0), jnp.int32(-1)
      ),
  }


def actions(info) -> jax.Array:
  """The bot's action for each row of a batched GinInfo (its own seat).

  i32[B]: the bot's action for each row of a batched GinInfo (its own seat).

  Args:
    info: A batched GinInfo.

  Returns:
    i32[B] actions.
  """
  return jax.vmap(lambda i: simple_bot_action_opt(bot_view(i)))(info).astype(
      jnp.int32
  )


def in_domain(info) -> jax.Array:
  """bool[B]: the row's seat is to move at a live player node.

  Args:
    info: A batched GinInfo.

  Returns:
    bool[B], per row.
  """
  return (
      (info.current_player == info.seat)
      & ~info.done
      & (info.phase != cm.PHASE_DEAL)
  )


class GinRuleBotPolicy(types.Policy):
  """Stateless; params {}; carry ().

  Attributes:
    spec: The policy's PolicySpec.
  """

  ARCH = ARCH
  decider = "scripted"

  def __init__(self):
    self.spec = types.PolicySpec(
        game="gin_rummy",
        obs_format="gin_info",
        num_actions=cm.NUM_ACTIONS,
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
        actions(info), in_domain(info), cm.NUM_ACTIONS, 2
    )


registry.register(ARCH, GinRuleBotPolicy)
