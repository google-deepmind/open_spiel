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

"""Liar's dice Env over the JAX engine `liars_dice`.

With a LiarsDiceInfo InfoState.

The JAX engine plays OpenSpiel `liars_dice(numdice=2,dice_sides=5)`
(reset-face bidding): 2 seats, 2 dice each, faces 1..5, the highest face (5)
wild. Chance outcome o rolls face o + 1;
seat 0 rolls both its dice, then seat 1 (four chance nodes at the root, no
mid-hand chance). Actions: bid id (quantity - 1) * 5 + (face - 1) for ids
0..19 (each bid must exceed the last), LIAR = 20 (legal once a bid exists)
ends the hand: the bidder wins (+1) if at least `quantity` dice show `face`
or 5, else the caller wins.

The engine's state is a dict (no `is_terminal`; `done`), every leaf int8, and
current_player() is -4 at terminal; the wrapper reads `done` and the adapter
normalizes actor() to -1. Dice are faces 1..5 in the InfoState (the engine
sorts each seat's dice once all are rolled, as OpenSpiel's strings do).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import liars_dice
from open_spiel.python.scale.envs import jax_game_env

NUM_DICE = liars_dice.BENCHMARK_NUMDICE  # per seat
DICE_SIDES = liars_dice.BENCHMARK_DICE_SIDES
LIAR = liars_dice.liar_action(NUM_DICE, DICE_SIDES)  # 20
NUM_ACTIONS = liars_dice.num_actions(NUM_DICE, DICE_SIDES)  # 21
MAX_LEN = NUM_ACTIONS  # 20 strictly increasing bids, then LIAR


class LiarsDiceInfo(NamedTuple):
  """What one seat knows: its seat, its own dice, the public bids in order.

  Attributes:
    seat: i32.
    dice: i32[2] own faces 1..5, sorted.
    bids: i32[21] actions in order (bids, then LIAR), -1 pad.
    num_bids: i32.
  """

  seat: jax.Array
  dice: jax.Array
  bids: jax.Array
  num_bids: jax.Array


class LiarsDiceEnv(jax_game_env.JaxGameEnv):
  """Liar's dice with 2 dice of 5 faces per seat.

  2 seats, 21 actions, 5 chance outcomes.
  """

  name = "liars_dice"
  max_len = MAX_LEN
  zero_sum = True

  def __init__(self):
    super().__init__(
        liars_dice.LiarsDice(numdice=NUM_DICE, dice_sides=DICE_SIDES)
    )
    assert self.num_players == 2 and self.num_actions == NUM_ACTIONS
    assert self.max_chance_outcomes == DICE_SIDES

  def terminal(self, s) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s["done"], jnp.bool_)

  def info_state(self, s, seat) -> LiarsDiceInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return LiarsDiceInfo(
        seat=seat,
        dice=jnp.asarray(s["dice"], jnp.int32)[seat],
        bids=jnp.asarray(s["bid_history"], jnp.int32),
        num_bids=jnp.asarray(s["num_bids"], jnp.int32),
    )

  def labels(self, s, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {"opp_dice": jnp.asarray(s["dice"], jnp.int32)[1 - seat]}


def hide_opponent_dice(env: LiarsDiceEnv, s, seat, key):
  """Privacy hook for check_env: reroll `seat`'s opponent's dice uniformly.

  (sorted, as the engine keeps rolled dice); `seat`'s dice and the bids stay.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The state with the opponent's dice rerolled.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  faces = jax.random.randint(
      key, (NUM_DICE,), 1, DICE_SIDES + 1, dtype=jnp.int32
  )
  faces = jnp.sort(faces).astype(s["dice"].dtype)
  out = dict(s)
  out["dice"] = s["dice"].at[1 - seat].set(faces)
  return out
