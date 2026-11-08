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

"""Heads-up no-limit hold'em Env over the JAX engine `no_limit_holdem`.

The game is exactly OpenSpiel's

  universal_poker(betting=nolimit,bettingAbstraction=fullgame,blind=2 1,
      firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,
      numRanks=13,numRounds=4,numSuits=4,stack=200 200)

The Env is named `universal_poker`, the pyspiel short name a served request
carries (registry: Env.name = the pyspiel short name); GameSpec pins every
pyspiel parameter to that configuration, and a served request must state the
rule parameters (pyspiel's universal_poker DEFAULTS are another game).

Conventions (the engine = pyspiel, nothing renumbered):
  * seats: 0 is the big blind (2), 1 the small blind (1); seat 1 acts first
    preflop, seat 0 first on the flop, turn and river;
  * actions: 0 fold, 1 call/check, k in 2..200 raise to a total commitment of k
    chips (STACK = 200 is all-in);
  * chance: card ids 0..51, card = 4 * rank + suit (rank 0 = deuce, 12 = ace),
    dealt seat 0's two hole cards, seat 1's two, then the board (3, 1, 1),
    each from the remaining deck uniformly; an all-in runs the board out.

Normalized in the wrapper: the engine reports current_player -4 at terminal
(the adapter reports actor -1), pads cards with INVALID_CARD (-10000; the
InfoState pads with -1), and sets round to 4 after the river closes (the
InfoState clips it to 3). The InfoState keeps no deal events: the board and
the betting history carry everything public. The opponent's hole cards stay
absent at a showdown terminal (OpenSpiel's information state never shows
them either; returns() carries the outcome).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_base
from open_spiel.python.jax.games import no_limit_holdem as nlh
from open_spiel.python.scale.envs import jax_game_env

FOLD = nlh.FOLD
CALL = nlh.CALL
MIN_RAISE = nlh.MIN_RAISE_ACTION
STACK = nlh.STACK
NUM_ACTIONS = nlh.NUM_ACTIONS  # 201
NUM_CARDS = nlh.NUM_CARDS  # 52
NUM_HOLE = nlh.NUM_HOLE_CARDS  # 2
NUM_BOARD = nlh.MAX_BOARD_CARDS  # 5
NUM_ROUNDS = 4
GAME_STRING = nlh.UNIVERSAL_POKER_GAME_STRING
# Player decisions in one hand, at most: every raise lifts the highest
# commitment by >= 2 chips (the minimum raise is the big blind; an all-in
# exactly to 200 is the last), from 2 to 200, so <= 99 raises in the hand;
# each round has <= 2 other actions (the opening limp / check, the closing
# call / check / fold). 99 + 4 * 2 = 107, reached by
# limp, check; then on each of flop, turn and river: check, min-raises,
# call, with the 99th raise all-in (universal_poker_test pins the path).
MAX_LEN = 99 + NUM_ROUNDS * 2

_DEAL = ord("d")
_FOLD_CH, _CALL_CH, _RAISE_CH = ord("f"), ord("c"), ord("p")
_PRIVATE_DEALS = 2 * NUM_HOLE  # 4


class HoldemInfo(NamedTuple):
  """What one seat knows: its seat and hole cards, and everything public.

  Attributes:
    seat: i32.
    hole: i32[2] own hole cards 0..51 in deal order.
    board: i32[5] board cards in deal order, -1 not yet dealt.
    round: i32 0 preflop, 1 flop, 2 turn, 3 river.
    spent: i32[2] each seat's total commitment (blinds included).
    actions: i32[MAX_LEN] player actions in order (0, 1, raise-to k), -1 pad.
    action_rounds: i32[MAX_LEN] the round of each action, -1 pad.
    num_actions: i32.
  """

  seat: jax.Array
  hole: jax.Array
  board: jax.Array
  round: jax.Array
  spent: jax.Array
  actions: jax.Array
  action_rounds: jax.Array
  num_actions: jax.Array


def _cards(x) -> jax.Array:
  """Cards with INVALID_CARD as -1.

  Args:
    x: The cards.

  Returns:
    i32 cards.
  """
  x = jnp.asarray(x, jnp.int32)
  return jnp.where(x == game_base.INVALID_CARD, jnp.int32(-1), x)


def player_actions(s: nlh.GameState) -> tuple[jax.Array, jax.Array, jax.Array]:
  """The player actions of the engine's interleaved history.

  (i32[MAX_LEN] actions, i32[MAX_LEN] their rounds, i32 count): the player
  actions of the engine's interleaved history (deal events dropped), as action
  ids, in order, -1 padded.

  Args:
    s: The engine state.

  Returns:
    (i32[MAX_LEN] actions, i32[MAX_LEN] their rounds, i32 count), deal events
    dropped, -1 padded.
  """
  n = s.action_chars.shape[0]
  idx = jnp.arange(n, dtype=jnp.int32)
  chars = jnp.asarray(s.action_chars, jnp.int32)
  active = idx < jnp.asarray(s.history_len, jnp.int32)
  deal = active & (chars == _DEAL)
  mine = active & (chars != _DEAL)
  deals_before = jnp.cumsum(deal, dtype=jnp.int32)  # deals at or before i
  rnd = jnp.where(
      deals_before <= _PRIVATE_DEALS, 0, deals_before - _PRIVATE_DEALS - 2
  )
  # 7 -> 1, 8 -> 2, 9 -> 3
  rnd = jnp.clip(rnd, 0, NUM_ROUNDS - 1).astype(jnp.int32)
  act = jnp.where(
      chars == _RAISE_CH,
      jnp.asarray(s.action_sizes, jnp.int32),
      jnp.where(chars == _CALL_CH, jnp.int32(CALL), jnp.int32(FOLD)),
  )
  pos = jnp.where(
      mine, jnp.cumsum(mine, dtype=jnp.int32) - 1, jnp.int32(MAX_LEN)
  )
  out_a = jnp.full((MAX_LEN,), -1, jnp.int32).at[pos].set(act, mode="drop")
  out_r = jnp.full((MAX_LEN,), -1, jnp.int32).at[pos].set(rnd, mode="drop")
  return out_a, out_r, jnp.sum(mine, dtype=jnp.int32)


class NoLimitHoldemEnv(jax_game_env.JaxGameEnv):
  """Heads-up no-limit hold'em, 200-chip stacks, blinds 1/2.

  2 seats, 201 actions.
  """

  name = "universal_poker"
  max_len = MAX_LEN
  zero_sum = True

  def __init__(self):
    # The LUT showdown evaluator: same scores as the categorical one
    # (the engine tests every five-card hand), faster on accelerators.
    super().__init__(
        nlh.NoLimitHoldem(
            use_showdown_cache=True, track_information_state_history=True
        )
    )
    assert self.max_chance_outcomes == NUM_CARDS, self.max_chance_outcomes
    assert self.num_actions == NUM_ACTIONS and self.num_players == 2
    assert MAX_LEN + nlh.MAX_DEALT_CARDS <= nlh.MAX_GAME_LENGTH

  def info_state(self, s: nlh.GameState, seat) -> HoldemInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    acts, rounds, n = player_actions(s)
    return HoldemInfo(
        seat=seat,
        hole=_cards(s.private_cards)[seat],
        board=_cards(s.board_cards),
        round=jnp.minimum(
            jnp.asarray(s.round, jnp.int32), jnp.int32(NUM_ROUNDS - 1)
        ),
        spent=jnp.asarray(s.spent, jnp.int32),
        actions=acts,
        action_rounds=rounds,
        num_actions=n,
    )

  def labels(self, s: nlh.GameState, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {"opp_hole": _cards(s.private_cards)[1 - seat]}


def hide_opponent_hole(env: NoLimitHoldemEnv, s: nlh.GameState, seat, key):
  """Privacy hook for check_env: redraw `seat`'s opponent's hole cards.

  Privacy hook for check_env: redraw `seat`'s opponent's two hole cards
  uniformly (without replacement) from the cards `seat` cannot rule out (not
  its own, not on the board), keeping deck_mask consistent with the dealt
  cards. Only states with every hole card dealt reach it (init resolves the
  private deal).

  Args:
    env: The env.
    s: The engine state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The state with the opponent's hole cards redrawn.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  cards = jnp.asarray(s.private_cards, jnp.int32)
  board = jnp.asarray(s.board_cards, jnp.int32)
  ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  seen = (ids[:, None] == cards[seat][None, :]).any(-1) | (
      ids[:, None] == board[None, :]
  ).any(-1)
  r = jax.random.uniform(key, (NUM_CARDS,), jnp.float32)
  # unseen first, random order
  order = jnp.argsort(jnp.where(seen, jnp.float32(2.0), r))
  opp = order[:NUM_HOLE].astype(jnp.int32)
  new_cards = cards.at[1 - seat].set(opp)
  dealt = (ids[:, None] == new_cards.reshape(-1)[None, :]).any(-1) | (
      ids[:, None] == board[None, :]
  ).any(-1)
  return s._replace(private_cards=new_cards, deck_mask=~dealt)
