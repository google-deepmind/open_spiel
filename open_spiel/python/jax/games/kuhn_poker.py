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

"""Kuhn poker, OpenSpiel's `kuhn_poker` (two players).

A three-card poker game (Kuhn 1950) used as the smallest benchmark for
imperfect-information algorithms.

State: a `GameState` with each player's card (-1 until dealt), the betting
history and the returns. Two chance nodes deal player 0 then player 1 a card
from {J=0, Q=1, K=2}; the chance action id is the card. Each player antes 1.

Actions: pass (0) and bet (1). Betting sequences and outcomes:
  pp   showdown for the antes (the higher card wins 1);
  bp   player 1 folds, player 0 wins 1;
  bb   showdown for 2;
  pbp  player 0 folds, player 1 wins 1;
  pbb  showdown for 2.

Observation (f32[7]): (player 0 flag, player 1 flag, own card one-hot J, Q,
K, chips in the pot from player 0, from player 1), OpenSpiel's layout.
Information state (f32[11]): the same flags and card, then a (pass, bet)
one-hot per betting slot, zero for slots not yet played.
"""

import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel


_NUM_CARDS = 3
_NUM_BET_ACTIONS = 2
_MAX_BETS = 3
_OBS_DIM = 7
_INFO_DIM = 11


class GameState(NamedTuple):
  """State of a Kuhn Poker game.

  Attributes:
    bets: Shape (3,), int32, -1 for unused slots (0=pass, 1=bet).
    current_player: int32: -1 (chance) or 0/1.
    is_terminal: bool.
    num_bets: int32, number of betting actions taken.
    p0_card: int32, -1 if not dealt.
    p1_card: int32, -1 if not dealt.
    returns: Shape (2,), float32.
  """

  p0_card: jax.Array
  p1_card: jax.Array
  bets: jax.Array
  num_bets: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  returns: jax.Array


def _compute_pot(
    bets: jax.Array, num_bets: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """The pot contributions given the betting history.

  Each player antes 1 chip. Betting actions alternate P0, P1, P0.
  A `bet` (action 1) adds 1 to that player's contribution.

  Args:
    bets: The betting actions.
    num_bets: How many entries of `bets` were played.

  Returns:
    (pot_p0, pot_p1).
  """
  idx = jnp.arange(_MAX_BETS, dtype=jnp.int32)
  active = idx < num_bets
  is_bet = bets == 1
  actor_is_p0 = (idx % 2) == 0  # betting alternates starting with P0
  bet_counts_p0 = jnp.sum((is_bet & actor_is_p0 & active).astype(jnp.int32))
  bet_counts_p1 = jnp.sum((is_bet & ~actor_is_p0 & active).astype(jnp.int32))
  return (
      jnp.int32(1) + bet_counts_p0,
      jnp.int32(1) + bet_counts_p1,
  )


def _compute_terminal(num_bets: jax.Array, bets: jax.Array) -> jax.Array:
  """Returns True if the betting history is terminal.

  Args:
    num_bets: The number of bets.
    bets: The betting actions.

  Returns:
    bool[], whether the history is terminal.
  """
  is_pb = (bets[0] == 0) & (bets[1] == 1)
  terminal_at_2 = (num_bets == 2) & ~is_pb
  terminal_at_3 = num_bets >= 3
  return terminal_at_2 | terminal_at_3


def _compute_current_player(
    p0_card: jax.Array,
    p1_card: jax.Array,
    num_bets: jax.Array,
    bets: jax.Array,
    is_terminal: jax.Array,
) -> jax.Array:
  """Compute whose move it is: -1 chance, 0 or 1 otherwise.

  Args:
    p0_card: Player 0's card, or -1 before the deal.
    p1_card: Player 1's card, or -1 before the deal.
    num_bets: The number of betting actions so far.
    bets: The betting actions.
    is_terminal: Whether the state is terminal.

  Returns:
    i32[], the player to move (chance -1, terminal id when over).
  """
  is_dealing_p0 = p0_card == -1
  is_dealing_p1 = (p0_card >= 0) & (p1_card == -1)
  is_chance = is_dealing_p0 | is_dealing_p1

  # After cards are dealt: P0 always acts first, then P1, then P0 again
  # (only in the pb extension).
  is_pb = (bets[0] == 0) & (bets[1] == 1)
  is_p0_turn = (num_bets == 0) | ((num_bets == 2) & is_pb)
  betting_player = jnp.where(is_p0_turn, jnp.int32(0), jnp.int32(1))

  return jnp.where(
      is_terminal,
      jnp.int32(game_lib.TERMINAL_PLAYER_ID),
      jnp.where(
          is_chance, jnp.int32(game_lib.CHANCE_PLAYER_ID), betting_player
      ),
  )


def _compute_returns(
    p0_card: jax.Array,
    p1_card: jax.Array,
    num_bets: jax.Array,
    bets: jax.Array,
) -> jax.Array:
  """Compute returns at terminal state.

  Args:
    p0_card: Player 0's card.
    p1_card: Player 1's card.
    num_bets: The number of betting actions.
    bets: The betting actions.

  Returns:
    f32[2], the players' returns.
  """
  p0_wins_showdown = p0_card > p1_card

  is_pp = (num_bets == 2) & (bets[0] == 0) & (bets[1] == 0)
  is_bb = (num_bets == 2) & (bets[0] == 1) & (bets[1] == 1)
  is_bp = (num_bets == 2) & (bets[0] == 1) & (bets[1] == 0)
  is_pbp = (num_bets == 3) & (bets[0] == 0) & (bets[1] == 1) & (bets[2] == 0)
  is_pbb = (num_bets == 3) & (bets[0] == 0) & (bets[1] == 1) & (bets[2] == 1)

  # Showdowns: pp (±1), bb / pbb (±2) for the higher card.
  is_showdown = is_pp | is_bb | is_pbb
  showdown_p0 = jnp.where(p0_wins_showdown, 1.0, -1.0)
  bet_multiplier = jnp.where(is_pp, 1.0, 2.0)
  showdown_p0 = showdown_p0 * bet_multiplier

  # Folds: bp → P0 wins 1 (P1 folded); pbp → P1 wins 1 (P0 folded).
  fold_p0 = jnp.where(is_bp, 1.0, jnp.where(is_pbp, -1.0, 0.0))

  p0_return = jnp.where(is_showdown, showdown_p0, fold_p0)
  return jnp.stack([p0_return, -p0_return]).astype(jnp.float32)


class KuhnPoker(game_lib.Game, interop.PyspielInterop):
  """Two-player Kuhn poker."""

  def __init__(self, players: int = 2):
    """Builds the game.

    Args:
      players: OpenSpiel's parameter; only its default, 2, is implemented.

    Raises:
      ValueError: `players` is not 2.
    """
    if players != 2:
      raise ValueError(f"kuhn_poker supports only players=2, got {players!r}")

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return _NUM_BET_ACTIONS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (_OBS_DIM,)

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    return (_INFO_DIM,)

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return _NUM_CARDS

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return GameState(
        p0_card=jnp.int32(-1),
        p1_card=jnp.int32(-1),
        bets=jnp.full((_MAX_BETS,), -1, dtype=jnp.int32),
        num_bets=jnp.int32(0),
        current_player=jnp.int32(game_lib.CHANCE_PLAYER_ID),
        is_terminal=jnp.bool_(False),
        returns=jnp.zeros((2,), dtype=jnp.float32),
    )

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    action_i = jnp.asarray(action).astype(jnp.int32)

    is_dealing_p0 = state.p0_card == -1
    is_dealing_p1 = (state.p0_card >= 0) & (state.p1_card == -1)
    is_chance = is_dealing_p0 | is_dealing_p1
    is_betting = (~is_chance) & (~state.is_terminal)

    new_p0_card = jnp.where(is_dealing_p0, action_i, state.p0_card)
    new_p1_card = jnp.where(is_dealing_p1, action_i, state.p1_card)

    new_bets = jnp.where(
        is_betting,
        state.bets.at[state.num_bets].set(action_i),
        state.bets,
    )
    new_num_bets = jnp.where(is_betting, state.num_bets + 1, state.num_bets)

    new_is_terminal = jnp.where(
        is_chance,
        state.is_terminal,
        _compute_terminal(new_num_bets, new_bets) | state.is_terminal,
    )

    new_returns = jnp.where(
        new_is_terminal & ~state.is_terminal,
        _compute_returns(new_p0_card, new_p1_card, new_num_bets, new_bets),
        state.returns,
    )

    new_current_player = _compute_current_player(
        new_p0_card, new_p1_card, new_num_bets, new_bets, new_is_terminal
    )

    return GameState(
        p0_card=new_p0_card,
        p1_card=new_p1_card,
        bets=new_bets,
        num_bets=new_num_bets,
        current_player=new_current_player,
        is_terminal=new_is_terminal,
        returns=new_returns,
    )

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return state.current_player == game_lib.CHANCE_PLAYER_ID

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """Returns the chance distribution, indexed by card.

    1/3 each for player 0's card, then 1/2 for each card player 0 does not hold.

    Args:
      state: The state.

    Returns:
      f32[3].
    """
    is_dealing_p0 = state.p0_card == -1
    cards = jnp.arange(3, dtype=jnp.int32)
    p0_probs = jnp.full((3,), 1.0 / 3.0, dtype=jnp.float32)
    p1_probs = jnp.where(
        cards == state.p0_card, jnp.float32(0.0), jnp.float32(0.5)
    )
    return jnp.where(is_dealing_p0, p0_probs, p1_probs)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal actions over (pass, bet).

    Args:
      state: The state.

    Returns:
      bool[2]; all False at chance and terminal states.
    """
    not_playing = state.is_terminal | (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    )
    return jnp.where(
        not_playing,
        jnp.zeros((_NUM_BET_ACTIONS,), dtype=jnp.bool_),
        jnp.ones((_NUM_BET_ACTIONS,), dtype=jnp.bool_),
    )

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the observation; see the module docstring.

    The card one-hot is zero before the deal.

    Args:
      state: The state.
      observing_player: The observing player.

    Returns:
      f32[7].
    """
    observing_player_i = jnp.asarray(observing_player).astype(jnp.int32)

    player_onehot = jax.nn.one_hot(observing_player_i, 2, dtype=jnp.float32)

    my_card = jnp.where(observing_player_i == 0, state.p0_card, state.p1_card)
    card_valid = my_card >= 0
    card_onehot = jnp.where(
        card_valid,
        jax.nn.one_hot(jnp.maximum(my_card, 0), _NUM_CARDS, dtype=jnp.float32),
        jnp.zeros((_NUM_CARDS,), dtype=jnp.float32),
    )

    pot_p0, pot_p1 = _compute_pot(state.bets, state.num_bets)
    pot = jnp.stack([pot_p0, pot_p1]).astype(jnp.float32)

    return jnp.concatenate([player_onehot, card_onehot, pot])

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the information state; see the module docstring.

    Args:
      state: The state.
      observing_player: The observing player.

    Returns:
      f32[11].
    """
    observing_player_i = jnp.asarray(observing_player).astype(jnp.int32)

    player_onehot = jax.nn.one_hot(observing_player_i, 2, dtype=jnp.float32)

    my_card = jnp.where(observing_player_i == 0, state.p0_card, state.p1_card)
    card_valid = my_card >= 0
    card_onehot = jnp.where(
        card_valid,
        jax.nn.one_hot(jnp.maximum(my_card, 0), _NUM_CARDS, dtype=jnp.float32),
        jnp.zeros((_NUM_CARDS,), dtype=jnp.float32),
    )

    idx = jnp.arange(_MAX_BETS, dtype=jnp.int32)
    active = idx < state.num_bets
    # Per-slot one-hot over {pass=0, bet=1}, zeroed for inactive slots.
    slot_onehot = jnp.where(
        active[:, None],
        jax.nn.one_hot(
            jnp.maximum(state.bets, 0), _NUM_BET_ACTIONS, dtype=jnp.float32
        ),
        jnp.zeros((_MAX_BETS, _NUM_BET_ACTIONS), dtype=jnp.float32),
    )
    history = slot_onehot.reshape(-1)

    return jnp.concatenate([player_onehot, card_onehot, history])

  def returns(self, state: GameState) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    return state.returns

  def observation_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    return jnp.asarray(
        np.asarray(os_state.observation_tensor(player), dtype=np.float32)
    )

  def information_state_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's information-state tensor of `player`.

    Games whose information state is their observation inherit this
    default, which returns `observation_from_os`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The information-state tensor the JAX game gives `player` in the same
      position.
    """
    return jnp.asarray(
        np.asarray(os_state.information_state_tensor(player), dtype=np.float32)
    )
