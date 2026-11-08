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

"""Leduc poker, OpenSpiel's `leduc_poker` (two players).

A two-round poker game (Southey et al. 2005) with a six-card deck: jacks,
queens and kings in two suits. Card id c has rank c // 2.

State: a `GameState` with the private and public cards (`INVALID_CARD`
until dealt), the per-round betting sequences, antes and chips. Two chance
nodes deal the private cards and one deals the public card after round 1;
the chance action id is the card, uniform over the cards left.

Actions: fold (0), call or check (1), raise (2). Raises are 2 in round 1 and
4 in round 2, at most two per round. A pair with the public card beats any
unpaired hand; otherwise the higher private card wins; equal hands split the
pot.

Observation (f32[16]): observer one-hot, private card one-hot, public card
one-hot, chips each player has put in. Information state (f32[30]): the same
one-hots, then each round's betting sequence as (call, raise) bits per slot.
Both match OpenSpiel's layouts.
"""

import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel

# Default parameters.
_DEFAULT_PLAYERS = 2
_NUM_SUITS = 2
_FIRST_RAISE_AMOUNT = 2
_SECOND_RAISE_AMOUNT = 4
_MAX_RAISES = 2
_ANTE = 1
_STARTING_MONEY = 100

# Actions.
_FOLD = 0
_CALL = 1
_RAISE = 2
_NUM_ACTIONS = 3
# An unused betting-sequence slot. Its value is the poker games' undealt-card
# value (game.INVALID_CARD), so states compare across versions.
NO_ACTION = -10000
assert NO_ACTION == game_lib.INVALID_CARD


class GameState(NamedTuple):
  """State of the Leduc Poker game.

  Attributes:
    ante: (num_players,) int32.
    current_player: Scalar int32: -1 (chance), 0, 1, ...
    deck_mask: (total_cards,) bool.
    folded: (num_players,) bool.
    is_terminal: Scalar bool.
    legal_actions_mask: (3,) bool.
    money: (num_players,) int32.
    num_calls: Scalar int32.
    num_raises: Scalar int32.
    pot: Scalar int32.
    private_cards: (num_players,) int32.
    private_cards_dealt: Scalar int32.
    public_card: Scalar int32.
    remaining_players: Scalar int32.
    round: Scalar int32: 1 or 2.
    round1_seq_len: Scalar int32.
    round1_sequence: (max_bets_per_round,) int32.
    round2_seq_len: Scalar int32.
    round2_sequence: (max_bets_per_round,) int32.
    stakes: Scalar int32.
  """

  private_cards: jax.Array
  public_card: jax.Array
  deck_mask: jax.Array

  current_player: jax.Array
  round: jax.Array
  stakes: jax.Array
  pot: jax.Array
  ante: jax.Array
  money: jax.Array
  folded: jax.Array

  # Betting sequences, unused slots NO_ACTION.
  round1_sequence: jax.Array
  round2_sequence: jax.Array
  round1_seq_len: jax.Array
  round2_seq_len: jax.Array

  num_raises: jax.Array
  num_calls: jax.Array
  private_cards_dealt: jax.Array
  remaining_players: jax.Array

  is_terminal: jax.Array
  legal_actions_mask: jax.Array


class LeducPoker(game_lib.Game, interop.PyspielInterop):
  """Two-player Leduc poker. Supports exactly two players."""

  def __init__(
      self,
      players: int = _DEFAULT_PLAYERS,
      action_mapping: bool = False,
      suit_isomorphism: bool = False,
  ):
    """The parameters are OpenSpiel's; only their defaults are implemented."""
    if players != _DEFAULT_PLAYERS:
      raise ValueError(
          f"leduc_poker supports only players={_DEFAULT_PLAYERS}, got "
          f"{players!r}"
      )
    if action_mapping:
      raise ValueError("leduc_poker supports only action_mapping=false")
    if suit_isomorphism:
      raise ValueError("leduc_poker supports only suit_isomorphism=false")
    self._num_players = players
    self._total_cards = (players + 1) * _NUM_SUITS
    self._max_bets_per_round = 3 * players - 2
    self._max_game_length = 2 * self._max_bets_per_round

    self._obs_size = players + self._total_cards * 2 + players
    self._info_state_size = (
        players + self._total_cards * 2 + self._max_game_length * 2
    )

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return _NUM_ACTIONS

  @property
  def num_players(self) -> int:
    """The number of players."""
    return self._num_players

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self._obs_size,)

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    return (self._info_state_size,)

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return self._total_cards

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return GameState(
        private_cards=jnp.full(
            self._num_players, game_lib.INVALID_CARD, dtype=jnp.int32
        ),
        public_card=jnp.array(game_lib.INVALID_CARD, dtype=jnp.int32),
        deck_mask=jnp.ones(self._total_cards, dtype=jnp.bool_),
        current_player=jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
        round=jnp.array(1, dtype=jnp.int32),
        stakes=jnp.array(_ANTE, dtype=jnp.int32),
        pot=jnp.array(_ANTE * self._num_players, dtype=jnp.int32),
        ante=jnp.full(self._num_players, _ANTE, dtype=jnp.int32),
        money=jnp.full(
            self._num_players, _STARTING_MONEY - _ANTE, dtype=jnp.int32
        ),
        folded=jnp.zeros(self._num_players, dtype=jnp.bool_),
        round1_sequence=jnp.full(
            self._max_bets_per_round, NO_ACTION, dtype=jnp.int32
        ),
        round2_sequence=jnp.full(
            self._max_bets_per_round, NO_ACTION, dtype=jnp.int32
        ),
        round1_seq_len=jnp.array(0, dtype=jnp.int32),
        round2_seq_len=jnp.array(0, dtype=jnp.int32),
        num_raises=jnp.array(0, dtype=jnp.int32),
        num_calls=jnp.array(0, dtype=jnp.int32),
        private_cards_dealt=jnp.array(0, dtype=jnp.int32),
        remaining_players=jnp.array(self._num_players, dtype=jnp.int32),
        is_terminal=jnp.array(False),
        legal_actions_mask=jnp.zeros(_NUM_ACTIONS, dtype=jnp.bool_),
    )

  def _compute_legal_actions_mask(self, state: GameState) -> jax.Array:
    """Compute legal actions for a player node.

    Args:
      state: The state.

    Returns:
      bool[3], the legal fold, call and raise.
    """
    is_chance = state.current_player == game_lib.CHANCE_PLAYER_ID
    is_terminal = state.is_terminal

    # Can fold only if facing a bet (stakes > player's ante).
    cur_player = jnp.maximum(0, state.current_player)
    can_fold = state.stakes > state.ante[cur_player]
    # Can always call/check.
    can_call = True
    # Can raise if under max raises.
    can_raise = state.num_raises < _MAX_RAISES

    mask = jnp.array([can_fold, can_call, can_raise])
    # Zero out if chance or terminal.
    mask = jnp.where(is_chance | is_terminal, jnp.zeros_like(mask), mask)
    return mask

  def _apply_chance_action(
      self, state: GameState, action: jax.Array
  ) -> GameState:
    """Handle card dealing.

    Args:
      state: The state.
      action: The card dealt.

    Returns:
      The state after the deal.
    """
    is_private = state.private_cards_dealt < self._num_players

    # Branch 1: Deal private card.
    p_idx = state.private_cards_dealt
    new_private_cards = state.private_cards.at[p_idx].set(action)
    new_deck_mask = state.deck_mask.at[action].set(False)
    new_dealt = state.private_cards_dealt + 1

    # If all private cards dealt, player 0 starts betting.
    next_player = jnp.where(
        new_dealt == self._num_players,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
    )

    state_private = state._replace(
        private_cards=new_private_cards,
        deck_mask=new_deck_mask,
        private_cards_dealt=new_dealt,
        current_player=next_player,
    )

    # Branch 2: Deal public card.
    state_public = state._replace(
        public_card=action,
        deck_mask=new_deck_mask,
        current_player=jnp.array(0, dtype=jnp.int32),  # Player 0 starts round 2
    )

    new_state = jax.tree.map(
        lambda pr, pu: jnp.where(is_private, pr, pu),
        state_private,
        state_public,
    )

    # Update legal actions for the new state.
    mask = self._compute_legal_actions_mask(new_state)
    return new_state._replace(legal_actions_mask=mask)

  def _apply_player_action(
      self, state: GameState, action: jax.Array
  ) -> GameState:
    """Handle betting actions.

    Args:
      state: The state.
      action: The fold, call or raise.

    Returns:
      The state after the action.
    """
    cur_player = state.current_player

    is_fold = action == _FOLD
    is_call = action == _CALL
    is_raise = action == _RAISE

    # Branch 0: Fold.
    new_folded = state.folded.at[cur_player].set(True)
    new_remaining = jnp.where(
        is_fold, state.remaining_players - 1, state.remaining_players
    )
    is_terminal_fold = is_fold & (new_remaining == 1)

    # Branch 1: Call.
    amount_to_call = state.stakes - state.ante[cur_player]
    new_ante_call = state.ante.at[cur_player].add(amount_to_call)
    new_money_call = state.money.at[cur_player].subtract(amount_to_call)
    new_pot_call = state.pot + amount_to_call
    new_num_calls = state.num_calls + 1

    # Branch 2: Raise.
    raise_amount = jnp.where(
        state.round == 1, _FIRST_RAISE_AMOUNT, _SECOND_RAISE_AMOUNT
    )
    total_put_in = amount_to_call + raise_amount
    new_ante_raise = state.ante.at[cur_player].add(total_put_in)
    new_money_raise = state.money.at[cur_player].subtract(total_put_in)
    new_pot_raise = state.pot + total_put_in
    new_stakes = state.stakes + raise_amount
    new_num_raises = state.num_raises + 1
    # Reset calls on raise.

    # Determine next player and round transitions.

    # Update sequences using proper sequence length counters.
    r1_idx = state.round1_seq_len
    r2_idx = state.round2_seq_len

    new_round1_seq = state.round1_sequence
    new_round2_seq = state.round2_sequence

    # Update the appropriate sequence.
    new_round1_seq = jnp.where(
        state.round == 1, new_round1_seq.at[r1_idx].set(action), new_round1_seq
    )
    new_round2_seq = jnp.where(
        state.round == 2, new_round2_seq.at[r2_idx].set(action), new_round2_seq
    )

    new_r1_len = jnp.where(state.round == 1, r1_idx + 1, state.round1_seq_len)
    new_r2_len = jnp.where(state.round == 2, r2_idx + 1, state.round2_seq_len)

    # Compute actual post-action num_calls and num_raises.
    # Fold: no change to num_calls or num_raises.
    # Call: num_calls++.
    # Raise: num_raises++, num_calls = 0.
    actual_num_calls = jnp.where(
        is_call,
        new_num_calls,
        jnp.where(is_raise, jnp.array(0, dtype=jnp.int32), state.num_calls),
    )
    actual_num_raises = jnp.where(is_raise, new_num_raises, state.num_raises)

    # Ready for next round or terminal?
    # ReadyForNextRound: (raises==0 && calls==remaining) ||
    #                    (raises>0 && calls==remaining-1)
    ready_next_round = jnp.where(
        actual_num_raises == 0,
        actual_num_calls == state.remaining_players,
        actual_num_calls == (state.remaining_players - 1),
    )
    # Only call/raise can trigger ready_next_round (fold goes to terminal).
    ready_next_round = ready_next_round & (~is_fold)

    # Terminal if fold (remaining=1) or (round=2 and ready_next_round).
    term = is_terminal_fold | ((state.round == 2) & ready_next_round)

    # Next player.
    next_p = 1 - cur_player  # 2 players only!

    # Transition to round 2.
    goes_to_round2 = (state.round == 1) & ready_next_round & (~term)

    # NewRound resets num_calls and num_raises to 0.
    final_num_calls = jnp.where(
        goes_to_round2, jnp.array(0, dtype=jnp.int32), actual_num_calls
    )
    final_num_raises = jnp.where(
        goes_to_round2, jnp.array(0, dtype=jnp.int32), actual_num_raises
    )

    result_state = state._replace(
        folded=jnp.where(is_fold, new_folded, state.folded),
        remaining_players=new_remaining,
        is_terminal=term,
        ante=jnp.where(
            is_call,
            new_ante_call,
            jnp.where(is_raise, new_ante_raise, state.ante),
        ),
        money=jnp.where(
            is_call,
            new_money_call,
            jnp.where(is_raise, new_money_raise, state.money),
        ),
        pot=jnp.where(
            is_call, new_pot_call, jnp.where(is_raise, new_pot_raise, state.pot)
        ),
        num_calls=final_num_calls,
        num_raises=final_num_raises,
        stakes=jnp.where(is_raise, new_stakes, state.stakes),
        round1_sequence=new_round1_seq,
        round2_sequence=new_round2_seq,
        round1_seq_len=new_r1_len,
        round2_seq_len=new_r2_len,
        current_player=jnp.where(
            term,
            jnp.array(game_lib.TERMINAL_PLAYER_ID, dtype=jnp.int32),
            jnp.where(
                goes_to_round2,
                jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
                next_p,
            ),
        ),
        round=jnp.where(
            goes_to_round2, jnp.array(2, dtype=jnp.int32), state.round
        ),
    )

    # Recompute legal actions.
    mask = self._compute_legal_actions_mask(result_state)
    return result_state._replace(legal_actions_mask=mask)

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
    # An int64 action (the default integer dtype under x64) would promote
    # int32 state fields such as public_card, so the state's pytree type would
    # change across a lax.scan step. Casting once here keeps every field int32.
    action = jnp.asarray(action, dtype=jnp.int32)
    is_chance = state.current_player == game_lib.CHANCE_PLAYER_ID

    chance_state = self._apply_chance_action(state, action)
    player_state = self._apply_player_action(state, action)

    return jax.tree.map(
        lambda c, p: jnp.where(is_chance, c, p), chance_state, player_state
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
    return (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    ) & ~state.is_terminal

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    return state.legal_actions_mask

  def returns(self, state: GameState) -> jax.Array:
    """Returns the chips won or lost by each player.

    Args:
      state: The state.

    Returns:
      f32[2]; zero before the end.
    """
    # If not terminal, return zeros.
    # Fold case: remaining_players==1, non-folded player gets the pot.
    p0_folded = state.folded[0]
    p1_folded = state.folded[1]

    # Showdown case: rank hands.
    # card // _NUM_SUITS gives the rank: J=0, Q=1, K=2.
    # A pair is when private card rank == public card rank.
    pub_rank = state.public_card // _NUM_SUITS
    p0_rank = state.private_cards[0] // _NUM_SUITS
    p1_rank = state.private_cards[1] // _NUM_SUITS

    p0_is_pair = p0_rank == pub_rank
    p1_is_pair = p1_rank == pub_rank

    # Pair beats non-pair. Among same type, higher rank wins.
    # Since both players share the same public card, we only
    # need to compare private card ranks for non-pair ties.
    p0_wins_showdown = (
        (~p0_folded)
        & (~p1_folded)
        & (
            (p0_is_pair & ~p1_is_pair)
            | (p0_is_pair & p1_is_pair & (p0_rank > p1_rank))
            | (~p0_is_pair & ~p1_is_pair & (p0_rank > p1_rank))
        )
    )
    p1_wins_showdown = (
        (~p0_folded)
        & (~p1_folded)
        & (
            (p1_is_pair & ~p0_is_pair)
            | (p0_is_pair & p1_is_pair & (p1_rank > p0_rank))
            | (~p0_is_pair & ~p1_is_pair & (p1_rank > p0_rank))
        )
    )
    is_draw = (
        (~p0_folded) & (~p1_folded) & ~p0_wins_showdown & ~p1_wins_showdown
    )

    # Fold case.
    p0_wins_fold = p1_folded & (~p0_folded)
    p1_wins_fold = p0_folded & (~p1_folded)

    p0_wins = p0_wins_showdown | p0_wins_fold
    p1_wins = p1_wins_showdown | p1_wins_fold

    # Compute money after ResolveWinner.
    # Winner gets the full pot. Draw splits pot.
    pot = state.pot.astype(jnp.float32)
    p0_pot_share = jnp.where(p0_wins, pot, jnp.where(is_draw, pot / 2.0, 0.0))
    p1_pot_share = jnp.where(p1_wins, pot, jnp.where(is_draw, pot / 2.0, 0.0))

    p0_final_money = state.money[0].astype(jnp.float32) + p0_pot_share
    p1_final_money = state.money[1].astype(jnp.float32) + p1_pot_share

    ret0 = p0_final_money - _STARTING_MONEY
    ret1 = p1_final_money - _STARTING_MONEY

    ret = jnp.array([ret0, ret1])
    return jnp.where(state.is_terminal, ret, jnp.zeros(2, dtype=jnp.float32))

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """Returns the chance distribution: uniform over the cards not yet dealt.

    Args:
      state: The state.

    Returns:
      f32[num_cards].
    """
    probs = state.deck_mask.astype(jnp.float32)
    total_remaining = jnp.sum(probs)
    return probs / total_remaining

  def _encode_common(self, state, observing_player):
    """The one-hots shared by the observation and the information state.

    Args:
      state: The state.
      observing_player: The observing player.

    Returns:
      f32[...], the player, private card and public card one-hots.
    """
    # Observing player (one-hot)
    player_onehot = jnp.zeros(self._num_players, dtype=jnp.float32)
    player_onehot = player_onehot.at[observing_player].set(1.0)

    # Private card (one-hot)
    private_card = state.private_cards[observing_player]
    private_onehot = jnp.zeros(self._total_cards, dtype=jnp.float32)
    private_onehot = jnp.where(
        private_card != game_lib.INVALID_CARD,
        private_onehot.at[private_card].set(1.0),
        private_onehot,
    )

    # Public card (one-hot)
    public_card = state.public_card
    public_onehot = jnp.zeros(self._total_cards, dtype=jnp.float32)
    public_onehot = jnp.where(
        public_card != game_lib.INVALID_CARD,
        public_onehot.at[public_card].set(1.0),
        public_onehot,
    )

    return jnp.concatenate([player_onehot, private_onehot, public_onehot])

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    header = self._encode_common(state, observing_player)

    # Pot contribution per player (integers)
    contrib = state.ante.astype(jnp.float32)

    return jnp.concatenate([header, contrib])

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns everything `observing_player` has observed so far.

    In a perfect-information game the current observation determines the
    information state, and the game returns its observation tensor.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*info_state_shape]: the information-state tensor.
    """
    header = self._encode_common(state, observing_player)

    # Each betting slot is two bits: call (1, 0), raise (0, 1), else (0, 0).

    def encode_seq(seq):
      """Encodes a round's bet sequence as (call, raise) bits.

      Args:
        seq: The round's actions.

      Returns:
        f32[2 * max_bets_per_round].
      """
      bits = jnp.zeros((self._max_bets_per_round, 2), dtype=jnp.float32)
      for i in range(self._max_bets_per_round):
        action = seq[i]
        is_call = action == _CALL
        is_raise = action == _RAISE
        bits = bits.at[i, 0].set(jnp.where(is_call, 1.0, 0.0))
        bits = bits.at[i, 1].set(jnp.where(is_raise, 1.0, 0.0))
      return bits.flatten()

    b1 = encode_seq(state.round1_sequence)
    b2 = encode_seq(state.round2_sequence)

    return jnp.concatenate([header, b1, b2])

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
    return jnp.array(os_state.observation_tensor(player), dtype=jnp.float32)

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
    return jnp.array(
        os_state.information_state_tensor(player), dtype=jnp.float32
    )
