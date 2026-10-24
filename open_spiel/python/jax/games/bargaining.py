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

"""Bargaining, OpenSpiel's `bargaining` negotiation game (two players).

Two players split a pool of three item types (DeVault et al. 2015; Lewis et
al. 2017). Each player has private values for the items, and the players
alternate offers until one accepts or `max_turns` offers are made.

State: a `GameState` with the pool, both value vectors, the offers made, the
player to move and the discount. The first chance node picks an instance
(pool and values) uniformly from `instances`, by default the 1000 instances
bundled in `bargaining_instances.txt` (a copy of OpenSpiel's
`open_spiel/games/bargaining/bargaining_instances1000.txt`, the instances its
C++ game loads by default). With `prob_end > 0`, a chance node
after each offer from the second on ends the game with probability
`prob_end` or continues. Chance outcome ids are OpenSpiel's: the instance
draw is the instance index, and the continue / end outcomes come after the
instances (`continue_outcome` = number of instances, `end_outcome` = that
+ 1; 1000 and 1001 with the bundled instances), so `max_chance_outcomes` is
the number of instances + 2.

Actions: offer i ∈ [0, 120) proposes the i-th split in OpenSpiel's
enumeration of item counts (a, b, c) with a + b + c <= 7, giving the
proposer that bundle; action 120 accepts the last offer.

Observation: an agreement bit, the number of offers one-hot, pool and own
values as thermometers, and the last offer as a thermometer (zero before the
first offer).
Information state: the same header and every offer so far. Both are zero at
chance nodes and match OpenSpiel's layouts.

Returns: on acceptance, each player's value of their bundle, times
`discount` per turn from the third offer on; 0 for both otherwise.
"""

import os
import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel

# Constants of OpenSpiel's bargaining.
_NUM_ITEM_TYPES = 3
_POOL_MAX_NUM_ITEMS = 7
_TOTAL_VALUE_ALL_ITEMS = 10
_NUM_PLAYERS = 2

_DEFAULT_MAX_TURNS = 10
_DEFAULT_DISCOUNT = 1.0
_DEFAULT_PROB_END = 0.0


def enumerate_offers():
  """Enumerates all valid offers, matching OpenSpiel's order exactly.

  Returns:
    i32[num_offers, 3].
  """
  offers = []
  for c in range(_POOL_MAX_NUM_ITEMS + 1):
    for b in range(_POOL_MAX_NUM_ITEMS + 1):
      for a in range(_POOL_MAX_NUM_ITEMS + 1):
        if a + b + c <= _POOL_MAX_NUM_ITEMS:
          offers.append([a, b, c])
  return np.array(offers, dtype=np.int32)


def _parse_instances(text: str) -> np.ndarray:
  """Parse instance text into array of shape (N, 3, 3).

  Each row is [pool, p0_values, p1_values], each of shape (3,).

  Args:
    text: A string containing the instances, with each line representing one
      instance.

  Returns:
    A numpy array of shape (N, 3, 3) containing the parsed instances.
  """
  instances = []
  for line in text.strip().split("\n"):
    line = line.strip()
    if not line:
      continue
    parts = line.split()
    pool = [int(x) for x in parts[0].split(",")]
    vals0 = [int(x) for x in parts[1].split(",")]
    vals1 = [int(x) for x in parts[2].split(",")]
    instances.append([pool, vals0, vals1])
  return np.array(instances, dtype=np.int32)


def _load_default_instances() -> np.ndarray:
  """Load the default 1000 instances from the bundled data file.

  Returns:
    int[1000, 3, 3]: the parsed instances.
  """
  data_path = os.path.join(
      os.path.dirname(__file__), "bargaining_instances.txt"
  )
  with open(data_path, encoding="utf-8") as f:
    return _parse_instances(f.read())


def _thermometer(value, max_val):
  """Thermometer encoding: bits 0..value are 1, rest are 0.

  Args:
    value: The value.
    max_val: The largest value.

  Returns:
    f32[max_val + 1], the thermometer bits.
  """
  return (jnp.arange(max_val + 1) <= value).astype(jnp.float32)


class GameState(NamedTuple):
  """State of the Bargaining game.

  Attributes:
    agreement_reached: Scalar bool.
    current_player: Scalar int32: game_lib.CHANCE_PLAYER_ID (chance), 0, 1.
    discount: Scalar float32.
    game_ended: Scalar bool (ended by prob_end).
    is_terminal: Scalar bool.
    legal_actions_mask: (num_actions,) bool.
    num_offers: Scalar int32.
    offers: (max_turns, 3) int32.
    pool: (3,) int32.
    values: (2, 3) int32.
  """

  pool: jax.Array
  values: jax.Array
  offers: jax.Array
  num_offers: jax.Array
  current_player: jax.Array
  agreement_reached: jax.Array
  is_terminal: jax.Array
  game_ended: jax.Array
  discount: jax.Array
  legal_actions_mask: jax.Array


class Bargaining(game_lib.Game, interop.PyspielInterop):
  """Two-player bargaining over three item types.

  Attributes:
    all_offers: Every offer, indexed by its action id.
  """

  def __init__(
      self,
      max_turns: int = _DEFAULT_MAX_TURNS,
      discount: float = _DEFAULT_DISCOUNT,
      prob_end: float = _DEFAULT_PROB_END,
      instances: np.ndarray | None = None,
  ):
    self._max_turns = max_turns
    self._discount = discount
    self._prob_end = prob_end

    # Offers: shape (N, 3), N = 120 for default pool_max=7.
    self.all_offers = enumerate_offers()
    self._num_offers_total = len(self.all_offers)
    self._agree_action = self._num_offers_total  # Last action = agree.
    self._all_offers_jax = jnp.array(self.all_offers)

    # Instances: shape (num_instances, 3, 3).
    if instances is not None:
      self._instances = instances
    else:
      self._instances = _load_default_instances()
    self._instances_jax = jnp.array(self._instances)
    self._num_instances = len(self._instances)

    # Observation sizes.
    self._obs_size = (
        (max_turns + 2)  # one-hot num_offers
        + _NUM_ITEM_TYPES * (_POOL_MAX_NUM_ITEMS + 1)  # pool thermo
        + _NUM_ITEM_TYPES * (_TOTAL_VALUE_ALL_ITEMS + 1)  # my values thermo
        + _NUM_ITEM_TYPES * (_POOL_MAX_NUM_ITEMS + 1)  # last offer thermo
    )
    self._info_state_size = (
        (max_turns + 2)  # one-hot num_offers
        + _NUM_ITEM_TYPES * (_POOL_MAX_NUM_ITEMS + 1)  # pool thermo
        + _NUM_ITEM_TYPES * (_TOTAL_VALUE_ALL_ITEMS + 1)  # my values thermo
        + max_turns
        * _NUM_ITEM_TYPES
        * (_POOL_MAX_NUM_ITEMS + 1)  # all offers thermo
    )

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self._num_offers_total + 1  # offers + agree

  @property
  def num_players(self) -> int:
    """The number of players."""
    return _NUM_PLAYERS

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
    # OpenSpiel's MaxChanceOutcomes: the instances, then continue and end.
    return self._num_instances + 2

  @property
  def continue_outcome(self) -> int:
    """The id of the mid-game chance outcome that continues the game."""
    return self._num_instances

  @property
  def end_outcome(self) -> int:
    """The id of the mid-game chance outcome that ends the game."""
    return self._num_instances + 1

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return GameState(
        pool=jnp.zeros(_NUM_ITEM_TYPES, dtype=jnp.int32),
        values=jnp.zeros((_NUM_PLAYERS, _NUM_ITEM_TYPES), dtype=jnp.int32),
        offers=jnp.zeros((self._max_turns, _NUM_ITEM_TYPES), dtype=jnp.int32),
        num_offers=jnp.array(0, dtype=jnp.int32),
        current_player=jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
        agreement_reached=jnp.array(False),
        is_terminal=jnp.array(False),
        game_ended=jnp.array(False),
        discount=jnp.array(1.0, dtype=jnp.float32),
        legal_actions_mask=jnp.zeros(self.num_actions, dtype=jnp.bool_),
    )

  def _compute_legal_actions_mask(
      self, pool, num_offers, is_terminal, is_chance
  ):
    """Compute legal actions mask for a player node.

    Args:
      pool: int[3], the item pool.
      num_offers: The offers made so far.
      is_terminal: Whether the node is terminal.
      is_chance: Whether the node is a chance node.

    Returns:
      bool[num_actions], the legal-action mask.
    """
    # Each offer is legal if offer[i] <= pool[i] for all item types.
    offer_legal = jnp.all(self._all_offers_jax <= pool[None, :], axis=1)
    # Agree is legal if there has been at least one offer.
    agree_legal = num_offers > 0
    mask = jnp.concatenate([offer_legal, jnp.array([agree_legal])])
    # All actions are illegal at terminal or chance nodes.
    mask = mask & (~is_terminal) & (~is_chance)
    return mask

  def _apply_initial_chance(self, state, action):
    """Apply initial chance action: select instance.

    Args:
      state: The state.
      action: The instance index.

    Returns:
      The state with the instance's pool and values, player 0 to move.
    """
    instance = self._instances_jax[action]  # (3, 3)
    pool = instance[0]
    values = instance[1:]  # (2, 3)
    is_chance = jnp.array(False)
    mask = self._compute_legal_actions_mask(
        pool, jnp.array(0, dtype=jnp.int32), jnp.array(False), is_chance
    )
    return state._replace(
        pool=pool,
        values=values,
        current_player=jnp.array(0, dtype=jnp.int32),
        legal_actions_mask=mask,
    )

  def _apply_midgame_chance(self, state, action):
    """Apply mid-game chance action: continue_outcome or end_outcome.

    Args:
      state: The state.
      action: The continue or end outcome.

    Returns:
      The state after the chance outcome.
    """
    game_ends = action == self.end_outcome
    # Next player is the one whose turn it would be.
    next_player = state.num_offers % 2
    is_chance = jnp.array(False)
    mask = self._compute_legal_actions_mask(
        state.pool, state.num_offers, game_ends, is_chance
    )
    return state._replace(
        current_player=jnp.where(
            game_ends,
            jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
            next_player.astype(jnp.int32),
        ),
        is_terminal=game_ends,
        game_ended=game_ends,
        legal_actions_mask=mask,
    )

  def _apply_player_action(self, state, action):
    """Apply a player action: make an offer or agree.

    Args:
      state: The state.
      action: The offer or agree action.

    Returns:
      The state after the action.
    """
    is_agree = action == self._agree_action

    # --- Agree branch ---
    # The proposer is the player who made the last offer.
    # After agreement, game is terminal.

    # OpenSpiel discounts from move 3 on, counting the instance chance node
    # as move 0, so from the third offer on (num_offers >= 2).
    new_discount = jnp.where(
        state.num_offers >= 2,
        state.discount * self._discount,
        state.discount,
    )

    # --- Offer branch ---
    # Record the offer and increment num_offers.
    # Clamp action index to valid range for offer lookup.
    offer_idx = jnp.minimum(action, self._num_offers_total - 1)
    offer = self._all_offers_jax[offer_idx]

    new_offers = state.offers.at[state.num_offers].set(offer)
    new_num_offers = state.num_offers + 1

    # Determine terminal: max turns reached.
    max_turns_reached = new_num_offers >= self._max_turns

    # Next player: if prob_end > 0 and at least 2 offers have been made,
    # go to a mid-game chance node. Otherwise, alternate players.
    goes_to_chance = (self._prob_end > 0) & (new_num_offers >= 2)
    next_player_if_no_chance = 1 - state.current_player
    next_player = jnp.where(
        max_turns_reached,
        jnp.array(
            game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32
        ),  # unused: the state is terminal
        jnp.where(
            goes_to_chance,
            jnp.array(
                game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32
            ),  # chance node
            next_player_if_no_chance,
        ),
    )

    is_terminal_offer = max_turns_reached
    is_chance_next = goes_to_chance & (~max_turns_reached)

    mask = self._compute_legal_actions_mask(
        state.pool,
        new_num_offers,
        is_terminal_offer,
        is_chance_next,
    )

    # Build offer-branch state.
    offer_state = state._replace(
        offers=new_offers,
        num_offers=new_num_offers,
        current_player=next_player,
        is_terminal=is_terminal_offer,
        discount=new_discount,
        legal_actions_mask=mask,
    )

    # Build agree-branch state.
    agree_mask = jnp.zeros(self.num_actions, dtype=jnp.bool_)
    agree_state = state._replace(
        agreement_reached=jnp.array(True),
        is_terminal=jnp.array(True),
        current_player=jnp.array(game_lib.CHANCE_PLAYER_ID, dtype=jnp.int32),
        legal_actions_mask=agree_mask,
        discount=new_discount,
    )

    # Select between agree and offer branches.
    return jax.tree.map(
        lambda a, o: jnp.where(is_agree, a, o),
        agree_state,
        offer_state,
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
    initial_chance = self._apply_initial_chance(state, action)
    midgame_chance = self._apply_midgame_chance(state, action)
    player_action = self._apply_player_action(state, action)

    is_initial_chance = (state.current_player == game_lib.CHANCE_PLAYER_ID) & (
        state.num_offers == 0
    )
    is_midgame_chance = (state.current_player == game_lib.CHANCE_PLAYER_ID) & (
        state.num_offers > 0
    )

    # Select: initial_chance -> midgame_chance -> player_action.
    result = jax.tree.map(
        lambda ic, mc, pa: jnp.where(
            is_initial_chance, ic, jnp.where(is_midgame_chance, mc, pa)
        ),
        initial_chance,
        midgame_chance,
        player_action,
    )
    return result

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
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    # If no agreement or game ended by chance, both get 0.
    # If agreement: proposer gets dot(values[proposer], last_offer),
    #   other gets dot(values[other], pool - last_offer).
    # Proposer is the player who made the last offer.
    has_agreement = state.agreement_reached & (~state.game_ended)

    # The last offer is at index num_offers - 1.
    last_offer_idx = jnp.maximum(state.num_offers - 1, 0)
    last_offer = state.offers[last_offer_idx]

    # Proposer is the one who made the last offer.
    # Offers alternate: offer 0 is by P0, offer 1 is by P1, etc.
    proposer = last_offer_idx % 2
    other = 1 - proposer

    proposer_return = jnp.sum(state.values[proposer] * last_offer)
    other_return = jnp.sum(state.values[other] * (state.pool - last_offer))

    # Apply discount.
    proposer_return = proposer_return * state.discount
    other_return = other_return * state.discount

    # Build returns array: [p0_return, p1_return].
    ret = jnp.zeros(2, dtype=jnp.float32)
    ret = ret.at[proposer].set(proposer_return.astype(jnp.float32))
    ret = ret.at[other].set(other_return.astype(jnp.float32))

    return jnp.where(has_agreement, ret, jnp.zeros(2, dtype=jnp.float32))

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """Returns the chance distribution at `state`.

    Index i is the probability of chance action id i. At a chance node the
    entries are non-negative and sum to 1; outcomes that cannot occur have
    probability 0. Elsewhere the values are unspecified. The shape is static,
    so the method is safe under `jit` and `vmap`. To enumerate outcomes
    exactly, iterate over the nonzero entries; to simulate, sample an index.

    Games without chance nodes inherit this default, which returns `f32[0]`.

    Args:
      state: The current state.

    Returns:
      f32[max_chance_outcomes]: the chance distribution.

    Raises:
      NotImplementedError: if the game declares chance outcomes but does not
        override this method.
    """
    max_outcomes = self.max_chance_outcomes
    is_initial = state.num_offers == 0

    # Initial chance: uniform over instances.
    initial_probs = jnp.zeros(max_outcomes, dtype=jnp.float32)
    initial_probs = initial_probs.at[: self._num_instances].set(
        1.0 / self._num_instances
    )

    # Mid-game chance: 1 - prob_end on continue_outcome, prob_end on
    # end_outcome (OpenSpiel's ids, after the instances).
    midgame_probs = jnp.zeros(max_outcomes, dtype=jnp.float32)
    midgame_probs = midgame_probs.at[self.continue_outcome].set(
        1.0 - self._prob_end
    )
    midgame_probs = midgame_probs.at[self.end_outcome].set(self._prob_end)

    return jnp.where(is_initial, initial_probs, midgame_probs)

  def _encode_obs_header(self, state, observing_player):
    """Encodes the observation header: agreement, offers, pool and values.

    Args:
      state: The state.
      observing_player: The player whose view is encoded.

    Returns:
      f32[...], the header bits.
    """
    # Bit 0: agreement reached. Then the number of offers, one-hot over
    # 0 .. max_turns.
    num_offers_onehot = jnp.zeros(self._max_turns + 2, dtype=jnp.float32)
    num_offers_onehot = num_offers_onehot.at[0].set(
        state.agreement_reached.astype(jnp.float32)
    )
    num_offers_onehot = num_offers_onehot.at[1 + state.num_offers].set(1.0)

    # Pool thermometer.
    pool_thermo = jnp.concatenate([
        _thermometer(state.pool[i], _POOL_MAX_NUM_ITEMS)
        for i in range(_NUM_ITEM_TYPES)
    ])

    # My values thermometer.
    my_values = state.values[observing_player]
    values_thermo = jnp.concatenate([
        _thermometer(my_values[i], _TOTAL_VALUE_ALL_ITEMS)
        for i in range(_NUM_ITEM_TYPES)
    ])

    return jnp.concatenate([num_offers_onehot, pool_thermo, values_thermo])

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
    is_chance = (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    ) & ~state.is_terminal
    header = self._encode_obs_header(state, observing_player)

    # Last offer thermometer.
    last_offer_idx = jnp.maximum(state.num_offers - 1, 0)
    last_offer = state.offers[last_offer_idx]

    # Before the first offer the slot is all zeros, not a thermometer of 0,
    # as in OpenSpiel.
    has_offer = state.num_offers > 0

    last_offer_thermo = jnp.concatenate([
        _thermometer(last_offer[i], _POOL_MAX_NUM_ITEMS)
        for i in range(_NUM_ITEM_TYPES)
    ])
    # Zero out if no offers.
    last_offer_thermo = jnp.where(
        has_offer, last_offer_thermo, jnp.zeros_like(last_offer_thermo)
    )

    obs = jnp.concatenate([header, last_offer_thermo])

    # Zero out at chance nodes.
    obs = jnp.where(is_chance, jnp.zeros_like(obs), obs)
    return obs

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
    is_chance = (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    ) & ~state.is_terminal
    header = self._encode_obs_header(state, observing_player)

    # All offers thermometer: max_turns slots.
    offer_parts = []
    for k in range(self._max_turns):
      offer_k = state.offers[k]
      has_offer_k = k < state.num_offers
      offer_k = jnp.where(
          has_offer_k, offer_k, jnp.zeros(_NUM_ITEM_TYPES, dtype=jnp.int32)
      )
      for i in range(_NUM_ITEM_TYPES):
        thermo = _thermometer(offer_k[i], _POOL_MAX_NUM_ITEMS)
        thermo = jnp.where(
            has_offer_k,
            thermo,
            jnp.zeros(_POOL_MAX_NUM_ITEMS + 1, dtype=jnp.float32),
        )
        offer_parts.append(thermo)

    all_offers_thermo = jnp.concatenate(offer_parts)
    info_state = jnp.concatenate([header, all_offers_thermo])

    # Zero out at chance nodes.
    info_state = jnp.where(is_chance, jnp.zeros_like(info_state), info_state)
    return info_state

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
