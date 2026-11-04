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
"""Liar's dice, OpenSpiel's `liars_dice`, with OpenSpiel's parameters.

`numdice` dice per player (default 1) of `dice_sides` faces (default 6), two
players and the `reset-face` bidding rule: OpenSpiel's defaults. The highest
face is wild. The two-dice, five-face benchmark of Rudolph et al. (2025) is
`liars_dice(numdice=2,dice_sides=5)` (`BENCHMARK_NUMDICE`,
`BENCHMARK_DICE_SIDES`).

State: a dict with each player's dice (-1 until rolled), the bid history,
the current bid and the outcome. The 2 * numdice chance nodes roll player
0's dice then player 1's; chance action id f is face f + 1, uniform. Each
player's dice are sorted once all are rolled. Players then alternate bids,
player 0 first.

Actions, as in OpenSpiel: bid b in [0, 2 * numdice * dice_sides) claims at
least b // dice_sides + 1 dice showing face b % dice_sides + 1 (a bid must
exceed the current one); action 2 * numdice * dice_sides calls "liar".
Observation (f32[2 + numdice * dice_sides + num_actions]): observer one-hot,
own dice as face one-hots, and the last two bids as a multi-hot.
Information state (same size): the same with every bid so far. Returns: +1
for the winner of the challenge, -1 for the loser.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_PLAYERS = 2
# OpenSpiel's defaults.
DEFAULT_NUMDICE = 1
DEFAULT_DICE_SIDES = 6
# The Rudolph et al. (2025) benchmark: liars_dice(numdice=2,dice_sides=5).
BENCHMARK_NUMDICE = 2
BENCHMARK_DICE_SIDES = 5

STATE_DTYPES = {
    "dice": jnp.int8,
    "num_rolled": jnp.int8,
    "current_player": jnp.int8,
    "current_bid": jnp.int8,
    "bid_history": jnp.int8,
    "num_bids": jnp.int8,
    "bidding_player": jnp.int8,
    "calling_player": jnp.int8,
    "winner": jnp.int8,
    "loser": jnp.int8,
    "done": jnp.bool_,
    "returns": jnp.float32,
}


def assert_state_dtypes(state):
  """Raises if a field of `state` has a dtype other than STATE_DTYPES.

  A check for tests.

  Args:
    state: The state dict.
  """
  for field, expected_dtype in STATE_DTYPES.items():
    if field in state:
      chex.assert_type(state[field], expected_dtype)


def num_bids(numdice, dice_sides):
  """Returns the number of bids: OpenSpiel's total dice times faces.

  Args:
    numdice: The dice per player.
    dice_sides: The faces per die.

  Returns:
    The number of bids.
  """
  return NUM_PLAYERS * numdice * dice_sides


def liar_action(numdice, dice_sides):
  """Returns the "liar" call: the action after the last bid.

  Args:
    numdice: The dice per player.
    dice_sides: The faces per die.

  Returns:
    The action id.
  """
  return num_bids(numdice, dice_sides)


def num_actions(numdice, dice_sides):
  """Returns OpenSpiel's NumDistinctActions: every bid and the call.

  Args:
    numdice: The dice per player.
    dice_sides: The faces per die.

  Returns:
    The number of actions.
  """
  return num_bids(numdice, dice_sides) + 1


class LiarsDice(game_lib.Game, interop.PyspielInterop):
  """Liar's dice for two players with OpenSpiel's parameters and defaults.

  Attributes:
    dice_sides: The faces per die.
    liar: The action id of the "liar" call.
    numdice: The dice per player.
  """

  def __init__(
      self,
      numdice: int = DEFAULT_NUMDICE,
      dice_sides: int = DEFAULT_DICE_SIDES,
      players: int = NUM_PLAYERS,
      bidding_rule: str = "reset-face",
  ):
    for name, value in (
        ("numdice", numdice),
        ("dice_sides", dice_sides),
        ("players", players),
    ):
      if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"liars_dice {name} must be an int, got {value!r}")
    if players != NUM_PLAYERS:
      raise ValueError(
          f"liars_dice supports only players={NUM_PLAYERS}, got {players}"
      )
    if bidding_rule != "reset-face":
      raise ValueError(
          "liars_dice supports only bidding_rule='reset-face', "
          f"got {bidding_rule!r}"
      )
    if numdice < 1 or dice_sides < 1:
      raise ValueError(
          "liars_dice needs numdice >= 1 and dice_sides >= 1, got "
          f"numdice={numdice}, dice_sides={dice_sides}"
      )
    # The state keeps actions in int8.
    if num_actions(numdice, dice_sides) > np.iinfo(np.int8).max:
      raise ValueError(
          f"liars_dice(numdice={numdice},dice_sides={dice_sides}) has "
          f"{num_actions(numdice, dice_sides)} actions; at most "
          f"{np.iinfo(np.int8).max} are supported"
      )
    self.numdice = numdice
    self.dice_sides = dice_sides
    self.liar = liar_action(numdice, dice_sides)
    self._num_actions = num_actions(numdice, dice_sides)
    self._obs_dim = NUM_PLAYERS + numdice * dice_sides + self._num_actions

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self._num_actions

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self._obs_dim,)

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    return (self._obs_dim,)

  @property
  def num_players(self):
    """The number of players."""
    return NUM_PLAYERS

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return self.dice_sides

  @property
  def max_game_length(self) -> int:
    """OpenSpiel's MaxGameLength: every bid, then the call."""
    return self._num_actions

  def new_initial_state(self):
    """Returns the initial state, at player 0's first die roll.

    Returns:
      The state.
    """
    return {
        "dice": jnp.full((NUM_PLAYERS, self.numdice), -1, dtype=jnp.int8),
        "num_rolled": jnp.zeros(NUM_PLAYERS, dtype=jnp.int8),
        "current_player": jnp.int8(game_lib.CHANCE_PLAYER_ID),
        "current_bid": jnp.int8(-1),
        "bid_history": jnp.full((self._num_actions,), -1, dtype=jnp.int8),
        "num_bids": jnp.int8(0),
        "bidding_player": jnp.int8(-1),
        "calling_player": jnp.int8(-1),
        "winner": jnp.int8(-1),
        "loser": jnp.int8(-1),
        "done": jnp.bool_(False),
        "returns": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }

  def _resolve_winner(self, dice, current_bid, bidding_player, calling_player):
    """The winner and loser of a liar call.

    Args:
      dice: The dice.
      current_bid: The bid called.
      bidding_player: The bidder.
      calling_player: The caller.

    Returns:
      (winner, loser), each int8[].
    """
    quantity = current_bid // self.dice_sides + 1
    face = current_bid % self.dice_sides + 1
    matches = jnp.sum((dice == face) | (dice == self.dice_sides))
    bidder_wins = matches >= quantity
    winner = jnp.where(bidder_wins, bidding_player, calling_player)
    loser = jnp.where(bidder_wins, calling_player, bidding_player)
    return winner.astype(jnp.int8), loser.astype(jnp.int8)

  def _handle_chance(self, state, action):
    """Rolls one die for the next player still rolling.

    Args:
      state: The state.
      action: The face rolled, minus one.

    Returns:
      The state after the roll.
    """
    dice = state["dice"]
    num_rolled = state["num_rolled"]
    roller = jnp.where(num_rolled[0] < self.numdice, 0, 1)
    slot = num_rolled[roller]
    new_dice = dice.at[roller, slot].set(jnp.asarray(action + 1, jnp.int8))
    new_num_rolled = num_rolled.at[roller].add(jnp.int8(1))
    all_rolled = jnp.all(new_num_rolled == self.numdice)
    new_dice = jnp.where(all_rolled, jnp.sort(new_dice, axis=1), new_dice)
    new_current_player = jnp.where(
        all_rolled, jnp.int8(0), jnp.int8(game_lib.CHANCE_PLAYER_ID)
    )
    return {
        **state,
        "dice": new_dice,
        "num_rolled": new_num_rolled,
        "current_player": new_current_player,
    }

  def _handle_bid(self, state, action):
    """Applies a bid or a liar call.

    Args:
      state: The state.
      action: The bid or the liar action.

    Returns:
      The state after the action.
    """
    player = state["current_player"]
    current_bid = state["current_bid"]
    is_liar = action == self.liar
    new_bid_history = (
        state["bid_history"]
        .at[state["num_bids"]]
        .set(jnp.asarray(action, jnp.int8))
    )
    new_calling_player = jnp.where(is_liar, player, state["calling_player"])
    new_bidding_player = jnp.where(is_liar, state["bidding_player"], player)
    new_current_bid = jnp.where(
        is_liar, current_bid, jnp.asarray(action, jnp.int8)
    )
    new_winner, new_loser = self._resolve_winner(
        state["dice"], current_bid, state["bidding_player"], new_calling_player
    )
    new_winner = jnp.where(is_liar, new_winner, state["winner"])
    new_loser = jnp.where(is_liar, new_loser, state["loser"])
    p0_return = jnp.where(is_liar, jnp.where(new_winner == 0, 1.0, -1.0), 0.0)
    return {
        **state,
        "current_player": jnp.where(
            is_liar, jnp.int8(game_lib.TERMINAL_PLAYER_ID), jnp.int8(1) - player
        ),
        "current_bid": new_current_bid,
        "bid_history": new_bid_history,
        "num_bids": state["num_bids"] + jnp.int8(1),
        "bidding_player": new_bidding_player,
        "calling_player": new_calling_player,
        "winner": new_winner,
        "loser": new_loser,
        "done": jnp.asarray(is_liar, jnp.bool_),
        "returns": jnp.array([p0_return, -p0_return], dtype=jnp.float32),
    }

  def apply_action(self, state, action):
    """Returns the state after a die roll, a bid or a "liar" call.

    A terminal state is returned unchanged.

    Args:
      state: The state.
      action: The die face, bid or call.

    Returns:
      The next state.
    """

    def active(_):
      """Applies the action at a live state.

      Args:
        _: Unused.

      Returns:
        The next state.
      """
      chance_state = self._handle_chance(state, action)
      bid_state = self._handle_bid(state, action)
      return jax.tree.map(
          lambda c, b: jnp.where(self.is_chance_node(state), c, b),
          chance_state,
          bid_state,
      )

    return jax.lax.cond(state["done"], lambda _: state, active, operand=None)

  def current_player(self, state):
    """Returns the player to move.

    Args:
      state: The state.

    Returns:
      i32[]: the player, CHANCE_PLAYER_ID or TERMINAL_PLAYER_ID.
    """
    return state["current_player"].astype(jnp.int32)

  def is_terminal(self, state):
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state["done"]

  def is_chance_node(self, state):
    """Returns whether dice are being rolled.

    Args:
      state: The state.

    Returns:
      bool[].
    """
    return state["current_player"] == game_lib.CHANCE_PLAYER_ID

  def chance_outcomes(self, state):
    """Returns the uniform die-roll distribution.

    Args:
      state: Unused.

    Returns:
      f32[dice_sides].
    """
    del state
    return jnp.full((self.dice_sides,), 1.0 / self.dice_sides, jnp.float32)

  def legal_actions_mask(self, state):
    """Returns the legal bids and call.

    At a chance node, the first `dice_sides` entries (the roll outcomes).

    Args:
      state: The state.

    Returns:
      bool[num_actions].
    """
    actions = jnp.arange(self._num_actions, dtype=jnp.int32)
    chance_mask = actions < self.dice_sides
    bid_mask = (actions > state["current_bid"]) & (actions < self.liar)
    liar_mask = (actions == self.liar) & (state["num_bids"] > 0)
    mask = jnp.where(
        self.is_chance_node(state), chance_mask, bid_mask | liar_mask
    )
    return mask & ~state["done"]

  def returns(self, state):
    """Returns each player's return.

    Args:
      state: The state.

    Returns:
      f32[2]; zero before the end.
    """
    return state["returns"]

  def _own_dice(self, state, player):
    """The one-hot faces of `player`'s dice.

    Args:
      state: The state.
      player: The player.

    Returns:
      f32[numdice * dice_sides], the one-hots.
    """
    dice = state["dice"][player]
    onehot = jax.nn.one_hot(jnp.maximum(dice - 1, 0), self.dice_sides)
    return jnp.where((dice >= 1)[:, None], onehot, 0.0).reshape(-1)

  def _tensor(self, state, player, visible):
    """The tensor of `player`: player, own dice and the bids made.

    Args:
      state: The state.
      player: The player.
      visible: bool[...], the bids to include.

    Returns:
      f32[...], the tensor.
    """
    bid_indices = jnp.where(
        visible, state["bid_history"].astype(jnp.int32), self._num_actions
    )
    bids = jnp.any(
        jax.nn.one_hot(bid_indices, self._num_actions + 1, dtype=jnp.bool_)[
            :, : self._num_actions
        ],
        axis=0,
    )
    return jnp.concatenate([
        jax.nn.one_hot(player, NUM_PLAYERS),
        self._own_dice(state, player),
        bids.astype(jnp.float32),
    ]).astype(jnp.float32)

  def observation_tensor(self, state, observing_player):
    """Returns OpenSpiel's observation: the last two bids.

    Args:
      state: The state.
      observing_player: The observing player.

    Returns:
      The observation tensor.
    """
    index = jnp.arange(self._num_actions, dtype=jnp.int32)
    n = state["num_bids"].astype(jnp.int32)
    return self._tensor(
        state, observing_player, (index < n) & (index >= n - NUM_PLAYERS)
    )

  def information_state_tensor(self, state, observing_player):
    """Returns OpenSpiel's information state: every bid so far.

    Args:
      state: The state.
      observing_player: The observing player.

    Returns:
      The information-state tensor.
    """
    return self._tensor(state, observing_player, state["bid_history"] >= 0)

  def observation_from_os(self, os_state, player):
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    return jnp.asarray(os_state.observation_tensor(player), dtype=jnp.float32)

  def information_state_from_os(self, os_state, player):
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
        os_state.information_state_tensor(player), dtype=jnp.float32
    )
