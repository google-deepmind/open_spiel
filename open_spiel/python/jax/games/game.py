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

"""The abstract Game interface every game in this package implements.

A game is a set of pure functions over an immutable state pytree, so
whole games can be compiled with `jax.jit`, batched with `jax.vmap` and run on
an accelerator. Actions use OpenSpiel's action ids, so an action chosen
against a JAX state can be applied to the matching pyspiel state.

  from open_spiel.python.jax import games      # this package
  game = games.load_game("tic_tac_toe")
  state = game.new_initial_state()
  while not game.is_terminal(state):
    action = jnp.argmax(game.legal_actions_mask(state))
    state = game.apply_action(state, action)
  print(game.returns(state))                        # f32[num_players]

Key concepts:

- State: a pytree with fixed shapes and dtypes. Most games use a NamedTuple;
  `gin_rummy`, `liars_dice`, `phantom_ttt(_ir)` and `dark_hex(_ir)` use a dict
  with fixed keys. Treat the container as opaque and read a state only
  through the Game methods (`game.is_terminal(state)`,
  `game.current_player(state)`, ...), which work for every game. Every method
  is a pure function of its arguments and is jittable.
- Players: `current_player(state)` is a player id in `[0, num_players)`, or
  `CHANCE_PLAYER_ID` at a chance node. At a terminal state its value is
  game-specific; check `is_terminal` first.
- Chance: at a chance node `chance_outcomes(state)` is a dense probability
  vector indexed by chance action id, and `apply_action` takes that id.
- Observations: `observation_tensor(state, player)` is what `player` sees now;
  `information_state_tensor(state, player)` is everything `player` has seen
  so far. A game implements both explicitly, except a game whose OpenSpiel
  counterpart has no information-state tensor (`gin_rummy`): it sets
  `provides_information_state_tensor = False`, and then both
  `information_state_tensor` and `info_state_shape` raise
  NotImplementedError.
"""

import abc
from typing import Any

import jax
import jax.numpy as jnp

CHANCE_PLAYER_ID = -1
# What some games' current_player returns at a terminal state; the value
# there is game-specific, so callers check is_terminal.
TERMINAL_PLAYER_ID = -4
# Card padding value used by the poker games for cards not yet dealt.
INVALID_CARD = -10000


class Game(abc.ABC):
  """An abstract JAX game.

  Subclasses implement every abstract method as a pure, jittable function of
  its arguments. Games with chance nodes also override `max_chance_outcomes`
  and `chance_outcomes`.

  Attributes:
    provides_information_state_tensor: Whether `information_state_tensor`
      exists (module doc, Observations).
  """

  provides_information_state_tensor: bool = True

  @property
  @abc.abstractmethod
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """

  @property
  @abc.abstractmethod
  def num_players(self) -> int:
    """The number of players."""

  @property
  @abc.abstractmethod
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""

  @abc.abstractmethod
  def new_initial_state(self) -> Any:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """

  @abc.abstractmethod
  def apply_action(self, state: Any, action: jax.Array) -> Any:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """

  @abc.abstractmethod
  def current_player(self, state: Any) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """

  @abc.abstractmethod
  def is_terminal(self, state: Any) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """

  @abc.abstractmethod
  def is_chance_node(self, state: Any) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """

  @abc.abstractmethod
  def legal_actions_mask(self, state: Any) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """

  @abc.abstractmethod
  def observation_tensor(
      self, state: Any, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """

  @abc.abstractmethod
  def information_state_tensor(
      self, state: Any, observing_player: jax.Array
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

  @abc.abstractmethod
  def returns(self, state: Any) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    return self.obs_shape

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return 0

  def chance_outcomes(self, state: Any) -> jax.Array:
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
    del state
    if self.max_chance_outcomes != 0:
      raise NotImplementedError(
          f"{type(self).__name__} declares max_chance_outcomes="
          f"{self.max_chance_outcomes} but does not implement chance_outcomes"
      )
    return jnp.zeros((0,), dtype=jnp.float32)
