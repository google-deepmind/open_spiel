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

"""Conversion between the JAX games and pyspiel (OpenSpiel's C++ games).

A JAX game and its OpenSpiel counterpart share action ids, but their
observation tensors may use a different layout. This mixin converts a pyspiel
state into the JAX game's observation and information-state tensors, and
maps actions between the two where a game's action ids differ. It is what
the pyspiel equivalence tests use to check a JAX game move for move
against OpenSpiel.

  from open_spiel.python.jax import games  # this package
  game = games.load_game("connect_four")
  os_state = pyspiel.load_game("connect_four").new_initial_state()
  obs = game.observation_from_os(os_state, player=0)  # == game's own encoding
  action = game.action_from_os(os_state, os_state.legal_actions()[0])

Every JAX game inherits `PyspielInterop`. The methods run on the host
and take a `pyspiel.State`, so calling them needs pyspiel installed;
importing a game does not.
"""

import typing
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

if typing.TYPE_CHECKING:
  import pyspiel


class PyspielInterop:
  """Host-side conversions from pyspiel states to JAX game tensors."""

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
    raise NotImplementedError(
        f"{type(self).__name__} does not implement observation_from_os"
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
    return self.observation_from_os(os_state, player)

  def action_from_os(self, os_state: "pyspiel.State", os_action: int) -> int:
    """Returns the JAX game's action id of pyspiel action `os_action`.

    Identity unless a game's action ids differ from OpenSpiel's.

    Args:
      os_state: The pyspiel state the action is taken in.
      os_action: The pyspiel action id.

    Returns:
      The JAX game's action id.
    """
    del os_state
    return os_action

  def action_to_os(self, state: Any, action: int) -> int:
    """Returns the pyspiel action id of the JAX game's `action`.

    The inverse of action_from_os: a game whose OpenSpiel action id depends on
    the position reads it from `state`. Identity unless a game's action ids
    differ from OpenSpiel's.

    Args:
      state: The JAX state the action is taken in.
      action: The JAX game's action id.

    Returns:
      The pyspiel action id.
    """
    del state
    return action

  def legal_actions_mask_from_os(self, os_state: "pyspiel.State") -> jax.Array:
    """Returns the legal actions of `os_state` in the JAX game's action ids.

    Args:
      os_state: The pyspiel state.

    Returns:
      bool[num_actions]: True for each legal action.
    """
    # num_actions comes from the Game this mixin is combined with.
    # pyrefly: ignore[missing-attribute]
    mask = np.zeros(self.num_actions, dtype=bool)
    for os_action in os_state.legal_actions():
      mask[self.action_from_os(os_state, os_action)] = True
    return jnp.array(mask)
