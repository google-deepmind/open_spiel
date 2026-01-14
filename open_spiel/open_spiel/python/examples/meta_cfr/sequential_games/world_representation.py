# Copyright 2022 DeepMind Technologies Limited
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

"""API for world state representation."""

import abc
from typing import Any, List, Text, Tuple


class WorldState(abc.ABC):
  """Base class for world state representation.

    We can implement this class for world state representations in both
    sequential and matrix games.

    Attributes:
      chance_policy: Policy of the chance node in the game tree.
  """

  def __init__(self):
    self.chance_policy = {0: 1.0}
    self._history = []

  @abc.abstractmethod
  def get_distinct_actions(self) -> List[int]:
    """Returns all possible distinct actions in the game."""
    pass

  @abc.abstractmethod
  def is_terminal(self) -> bool:
    """Returns if the current state of the game is a terminal or not."""
    pass

  @abc.abstractmethod
  def get_actions(self) -> List[Any]:
    """Returns the list of legal actions from the current state of the game."""
    pass

  @abc.abstractmethod
  def get_infostate_string(self, player: int) -> Text:
    """Returns the string form of infostate representation of a given player.

    Args:
      player: Index of player.

    Returns:
      The string representation of the infostate of player.
    """

    pass

  @abc.abstractmethod
  def apply_actions(self, actions: Tuple[int, int, int]) -> None:
    """Applies the current player's action to change state of the world.

    At each timestep of the game, the state of the world is changing by the
    current player's action. At the same time, we should update self._history
    with actions, by appending actions to self._history.

    Args:
      actions: List of actions for chance node, player 1 and player 2.

    """
    pass

  @abc.abstractmethod
  def get_utility(self, player: int) -> float:
    """Returns player's utility when the game reaches to a terminal state.

    Args:
      player: Index of player.

    Returns:
      Utility that player receives when we reach a terminal state in the game.
    """
    pass


