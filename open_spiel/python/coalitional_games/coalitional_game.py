# Copyright 2023 DeepMind Technologies Limited
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

"""Coalitional Games in Open Spiel."""

import abc
import numpy as np


class CoalitionalGame(abc.ABC):
  """An abstract class for computing the value of a coalition."""

  def __init__(self, num_players: int):
    self._num_players = num_players

  @abc.abstractmethod
  def coalition_value(self, coalition: np.ndarray) -> float:
    """Returns the value of the coalition (the characteristic function).

    Args:
      coalition: an array of size num_players of ones (indicating player is
        included) and zeroes (the player is excluded).
    """

  def coalition_values(self, coalitions: np.ndarray) -> np.ndarray:
    """Returns the values of a batch of coalitions.

    Override to provide faster versions depending on the game.

    Args:
      coalitions: batch_size by num_players array of coalitions.
    """
    batch_size = coalitions.shape[0]
    return np.asarray(
        [self.coalition_value(coalitions[i]) for i in range(batch_size)]
    )

  def num_players(self) -> int:
    """Returns the number of players."""
    return self._num_players
