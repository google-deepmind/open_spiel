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

"""Some helpers for normal-form games."""

import collections
import numpy as np


class StrategyAverager(object):
  """A helper class for averaging strategies for players."""

  def __init__(self, num_players, action_space_shapes, window_size=None):
    """Initialize the average strategy helper object.

    Args:
      num_players (int): the number of players in the game,
      action_space_shapes:  an vector of n integers, where each element
          represents the size of player i's actions space,
      window_size (int or None): if None, computes the players' average
          strategies over the entire sequence, otherwise computes the average
          strategy over a finite-sized window of the k last entries.
    """
    self._num_players = num_players
    self._action_space_shapes = action_space_shapes
    self._window_size = window_size
    self._num = 0
    if self._window_size is None:
      self._sum_meta_strategies = [
          np.zeros(action_space_shapes[p]) for p in range(num_players)
      ]
    else:
      self._window = collections.deque(maxlen=self._window_size)

  def append(self, meta_strategies):
    """Append the meta-strategies to the averaged sequence.

    Args:
      meta_strategies: a list of strategies, one per player.
    """
    if self._window_size is None:
      for p in range(self._num_players):
        self._sum_meta_strategies[p] += meta_strategies[p]
    else:
      self._window.append(meta_strategies)
    self._num += 1

  def average_strategies(self):
    """Return each player's average strategy.

    Returns:
      The averaged strategies, as a list containing one strategy per player.
    """

    if self._window_size is None:
      avg_meta_strategies = [
          np.copy(x) for x in self._sum_meta_strategies
      ]
      num_strategies = self._num
    else:
      avg_meta_strategies = [
          np.zeros(self._action_space_shapes[p])
          for p in range(self._num_players)
      ]
      for i in range(len(self._window)):
        for p in range(self._num_players):
          avg_meta_strategies[p] += self._window[i][p]
      num_strategies = len(self._window)
    for p in range(self._num_players):
      avg_meta_strategies[p] /= num_strategies
    return avg_meta_strategies
