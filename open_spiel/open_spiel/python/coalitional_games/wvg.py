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

"""Weighted Voting Games.

A weighted voting game is a game where every player i has a weight w_i, and 
there is a fixed quota q, the characteristic function for coalition c is:

    v(c) = 1 if sum_{i in c} w_i > q,
           0 otherwise.

For more detail, see Chapter 4 of "Computational Aspects of Cooperative
Game Theory" text book by Georgios Chalkiadakis, Edith Elkind, and Michael
Wooldridge.
"""

import numpy as np
from open_spiel.python.coalitional_games import coalitional_game


class WeightedVotingGame(coalitional_game.CoalitionalGame):
  """Weighted Voting Game."""

  def __init__(self, weights: np.ndarray, quota: float):
    super().__init__(num_players=len(weights))
    assert len(weights.shape) == 1
    self._weights = weights
    self._quota = quota

  def coalition_value(self, coalition: np.ndarray) -> float:
    assert len(coalition) == self._num_players
    total_weight = np.inner(coalition, self._weights)
    if total_weight > self._quota:
      return 1.0
    else:
      return 0.0
