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
"""Fixed Point."""

from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
import pyspiel


class FixedPoint(object):
  """The fixed point algorithm.

  This algorithm is based on Banach-Picard iterations for the fixed point
  operator characterizing the Nash equilibrium. At each iteration, the policy is
  updated by computing a best response against the current mean-field, and the
  mean-field is updated by taking the mean-field induced by the current policy.
  """

  def __init__(self, game: pyspiel.Game):
    """Initializes the algorithm.

    Args:
      game: The game to analyze.
    """
    self._game = game
    self._policy = policy_lib.UniformRandomPolicy(self._game)
    self._distribution = distribution.DistributionPolicy(game, self._policy)

  def iteration(self):
    """An itertion of Fixed Point."""
    # Calculate the current distribution and the best response.
    distrib = distribution.DistributionPolicy(self._game, self._policy)
    br_value = best_response_value.BestResponse(
        self._game, distrib, value.TabularValueFunction(self._game))

    # Policy is greedy with respect to the best response.
    self._policy = greedy_policy.GreedyPolicy(
        self._game, list(range(self._game.num_players())), br_value)
    self._distribution = distribution.DistributionPolicy(
        self._game, self._policy)

  def get_policy(self) -> policy_lib.Policy:
    return self._policy

  @property
  def distribution(self) -> distribution.DistributionPolicy:
    return self._distribution
