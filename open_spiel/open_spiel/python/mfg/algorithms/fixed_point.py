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

from typing import Optional

from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import softmax_policy
import pyspiel


class FixedPoint(object):
  """The fixed point algorithm.

  This algorithm is based on Banach-Picard iterations for the fixed point
  operator characterizing the Nash equilibrium. At each iteration, the policy is
  updated by computing a best response against the current mean-field or a
  regularized version that is obtained by taking a softmax with respect to the
  optimal Q-function, and the mean-field is updated by taking the mean-field
  induced by the current policy.
  """

  def __init__(self, game: pyspiel.Game, temperature: Optional[float] = None):
    """Initializes the algorithm.

    Args:
      game: The game to analyze.
      temperature: If set, then instead of the greedy policy a softmax policy
        with the specified temperature will be used to update the policy at each
        iteration.
    """
    self._game = game
    self._temperature = temperature
    self._policy = policy_lib.UniformRandomPolicy(self._game)
    self._distribution = distribution.DistributionPolicy(game, self._policy)

  def iteration(self):
    """An itertion of Fixed Point."""
    # Calculate the current distribution and the best response.
    distrib = distribution.DistributionPolicy(self._game, self._policy)
    br_value = best_response_value.BestResponse(
        self._game, distrib, value.TabularValueFunction(self._game))

    # Policy is either greedy or softmax with respect to the best response if
    # temperature is specified.
    player_ids = list(range(self._game.num_players()))
    if self._temperature is None:
      self._policy = greedy_policy.GreedyPolicy(self._game, player_ids,
                                                br_value)
    else:
      self._policy = softmax_policy.SoftmaxPolicy(self._game, player_ids,
                                                  self._temperature, br_value)

    self._distribution = distribution.DistributionPolicy(
        self._game, self._policy)

  def get_policy(self) -> policy_lib.Policy:
    return self._policy

  @property
  def distribution(self) -> distribution.DistributionPolicy:
    return self._distribution
