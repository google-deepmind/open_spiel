# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical information about some games or some specific settings of games.

TODO(author2): Ideally, this should also be available from C++.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python import policy
import pyspiel


def kuhn_nash_equilibrium(alpha):
  """Returns a Nash Equilibrium in Kuhn parameterized by alpha in [0, 1/3].

  See https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy

  Args:
    alpha: The probability to bet on a Jack for Player 0.

  Raises:
    ValueError: If `alpha` is not within [0, 1/3].
  """
  if not 0 <= alpha <= 1 / 3:
    raise ValueError("alpha ({}) must be in [0, 1/3]".format(alpha))
  bet_probability = {
      # Player 0
      "0": alpha,
      "0pb": 0,
      "1": 0,
      "1pb": 1 / 3 + alpha,
      "2": 3 * alpha,
      "2pb": 1,
      # Player 1
      "0p": 1 / 3,
      "0b": 0,
      "1p": 0,
      "1b": 1 / 3,
      "2p": 1,
      "2b": 1,
  }
  game = pyspiel.load_game("kuhn_poker")
  tabular_policy = policy.TabularPolicy(game)
  for state, p in bet_probability.items():
    tabular_policy.policy_for_key(state)[:] = [1 - p, p]
  return tabular_policy
