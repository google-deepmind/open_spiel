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

"""Tests for greedy_policy."""

from absl.testing import absltest

from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel


class GreedyPolicyTest(absltest.TestCase):

  def test_greedy_python(self):
    """Check if the greedy policy works as expected.

    The test checks that a greedy policy with respect to an optimal value is
    an optimal policy.
    """
    game = crowd_modelling.MFGCrowdModellingGame()
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    br_value = best_response_value.BestResponse(game, dist)
    br_val = br_value(game.new_initial_state())

    greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    greedy_pi = greedy_pi.to_tabular()
    pybr_value = policy_value.PolicyValue(game, dist, greedy_pi)
    pybr_val = pybr_value(game.new_initial_state())
    self.assertAlmostEqual(br_val, pybr_val)

  def test_greedy_cpp(self):
    """Check if the greedy policy works as expected.

    The test checks that a greedy policy with respect to an optimal value is
    an optimal policy.
    """
    game = pyspiel.load_game("mfg_crowd_modelling")
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    br_value = best_response_value.BestResponse(game, dist)
    br_val = br_value(game.new_initial_state())

    greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    greedy_pi = greedy_pi.to_tabular()
    pybr_value = policy_value.PolicyValue(game, dist, greedy_pi)
    pybr_val = pybr_value(game.new_initial_state())
    self.assertAlmostEqual(br_val, pybr_val)


if __name__ == "__main__":
  absltest.main()
