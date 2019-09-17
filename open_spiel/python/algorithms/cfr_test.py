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

"""Tests for open_spiel.python.algorithms.cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel


class CFRTest(parameterized.TestCase, absltest.TestCase):

  def test_cfr_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(300):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

  def test_cfr_plus_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = cfr.CFRPlusSolver(game)
    for _ in range(200):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

  @parameterized.parameters(
      list(
          itertools.product([True, False], [True, False], [True, False],
                            [True, False])))
  def test_cfr_kuhn_poker_runs_with_multiple_players(
      self, initialize_cumulative_values, linear_averaging,
      regret_matching_plus, alternating_updates):
    num_players = 3

    game = pyspiel.load_game("kuhn_poker",
                             {"players": pyspiel.GameParameter(num_players)})
    cfr_solver = cfr._CFRSolver(
        game,
        initialize_cumulative_values=initialize_cumulative_values,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=linear_averaging,
        alternating_updates=alternating_updates)
    for _ in range(10):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * num_players)
    del average_policy_values

  @parameterized.parameters(
      list(itertools.product([False, True], [False, True])))
  def test_simultaneous_two_step_avg_1b_seq_in_kuhn_poker(
      self, regret_matching_plus, initialize_cumulative_values):
    num_players = 2
    game = pyspiel.load_game("kuhn_poker",
                             {"players": pyspiel.GameParameter(num_players)})
    cfr_solver = cfr._CFRSolver(
        game,
        initialize_cumulative_values=initialize_cumulative_values,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=False,
        alternating_updates=False)

    def check_avg_policy_is_uniform_random():
      policy = cfr_solver.average_policy()
      for player_info_states in policy.states_per_player:
        for info_state in player_info_states:
          state_policy = policy.policy_for_key(info_state)
          np.testing.assert_allclose(state_policy, [1.0 / len(state_policy)] *
                                     len(state_policy))

    check_avg_policy_is_uniform_random()

    cfr_solver.evaluate_and_update_policy()
    check_avg_policy_is_uniform_random()

    cfr_solver.evaluate_and_update_policy()

    # The acting player in 1b is player 1 and they have not acted before, so
    # the probability this player plays to this information state is 1, and
    # the sequence probability of any action is just the probability of that
    # action given the information state. On the first iteration, this
    # probability is 0.5 for both actions. On the second iteration, the
    # current policy is [0, 1], so the average cumulants should be
    # [0.5, 1.5]. Normalizing this gives the average policy.
    normalization = 0.5 + 0.5 + 1
    np.testing.assert_allclose(cfr_solver.average_policy().policy_for_key("1b"),
                               [0.5 / normalization, (0.5 + 1) / normalization])

  def test_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    solver = cfr.CFRPlusSolver(game)

    tabular_policy = solver.policy()
    self.assertLen(tabular_policy.state_lookup, 12)
    for info_state_str in tabular_policy.state_lookup.keys():
      np.testing.assert_equal(
          np.asarray([0.5, 0.5]), tabular_policy.policy_for_key(info_state_str))


class CFRBRTest(absltest.TestCase):

  def test_policy_and_average_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    cfrbr_solver = cfr.CFRBRSolver(game)
    for _ in range(300):
      cfrbr_solver.evaluate_and_update_policy()
    average_policy = cfrbr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

    cfrbr_solver.policy()


if __name__ == "__main__":
  absltest.main()
