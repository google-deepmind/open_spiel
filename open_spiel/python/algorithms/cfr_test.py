# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for open_spiel.python.algorithms.cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

_KUHN_GAME = pyspiel.load_game("kuhn_poker")
_LEDUC_GAME = pyspiel.load_game("leduc_poker")

_KUHN_UNIFORM_POLICY = policy.TabularPolicy(_KUHN_GAME)
_LEDUC_UNIFORM_POLICY = policy.TabularPolicy(_LEDUC_GAME)


class ModuleLevelFunctionTest(absltest.TestCase):

  def test__update_current_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)

    cumulative_regrets = np.arange(0, 12 * 2).reshape((12, 2))
    expected_policy = cumulative_regrets / np.sum(
        cumulative_regrets, axis=-1, keepdims=True)
    nodes_indices = {
        u"0": 0,
        u"0pb": 1,
        u"1": 2,
        u"1pb": 3,
        u"2": 4,
        u"2pb": 5,
        u"1p": 6,
        u"1b": 7,
        u"2p": 8,
        u"2b": 9,
        u"0p": 10,
        u"0b": 11,
    }
    # pylint: disable=g-complex-comprehension
    info_state_nodes = {
        key: cfr._InfoStateNode(
            legal_actions=[0, 1],
            index_in_tabular_policy=None,
            cumulative_regret=dict(enumerate(cumulative_regrets[index])),
            cumulative_policy=None) for key, index in nodes_indices.items()
    }
    # pylint: enable=g-complex-comprehension

    cfr._update_current_policy(tabular_policy, info_state_nodes)

    np.testing.assert_array_equal(expected_policy,
                                  tabular_policy.action_probability_array)


class CFRTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      list(itertools.product([True, False], [True, False], [True, False])))
  def test_policy_zero_is_uniform(self, linear_averaging, regret_matching_plus,
                                  alternating_updates):
    # We use Leduc and not Kuhn, because Leduc has illegal actions and Kuhn does
    # not.
    game = pyspiel.load_game("leduc_poker")
    cfr_solver = cfr._CFRSolver(
        game,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=linear_averaging,
        alternating_updates=alternating_updates)

    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.current_policy().action_probability_array)
    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.average_policy().action_probability_array)

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

  def test_cfr_plus_solver_best_response_mdp(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = cfr.CFRPlusSolver(game)
    for _ in range(200):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    pyspiel_avg_policy = policy.python_policy_to_pyspiel_policy(average_policy)
    br_computer = pyspiel.TabularBestResponseMDP(game, pyspiel_avg_policy)
    br_info = br_computer.exploitability()
    self.assertLessEqual(br_info.exploitability, 0.001)

  def test_cfr_cce_ce_dist_goofspiel(self):
    """Copy of the TestCCEDistCFRGoofSpiel in corr_dist_test.cc."""
    game = pyspiel.load_game(
        "turn_based_simultaneous_game(game=goofspiel(num_cards=3,points_order="
        "descending,returns_type=total_points))")
    for num_iterations in [1, 10, 100]:
      policies = []
      cfr_solver = cfr.CFRSolver(game)
      for _ in range(num_iterations):
        cfr_solver.evaluate_and_update_policy()
        policies.append(
            policy.python_policy_to_pyspiel_policy(cfr_solver.current_policy()))
      mu = pyspiel.uniform_correlation_device(policies)
      cce_dist_info = pyspiel.cce_dist(game, mu)
      print("goofspiel, cce test num_iters: {}, cce_dist: {}, per player: {}"
            .format(num_iterations, cce_dist_info.dist_value,
                    cce_dist_info.deviation_incentives))
      # Try converting one of the BR policies:
      _ = policy.pyspiel_policy_to_python_policy(
          game, cce_dist_info.best_response_policies[0])

      # Assemble the same correlation device manually, just as an example for
      # how to do non-uniform distributions of them and to test the python
      # bindings for lists of tuples works properly
      uniform_prob = 1.0 / len(policies)
      mu2 = [(uniform_prob, policy) for policy in policies]
      cce_dist_info2 = pyspiel.cce_dist(game, mu2)
      self.assertAlmostEqual(cce_dist_info2.dist_value,
                             sum(cce_dist_info.deviation_incentives))
      # Test the CEDist function too, why not. Disable the exact one, as it
      # takes too long for a test.
      # ce_dist_info = pyspiel.ce_dist(game, pyspiel.determinize_corr_dev(mu))
      ce_dist_info = pyspiel.ce_dist(
          game, pyspiel.sampled_determinize_corr_dev(mu, 100))
      print("goofspiel, ce test num_iters: {}, ce_dist: {}, per player: {}"
            .format(num_iterations, ce_dist_info.dist_value,
                    ce_dist_info.deviation_incentives))
      print("number of conditional best responses per player:")
      for p in range(game.num_players()):
        print("  player {}, num: {}".format(
            p, len(ce_dist_info.conditional_best_response_policies[p])))

  @parameterized.parameters(
      list(itertools.product([True, False], [True, False], [True, False])))
  def test_cfr_kuhn_poker_runs_with_multiple_players(self, linear_averaging,
                                                     regret_matching_plus,
                                                     alternating_updates):
    num_players = 3

    game = pyspiel.load_game("kuhn_poker", {"players": num_players})
    cfr_solver = cfr._CFRSolver(
        game,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=linear_averaging,
        alternating_updates=alternating_updates)
    for _ in range(10):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * num_players)
    del average_policy_values

  @parameterized.parameters(list(itertools.product([False, True])))
  def test_simultaneous_two_step_avg_1b_seq_in_kuhn_poker(
      self, regret_matching_plus):
    num_players = 2
    game = pyspiel.load_game("kuhn_poker", {"players": num_players})
    cfr_solver = cfr._CFRSolver(
        game,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=False,
        alternating_updates=False)

    def check_avg_policy_is_uniform_random():
      avg_policy = cfr_solver.average_policy()
      for player_info_states in avg_policy.states_per_player:
        for info_state in player_info_states:
          state_policy = avg_policy.policy_for_key(info_state)
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

    tabular_policy = solver.current_policy()
    self.assertLen(tabular_policy.state_lookup, 12)
    for info_state_str in tabular_policy.state_lookup.keys():
      np.testing.assert_equal(
          np.asarray([0.5, 0.5]), tabular_policy.policy_for_key(info_state_str))

  @parameterized.parameters([
      (pyspiel.load_game("kuhn_poker"), pyspiel.CFRSolver, cfr.CFRSolver),
      (pyspiel.load_game("leduc_poker"), pyspiel.CFRSolver, cfr.CFRSolver),
      (pyspiel.load_game("kuhn_poker"), pyspiel.CFRPlusSolver,
       cfr.CFRPlusSolver),
      (pyspiel.load_game("leduc_poker"), pyspiel.CFRPlusSolver,
       cfr.CFRPlusSolver),
  ])
  def test_cpp_algorithms_identical_to_python_algorithm(self, game, cpp_class,
                                                        python_class):
    cpp_solver = cpp_class(game)
    python_solver = python_class(game)

    for _ in range(5):
      cpp_solver.evaluate_and_update_policy()
      python_solver.evaluate_and_update_policy()

      cpp_avg_policy = cpp_solver.average_policy()
      python_avg_policy = python_solver.average_policy()

      # We do not compare the policy directly as we do not have an easy way to
      # convert one to the other, so we use the exploitability as a proxy.
      cpp_expl = pyspiel.nash_conv(game, cpp_avg_policy)
      python_expl = exploitability.nash_conv(game, python_avg_policy)
      self.assertEqual(cpp_expl, python_expl)
    # Then we also check the CurrentPolicy, just to check it is giving the same
    # results too
    cpp_current_policy = cpp_solver.current_policy()
    python_current_policy = python_solver.current_policy()
    cpp_expl = pyspiel.nash_conv(game, cpp_current_policy)
    python_expl = exploitability.nash_conv(game, python_current_policy)
    self.assertEqual(cpp_expl, python_expl)


if __name__ == "__main__":
  absltest.main()
