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
from open_spiel.python.algorithms import cfr_br
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

_KUHN_GAME = pyspiel.load_game("kuhn_poker")
_LEDUC_GAME = pyspiel.load_game("leduc_poker")

_KUHN_UNIFORM_POLICY = policy.TabularPolicy(_KUHN_GAME)
_LEDUC_UNIFORM_POLICY = policy.TabularPolicy(_LEDUC_GAME)
_EXPECTED_EXPLOITABILITIES_CFRBR_KUHN = [
    0.9166666666666666, 0.33333333333333337, 0.3194444444444445,
    0.2604166666666667, 0.22666666666666674
]
_EXPECTED_EXPLOITABILITIES_CFRBR_LEDUC = [
    4.747222222222222, 4.006867283950617, 3.4090489231017034,
    2.8982539553095172, 2.5367193593344504
]


class CFRBRTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      list(itertools.product([True, False], [True, False])))
  def test_policy_zero_is_uniform(self, linear_averaging, regret_matching_plus):
    game = pyspiel.load_game("leduc_poker")
    cfr_solver = cfr_br.CFRBRSolver(
        game,
        regret_matching_plus=regret_matching_plus,
        linear_averaging=linear_averaging)

    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.current_policy().action_probability_array)
    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.average_policy().action_probability_array)

  def test_policy_and_average_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    cfrbr_solver = cfr_br.CFRBRSolver(game)
    for _ in range(300):
      cfrbr_solver.evaluate_and_update_policy()
    average_policy = cfrbr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

    cfrbr_solver.current_policy()

  @parameterized.parameters([
      (_KUHN_GAME, pyspiel.CFRBRSolver, _EXPECTED_EXPLOITABILITIES_CFRBR_KUHN),
      (_KUHN_GAME, cfr_br.CFRBRSolver, _EXPECTED_EXPLOITABILITIES_CFRBR_KUHN),
      (_LEDUC_GAME, pyspiel.CFRBRSolver,
       _EXPECTED_EXPLOITABILITIES_CFRBR_LEDUC),
      (_LEDUC_GAME, cfr_br.CFRBRSolver, _EXPECTED_EXPLOITABILITIES_CFRBR_LEDUC),
  ])
  def test_cpp_and_python_cfr_br(self, game, solver_cls,
                                 expected_exploitability):
    solver = solver_cls(game)
    for step in range(5):
      solver.evaluate_and_update_policy()

      # We do not compare the policy directly as we do not have an easy way to
      # convert one to the other, so we use the exploitability as a proxy.
      avg_policy = solver.average_policy()
      if solver_cls == pyspiel.CFRBRSolver:
        exploitability_ = pyspiel.nash_conv(game, avg_policy)
      else:
        exploitability_ = exploitability.nash_conv(game, avg_policy)

      self.assertEqual(expected_exploitability[step], exploitability_)


if __name__ == "__main__":
  absltest.main()
