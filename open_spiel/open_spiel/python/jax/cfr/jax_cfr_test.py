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

"""Tests for open_spiel.python.algorithms.jax.jax_cfr.

All of them are taken from open_spiel.python.algorithms.cfr_test.py
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
import pyspiel


class CFRTest(parameterized.TestCase, absltest.TestCase):

  def test_cfr_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = JaxCFR(game)
    cfr_solver.multiple_steps(300)
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2
    )
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3
    )

  def test_cfr_plus_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = JaxCFR(game)
    cfr_solver.multiple_steps(200)
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2
    )
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3
    )

  def test_cfr_plus_solver_best_response_mdp(self):
    game = pyspiel.load_game("kuhn_poker")
    cfr_solver = JaxCFR(game, True, True, True)
    cfr_solver.multiple_steps(200)
    average_policy = cfr_solver.average_policy()
    pyspiel_avg_policy = policy.python_policy_to_pyspiel_policy(average_policy)
    br_computer = pyspiel.TabularBestResponseMDP(game, pyspiel_avg_policy)
    br_info = br_computer.exploitability()
    self.assertLessEqual(br_info.exploitability, 0.001)

  @parameterized.parameters(
      list(itertools.product([True, False], [True, False], [True, False]))
  )
  def test_cfr_kuhn_poker_runs_with_multiple_players(
      self, linear_averaging, alternating_updates, regret_matching_plus
  ):
    num_players = 3

    game = pyspiel.load_game("kuhn_poker", {"players": num_players})
    cfr_solver = JaxCFR(
        game,
        regret_matching_plus=regret_matching_plus,
        alternating_updates=alternating_updates,
        linear_averaging=linear_averaging,
    )
    # for _ in range(10):
    cfr_solver.multiple_steps(10)
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * num_players
    )
    del average_policy_values


if __name__ == "__main__":
  absltest.main()
