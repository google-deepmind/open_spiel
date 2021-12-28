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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.double_oracle."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import double_oracle
import pyspiel


class DoubleOracleTest(absltest.TestCase):

  def test_rock_paper_scissors(self):
    game = pyspiel.load_matrix_game("matrix_rps")
    solver = double_oracle.DoubleOracleSolver(game)
    solution, iteration, value = solver.solve(initial_strategies=[[0], [0]])
    np.testing.assert_allclose(solution[0], np.ones(3)/3.)
    np.testing.assert_allclose(solution[1], np.ones(3)/3.)
    self.assertEqual(iteration, 3)
    self.assertAlmostEqual(value, 0.0)

  def test_single_step(self):
    game = pyspiel.load_matrix_game("matrix_rps")
    solver = double_oracle.DoubleOracleSolver(game)
    solver.subgame_strategies = [[0], [0]]
    best_response, best_response_utility = solver.step()
    self.assertListEqual(best_response, [1, 1])
    self.assertListEqual(best_response_utility, [1.0, 1.0])

  def test_kuhn_poker(self):
    game = pyspiel.extensive_to_matrix_game(pyspiel.load_game("kuhn_poker"))
    solver = double_oracle.DoubleOracleSolver(game)
    solution, iteration, value = solver.solve(initial_strategies=[[0], [0]])

    # check if solution is Nash
    exp_utilty = solution[0] @ solver.payoffs @ solution[1]
    self.assertAlmostEqual(max(solver.payoffs[0] @ solution[1]), exp_utilty[0])
    self.assertAlmostEqual(max(solution[0] @ solver.payoffs[1]), exp_utilty[1])

    self.assertEqual(iteration, 8)
    self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
  absltest.main()
