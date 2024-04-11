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
"""Tests for open_spiel.python.algorithms.stackelberg_lp."""

from absl.testing import absltest
from absl.testing import parameterized
import nashpy as nash
import numpy as np

from open_spiel.python.algorithms.stackelberg_lp import solve_stackelberg
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

# Numerical tolerance for tests.
EPS = 1e-6

# game instances based on Conitzer & Sandholm'06 paper
game0 = pyspiel.create_matrix_game([[2, 4], [1, 3]], [[1, 0], [0, 1]])
commit_strategy0 = np.array([0.5, 0.5])
commit_value0 = 3.5

game1 = pyspiel.create_matrix_game([[2, 0, 0], [1, 0, 0]],
                                   [[0, 2, 5], [0, -1, -4]])
commit_strategy1 = np.array([1 / 3, 2 / 3])
commit_value1 = 4 / 3

# a game with dominated strategy
game2 = pyspiel.create_matrix_game([[3, 9], [9, 1]], [[0, 0], [1, 8]])
commit_strategy2 = np.array([1.0, 0.0])
commit_value2 = 9.0


class StackelbergLPTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("game0", game0, commit_strategy0, commit_value0),
      ("game1", game1, commit_strategy1, commit_value1),
      ("game2", game2, commit_strategy2, commit_value2),
  )
  def test_simple_games(self, game, commit_strategy, commit_value):
    leader_eq_strategy, _, leader_eq_value, _ = solve_stackelberg(game)

    with self.subTest("optimal commitment"):
      np.testing.assert_array_almost_equal(commit_strategy, leader_eq_strategy)
      self.assertAlmostEqual(commit_value, leader_eq_value)

    with self.subTest("Leader-payoff in SSE no less than in NE"):
      p_mat = game_payoffs_array(game)
      nashpy_game = nash.Game(p_mat[0], p_mat[1])
      for eq in nashpy_game.support_enumeration():
        leader_nash_value = eq[0].reshape(1,
                                          -1).dot(p_mat[0]).dot(eq[1].reshape(
                                              -1, 1))
        self.assertGreaterEqual(leader_eq_value - leader_nash_value, -EPS)


if __name__ == "__main__":
  absltest.main()
