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
"""Tests for open_spiel.python.algorithms.regret_matching."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import regret_matching
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel


class RegretMatchingTest(absltest.TestCase):

  def test_two_players(self):
    test_a = np.array([[2, 1, 0], [0, -1, -2]])
    test_b = np.array([[2, 1, 0], [0, -1, -2]])

    strategies = regret_matching.regret_matching(
        [test_a, test_b],
        initial_strategies=None,
        iterations=50000,
        prd_gamma=1e-8,
        average_over_last_n_strategies=10)

    self.assertLen(strategies, 2, "Wrong strategy length.")
    self.assertGreater(strategies[0][0], 0.999,
                       "Regret matching failed in trivial case.")

  def test_three_players(self):
    test_a = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    test_b = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    test_c = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])

    strategies = regret_matching.regret_matching(
        [test_a, test_b, test_c],
        initial_strategies=None,
        iterations=50000,
        gamma=1e-6,
        average_over_last_n_strategies=10)
    self.assertLen(strategies, 3, "Wrong strategy length.")
    self.assertGreater(strategies[0][0], 0.999,
                       "Regret matching failed in trivial case.")

  def test_rps(self):
    game = pyspiel.load_game("matrix_rps")
    payoffs_array = game_payoffs_array(game)
    strategies = regret_matching.regret_matching(
        [payoffs_array[0], payoffs_array[1]],
        initial_strategies=[
            np.array([0.1, 0.4, 0.5]),
            np.array([0.9, 0.1, 0.01])
        ],
        iterations=50000,
        gamma=1e-6)
    self.assertLen(strategies, 2, "Wrong strategy length.")
    # places=1 corresponds to an absolute difference of < 0.001
    self.assertAlmostEqual(strategies[0][0], 1 / 3., places=2)
    self.assertAlmostEqual(strategies[0][1], 1 / 3., places=2)
    self.assertAlmostEqual(strategies[0][2], 1 / 3., places=2)

  def test_biased_rps(self):
    game = pyspiel.load_game("matrix_brps")
    payoffs_array = game_payoffs_array(game)
    strategies = regret_matching.regret_matching(
        [payoffs_array[0], payoffs_array[1]], iterations=50000, gamma=1e-8)
    self.assertLen(strategies, 2, "Wrong strategy length.")
    # places=1 corresponds to an absolute difference of < 0.01
    self.assertAlmostEqual(strategies[0][0], 1 / 16., places=1)
    self.assertAlmostEqual(strategies[0][1], 10 / 16., places=1)
    self.assertAlmostEqual(strategies[0][2], 5 / 16., places=1)


if __name__ == "__main__":
  absltest.main()
