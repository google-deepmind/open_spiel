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
"""Tests for open_spiel.python.algorithms.nash_averaging."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python.algorithms.nash_averaging import nash_averaging
import pyspiel

# transitive game test case
game_trans = pyspiel.create_matrix_game(
    [[0.0, -1.0, -1.0], [1.0, 0.0, -1.0], [1.0, 1.0, 0.0]],
    [[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]])

eq_trans = np.asarray([0., 0., 1.])
value_trans = np.asarray([-1., -1., 0.])

# rock-paper-scissors test case
game_rps = pyspiel.create_matrix_game(
    [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
    [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
eq_rps = np.asarray([1 / 3, 1 / 3, 1 / 3])
value_rps = np.asarray([0., 0., 0.])

# game with one dominated strategy (AvA case)
p_mat0 = np.asarray([[0.0, 234., 34., -270.], [-234., 0., -38., -464.],
                     [-34., 38., 0., -270.], [270., 464., 270., 0.]])
game0 = pyspiel.create_matrix_game(p_mat0, -p_mat0)
dominated_idxs0 = [0, 1, 2]


# game with one dominated strategy (AvT case)
p_mat1 = np.asarray([
    [0.0, 0.0, 0.0],
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
])
game1 = pyspiel.create_matrix_game(p_mat1, -p_mat1)
dominated_idxs1 = [0, 1, 2]


# game with one multiple dominant strategy (AvT case)
p_mat2 = np.asarray([
    [0.0, 0.0, 0.0],
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [3.0, 30.0, 300.0],
])
game2 = pyspiel.create_matrix_game(p_mat2, -p_mat2)
dom_idxs2 = [3, 4]


class NashAveragingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("transitive_game", game_trans, eq_trans, value_trans),
      ("rps_game", game_rps, eq_rps, value_rps),
  )
  def test_simple_games(self, game, eq, value):

    maxent_nash, nash_avg_value = nash_averaging(game)
    with self.subTest("probability"):
      np.testing.assert_array_almost_equal(eq, maxent_nash.reshape(-1))

    with self.subTest("value"):
      np.testing.assert_array_almost_equal(value, nash_avg_value.reshape(-1))

  @parameterized.named_parameters(
      ("game0", game0, dominated_idxs0),)
  def test_ava_games_with_dominated_strategy(self, game, dominated_idxs):
    maxent_nash, _ = nash_averaging(game)
    with self.subTest("dominated strategies have zero Nash probs"):
      for idx in dominated_idxs:
        self.assertAlmostEqual(maxent_nash[idx].item(), 0.0)

  @parameterized.named_parameters(
      ("game1", game1, dominated_idxs1),
  )
  def test_avt_games_with_dominated_strategy(self, game, dominated_idxs):
    (agent_strategy, _), _ = nash_averaging(game, a_v_a=False)
    with self.subTest("dominated strategies have zero Nash probs"):
      for idx in dominated_idxs:
        self.assertAlmostEqual(agent_strategy[idx].item(), 0.0)

  @parameterized.named_parameters(
      ("game2", game2, dom_idxs2),
  )
  def test_avt_games_with_multiple_dominant_strategies(self, game, dom_idxs):
    (agent_strategy, _), (agent_values, _) = nash_averaging(game, a_v_a=False)
    with self.subTest("dominant strategies have equal Nash probs"):
      for idx in dom_idxs:
        self.assertAlmostEqual(agent_strategy[idx].item(), 1 / len(dom_idxs2))

    with self.subTest("dominant strategies have equal Nash values"):
      values = [agent_values[idx] for idx in dom_idxs]
      self.assertAlmostEqual(np.abs(np.max(values) - np.min(values)), 0.0)


if __name__ == "__main__":
  absltest.main()
