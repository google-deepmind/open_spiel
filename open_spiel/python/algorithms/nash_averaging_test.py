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
game1 = pyspiel.create_matrix_game(
    [[0.0, -1.0, -1.0], [1.0, 0.0, -1.0], [1.0, 1.0, 0.0]],
    [[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]])

eq1 = np.asarray([0., 0., 1.])
value1 = np.asarray([-1., -1., 0.])

# rock-paper-scissors test case
game2 = pyspiel.create_matrix_game(
    [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
    [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
eq2 = np.asarray([1/3, 1/3, 1/3])
value2 = np.asarray([0., 0., 0.])


class NashAveragingTest(parameterized.TestCase):
  @parameterized.named_parameters(
      ('transitive_game', game1, eq1, value1),
      ('rps_game', game2, eq2, value2),
  )
  def test_simple_games(self, game, eq, value):

    maxent_nash, nash_avg_value = nash_averaging(game)
    with self.subTest("probability"):
      np.testing.assert_array_almost_equal(eq, maxent_nash.reshape(-1))

    with self.subTest("value"):
      np.testing.assert_array_almost_equal(value, nash_avg_value.reshape(-1))


if __name__ == "__main__":
  absltest.main()
