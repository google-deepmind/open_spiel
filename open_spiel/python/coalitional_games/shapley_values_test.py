# Copyright 2023 DeepMind Technologies Limited
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

"""Tests for Shapley value calculations."""


from absl.testing import absltest
import numpy as np
from open_spiel.python.coalitional_games import basic_games
from open_spiel.python.coalitional_games import deon_larson20_games
from open_spiel.python.coalitional_games import shapley_values


SEED = 23856711


class ShapleyValuesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(SEED)

  def test_ice_cream_game(self):
    """Example 2.11 from CACGT book by Chalkiadakis, Elkind, and Wooldridge."""
    game = basic_games.IceCreamGame()
    svals = shapley_values.compute_shapley_values(game)
    self.assertAlmostEqual(svals[0], 250.0)

  def test_ice_cream_game_approximate(self):
    """Monte Carlo sampling version of Shapley value computation."""
    game = basic_games.IceCreamGame()
    svals = shapley_values.compute_approximate_shapley_values(game, 1000)
    self.assertAlmostEqual(svals[0]/1000.0, 0.250, places=2)

  def test_deon_larson20_games(self):
    for name, values in deon_larson20_games.SHAPLEY_VALUES.items():
      values_arr = np.asarray(values)
      game = deon_larson20_games.make_game(name)
      svals = shapley_values.compute_shapley_values(game)
      self.assertTrue(np.allclose(svals, values_arr))


if __name__ == "__main__":
  absltest.main()
