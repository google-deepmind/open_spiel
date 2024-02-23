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

from absl.testing import absltest
import numpy as np
from open_spiel.python.coalitional_games import least_core_lp
from open_spiel.python.coalitional_games import shapley_values
from open_spiel.python.coalitional_games import wvg


SEED = 2093777


class WeightedVotingGamesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(SEED)

  def test_basic_wvg_equal_weights(self):
    # Equal weights.
    game = wvg.WeightedVotingGame(weights=np.asarray([10]*4), quota=35.0)
    svals = shapley_values.compute_shapley_values(game)
    self.assertTrue(np.allclose(svals, np.asarray([0.25, 0.25, 0.25, 0.25])))
    lc_imputation, epsilon = least_core_lp.solve_least_core_lp(
        game, least_core_lp.add_all_constraints)
    self.assertTrue(np.allclose(lc_imputation,
                                np.asarray([0.25, 0.25, 0.25, 0.25])))
    self.assertAlmostEqual(epsilon, 0.0)

  def test_basic_wvg_unequal_weights(self):
    # Example 2.3 of the CACGT book by by Chalkiadakis, Elkind, and Wooldridge.
    game = wvg.WeightedVotingGame(weights=np.asarray([40.0, 22.0, 30.0, 9.0]),
                                  quota=51.0)
    svals = shapley_values.compute_shapley_values(game)
    self.assertTrue(np.allclose(svals, np.asarray([1.0/3, 1.0/3, 1.0/3, 0])))
    lc_imputation, epsilon = least_core_lp.solve_least_core_lp(
        game, least_core_lp.add_all_constraints)
    print(lc_imputation)   # prints [0.33, 0.33, 0.33, 0]
    print(epsilon)         # prints 0.33
    self.assertTrue(np.allclose(lc_imputation,
                                np.asarray([1.0/3, 1.0/3, 1.0/3, 0])))
    self.assertAlmostEqual(epsilon, 1.0/3.0)


if __name__ == "__main__":
  absltest.main()
