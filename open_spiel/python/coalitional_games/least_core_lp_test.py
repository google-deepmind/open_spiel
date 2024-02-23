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

"""Tests for least-core LP calculations."""


from absl.testing import absltest
import numpy as np
from open_spiel.python.coalitional_games import basic_games
from open_spiel.python.coalitional_games import least_core_lp


SEED = 817346817


class LeastCoreLPTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(SEED)

  def test_ice_cream_example_full_lp(self):
    """Solve the full LP."""
    game = basic_games.IceCreamGame()
    imputation, epsilon = least_core_lp.solve_least_core_lp(
        game, least_core_lp.add_all_constraints)
    self.assertAlmostEqual(imputation.sum(), 1000.0)
    self.assertGreater(imputation.all(), 0.0)
    self.assertLess(epsilon, 1e-6)

  def test_ice_cream_example_uniform_sample_lp(self):
    """Solve the LP with 20 uniformly sampled constraints."""
    game = basic_games.IceCreamGame()
    cons_func = least_core_lp.make_uniform_sampling_constraints_function(20)
    imputation, epsilon = least_core_lp.solve_least_core_lp(game, cons_func)
    self.assertAlmostEqual(imputation.sum(), 1000.0)
    self.assertGreater(imputation.all(), 0.0)
    self.assertLess(epsilon, 1e-6)


if __name__ == "__main__":
  absltest.main()
