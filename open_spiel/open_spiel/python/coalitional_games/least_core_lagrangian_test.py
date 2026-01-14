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

"""Tests for least-core lagrangian calculations."""

from absl.testing import absltest
from ml_collections import config_dict as configdict
import numpy as np

from open_spiel.python.coalitional_games import basic_games
from open_spiel.python.coalitional_games import least_core_lagrangian


SEED = 817346817


def get_alg_config():
  """Get configuration for botched trades experiment."""
  alg_config = configdict.ConfigDict()

  alg_config.init = configdict.ConfigDict()
  alg_config.init.lr_primal = 1e-2
  alg_config.init.lr_dual = 1e-2

  alg_config.solve = configdict.ConfigDict()
  alg_config.solve.batch_size = 2**3
  alg_config.solve.mu_init = 1000
  alg_config.solve.gamma = 1e-8
  alg_config.solve.n_iter = 110_000
  alg_config.solve.seed = 0
  alg_config.solve.save_every = 10_000

  alg_config.eval = configdict.ConfigDict()
  alg_config.eval.evaluation_iterations = 2**3

  return alg_config


class LeastCoreLagrangianTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(SEED)
    self.config = get_alg_config()

  def test_ice_cream_example_full_lagrangian(self):
    """Solve the least core Lagrangian."""
    game = basic_games.IceCreamGame()
    least_core_value = least_core_lagrangian.compute_least_core_value(
        game, self.config)
    imputation = least_core_value.payoff
    epsilon = least_core_value.lcv
    self.assertAlmostEqual(imputation.sum(), 1000.0, places=3)
    self.assertGreater(imputation.all(), -1e-10)
    self.assertLess(epsilon, 1e-6)


if __name__ == "__main__":
  absltest.main()
