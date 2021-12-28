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

"""Tests for open_spiel.python.pytorch.deep_cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import policy
import pyspiel
from open_spiel.python.pytorch import deep_cfr


class DeepCFRPyTorchTest(parameterized.TestCase):

  @parameterized.parameters('leduc_poker', 'kuhn_poker', 'liars_dice')
  def test_deep_cfr_runs(self, game_name):
    game = pyspiel.load_game(game_name)
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(8, 4),
        advantage_network_layers=(4, 2),
        num_iterations=2,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7)
    deep_cfr_solver.solve()

  def test_matching_pennies_3p(self):
    game = pyspiel.load_game_as_turn_based('matching_pennies_3p')
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(16, 8),
        advantage_network_layers=(32, 16),
        num_iterations=2,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7)
    deep_cfr_solver.solve()
    conv = pyspiel.nash_conv(
        game,
        policy.python_policy_to_pyspiel_policy(
            policy.tabular_policy_from_callable(
                game, deep_cfr_solver.action_probabilities)))
    logging.info('Deep CFR in Matching Pennies 3p. NashConv: %.2f', conv)


def main(_):
  absltest.main()


if __name__ == '__main__':
  # Necessary to run main via app.run for internal tests.
  app.run(main)
