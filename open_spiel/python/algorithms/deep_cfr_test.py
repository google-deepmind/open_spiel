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

"""Tests for open_spiel.python.algorithms.deep_cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()


class DeepCFRTest(parameterized.TestCase):

  @parameterized.parameters('leduc_poker', 'kuhn_poker', 'liars_dice')
  def test_deep_cfr_runs(self, game_name):
    game = pyspiel.load_game(game_name)
    with tf.Session() as sess:
      deep_cfr_solver = deep_cfr.DeepCFRSolver(
          sess,
          game,
          policy_network_layers=(8, 4),
          advantage_network_layers=(4, 2),
          num_iterations=2,
          num_traversals=2,
          learning_rate=1e-3,
          batch_size_advantage=None,
          batch_size_strategy=None,
          memory_capacity=1e7)
      sess.run(tf.global_variables_initializer())
      deep_cfr_solver.solve()

  def test_matching_pennies_3p(self):
    # We don't expect Deep CFR to necessarily converge on 3-player games but
    # it's nonetheless interesting to see this result.
    game = pyspiel.load_game_as_turn_based('matching_pennies_3p')
    with tf.Session() as sess:
      deep_cfr_solver = deep_cfr.DeepCFRSolver(
          sess,
          game,
          policy_network_layers=(16, 8),
          advantage_network_layers=(32, 16),
          num_iterations=2,
          num_traversals=2,
          learning_rate=1e-3,
          batch_size_advantage=None,
          batch_size_strategy=None,
          memory_capacity=1e7)
      sess.run(tf.global_variables_initializer())
      deep_cfr_solver.solve()
      conv = exploitability.nash_conv(
          game,
          policy.tabular_policy_from_callable(
              game, deep_cfr_solver.action_probabilities))
      print('Deep CFR in Matching Pennies 3p. NashConv: {}'.format(conv))


if __name__ == '__main__':
  tf.test.main()
