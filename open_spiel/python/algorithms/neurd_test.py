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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import neurd
import pyspiel

# Temporarily disable TF2 behavior while the code is not updated.
tf.disable_v2_behavior()

tf.enable_eager_execution()

_GAME = pyspiel.load_game('kuhn_poker')


def _new_model():
  return neurd.DeepNeurdModel(
      _GAME,
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=1,
      use_skip_connections=True,
      autoencode=True)


class NeurdTest(tf.test.TestCase):

  def setUp(self):
    super(NeurdTest, self).setUp()
    tf.set_random_seed(42)

  def test_neurd(self):
    num_iterations = 2
    models = [_new_model() for _ in range(_GAME.num_players())]

    solver = neurd.CounterfactualNeurdSolver(_GAME, models)

    average_policy = solver.average_policy()
    self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

    @tf.function
    def _train(model, data):
      neurd.train(
          model=model,
          data=data,
          batch_size=12,
          step_size=10.0,
          autoencoder_loss=tf.losses.huber_loss)

    for _ in range(num_iterations):
      solver.evaluate_and_update_policy(_train)

    average_policy = solver.average_policy()
    self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)


if __name__ == '__main__':
  absltest.main()
