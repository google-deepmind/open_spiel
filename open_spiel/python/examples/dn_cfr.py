# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six

import tensorflow as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import dn_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import os
from datetime import datetime
#tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 1500, "Number of traversals/games")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  pathname = os.path.join('output', datetime.utcnow().strftime('%Y_%m_%d_%I_%M_%S'))
  os.makedirs(pathname)
  deep_cfr_solver = dn_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(64,64,64,64),
      advantage_network_layers=(64,64,64,64),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=2048,
      batch_size_strategy=2048,
      memory_capacity=1e6,
      policy_network_train_steps=500,
      advantage_network_train_steps=500,
      reinitialize_advantage_networks=True,
      save_strategy_memories=pathname,
      infer_device='cpu',
      train_device='gpu')
  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in six.iteritems(advantage_losses):
    logging.info("Advantage for player %d: %s", player,
                  losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                  len(deep_cfr_solver.advantage_buffer))
  logging.info("Strategy Buffer Size: '%s'",
                len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)

  conv = exploitability.nash_conv(game, average_policy)
  logging.info("Deep CFR in '%s' - NashConv: %s", FLAGS.game_name, conv)

  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  print("Computed player 0 value: {}".format(average_policy_values[0]))
  print("Computed player 1 value: {}".format(average_policy_values[1]))


if __name__ == "__main__":
  app.run(main)
