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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr_tf2
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 150, "Number of traversals/games")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
      game,
      policy_network_layers=(64, 64, 64, 64),
      advantage_network_layers=(64, 64, 64, 64),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=2048,
      batch_size_strategy=2048,
      memory_capacity=1e6,
      policy_network_train_steps=5000,
      advantage_network_train_steps=500,
      reinitialize_advantage_networks=True,
      infer_device="cpu",
      train_device="cpu")
  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
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
