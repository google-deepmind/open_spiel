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

# pylint: disable=logging-fstring-interpolation

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import deep_cfr
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 101, "Number of iterations")
flags.DEFINE_integer("num_traversals", 375, "Number of traversals/games")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")

# Recommended parameters:
#   For more, see https://github.com/aicenter/openspiel_reproductions/

#   Parameter                        Value
#   ---------------------------------------
#   num_traversals                   1500
#   batch_size_advantage             2048
#   batch_size_strategy              2048
#   num_hidden                       64
#   num_layers                       3
#   reinitialize_advantage_networks  True
#   learning_rate                    1e-3
#   memory_capacity                  1e6
#   policy_network_train_steps       5000
#   advantage_network_train_steps    750


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)

  game = pyspiel.load_game(FLAGS.game_name)
  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(64,),
      advantage_network_layers=(64,),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      reinitialize_advantage_networks=True,
      learning_rate=1e-3,
      batch_size_advantage=256,
      batch_size_strategy=256,
      memory_capacity=100000,
      policy_network_train_steps=2500,
      advantage_network_train_steps=375,
      print_nash_convs=False,  # for debugging purposes
  )

  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info(
        f"Advantage Buffer Size for player {player}:"
        f" {len(deep_cfr_solver.advantage_buffers[player])}"
    )
  logging.info(f"Strategy Buffer Size: {len(deep_cfr_solver.strategy_buffer)}")
  logging.info(f"Final policy loss: {policy_loss}")

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)

  conv = exploitability.nash_conv(game, average_policy)
  logging.info(f"Deep CFR in {FLAGS.game_name} - NashConv: {conv}")

  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  if FLAGS.game_name == "kuhn_poker":
    # We know EVs
    logging.info(
        f"Computed player 0 value: {average_policy_values[0]:.2f} (expected:"
        f" {-1/18:.2f})."
    )
    logging.info(
        f"Computed player 1 value: {average_policy_values[1]:.2f} (expected:"
        f" {1/18:.2f})."
    )
  else:
    logging.info(f"Computed player 0 value: {average_policy_values[0]:.2f}")
    logging.info(f"Computed player 1 value: {average_policy_values[1]:.2f}")


if __name__ == "__main__":
  app.run(main)
