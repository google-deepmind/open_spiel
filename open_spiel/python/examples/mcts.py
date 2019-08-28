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

"""MCTS example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import mcts
import pyspiel

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
flags.DEFINE_integer("mcts_player", 0, "Which player uses MCTS.")
flags.DEFINE_integer("rollout_count", 100, "How many rollouts to do.")
flags.DEFINE_integer("max_search_nodes", 10000, "How many nodes to expand.")

FLAGS = flags.FLAGS


def main(unused_argv):
  uct_c = 2
  game = pyspiel.load_game(FLAGS.game)
  state = game.new_initial_state()
  print("Initial state: ", str(state))

  # Check that the games satisfies the conditions for the implemented MCTS
  # algorithm
  if game.num_players() not in (1, 2):
    raise ValueError("Game must be a 1-player game or 2-player zero-sum game")
  if (game.num_players() == 2 and
      game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM):
    raise ValueError("Game must be a 1-player game or 2-player zero-sum game")

  # Create MCTS bot
  evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count)
  mcts_bot = mcts.MCTSBot(game, FLAGS.mcts_player, uct_c,
                          FLAGS.max_search_nodes, evaluator)

  # Create random bot
  random_bot = pyspiel.make_uniform_random_bot(game, 1 - FLAGS.mcts_player, 123)

  if FLAGS.mcts_player == 0:
    bots = [mcts_bot, random_bot]
  else:
    bots = [random_bot, mcts_bot]

  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      _, action = bots[state.current_player()].step(state)
      print("Player ", state.current_player(), ", randomly sampled action: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)

    print("Next state: ", str(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Return for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
