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

import collections
import sys
import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

_KNOWN_PLAYERS = ["mcts", "random", "human"]

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 10, "How many rollouts to do.")
flags.DEFINE_integer("max_search_nodes", 10000, "How many nodes to expand.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  if bot_type == "mcts":
    evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count)
    return mcts.MCTSBot(game, player_id, FLAGS.uct_c,
                        FLAGS.max_search_nodes, evaluator)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(game, player_id, np.random)
  if bot_type == "human":
    return human.HumanBot(game, player_id)
  raise ValueError("Invalid bot type: %s" % bot_type)


def _play_game(game):
  """Plays one game."""
  state = game.new_initial_state()
  _opt_print("Initial state: ", str(state))

  bots = [
      _init_bot(FLAGS.player1, game, 0),
      _init_bot(FLAGS.player2, game, 1),
  ]

  history = []
  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(state.current_player(), action)
      _opt_print("Sampled outcome: ", action_str)
      state.apply_action(action)
      history.append(action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      t1 = time.time()
      # Decision node: sample action for the single current player
      bot = bots[state.current_player()]
      _, action = bot.step(state)
      action_str = state.action_to_string(state.current_player(), action)
      _opt_print("Player {} sampled action: {}".format(
          state.current_player(), action_str))
      diff = time.time() - t1
      if isinstance(bot, mcts.MCTSBot):
        _opt_print("Took %.3f secs, %.1f rollouts/s" % (
            diff, (FLAGS.rollout_count * FLAGS.max_search_nodes) / diff))
      history.append(action_str)
      state.apply_action(action)

    _opt_print("Next state: ", str(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  print("Returns:", " ".join(map(str, returns)),
        ", Game actions:", " ".join(history))
  return returns, history


def main(unused_argv):
  game = pyspiel.load_game(FLAGS.game)

  # Check that the games satisfies the conditions for the implemented MCTS
  # algorithm.
  if game.num_players() not in (1, 2):
    sys.exit("Game must be a 1-player game or 2-player zero-sum game")
  if (game.num_players() == 2 and
      game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM):
    sys.exit("Game must be a 1-player game or 2-player zero-sum game")

  histories = collections.defaultdict(int)
  overall_returns = [0, 0]
  overall_wins = [0, 0]
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game)
      histories[" ".join(history)] += 1
      for i, v in enumerate(returns):
        overall_returns[i] += v
        if v > 0:
          overall_wins[i] += 1
  except (KeyboardInterrupt, EOFError):
    game_num -= 1
    print("Caught a KeyboardInterrupt, stopping early.")
  print("Number of games played:", game_num + 1)
  print("Number of distinct games played:", len(histories))
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)


if __name__ == "__main__":
  app.run(main)
