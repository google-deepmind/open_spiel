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

"""Python spiel example to get all the states in the game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

# pylint: disable=unused-import
from open_spiel.python import games
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.mfg import games as mfg_games
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game")
flags.DEFINE_integer("players", None, "Number of players")
flags.DEFINE_integer("depth_limit", -1, "Depth limit to stop at")
flags.DEFINE_bool("include_terminals", True, "Include terminal states?")
flags.DEFINE_bool("include_chance_states", True, "Include chance states?")


def main(_):
  games_list = pyspiel.registered_games()
  print("Registered games:")
  for game in games_list:
    print(" ", game.short_name)
  print()

  print("Creating game:", FLAGS.game)
  params = {}
  if FLAGS.players is not None:
    params["players"] = FLAGS.players
  game = pyspiel.load_game(FLAGS.game, params)

  print("Getting all states; depth_limit = {}".format(FLAGS.depth_limit))
  all_states = get_all_states.get_all_states(game, FLAGS.depth_limit,
                                             FLAGS.include_terminals,
                                             FLAGS.include_chance_states)

  count = 0
  for state in all_states:
    print(state)
    count += 1

  print()
  print("Total: {} states.".format(count))


if __name__ == "__main__":
  app.run(main)
