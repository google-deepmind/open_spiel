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

"""Export game in gambit .efg format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms.gambit import export_gambit
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_string("out", "/tmp/gametree.efg", "Name of output file, e.g., "
                    "[*.efg].")
flags.DEFINE_boolean("print", False, "Print the tree to stdout "
                     "instead of saving to file.")


def main(argv):
  del argv

  game = pyspiel.load_game(FLAGS.game)
  game_type = game.get_type()

  if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
    logging.warn("%s is not turn-based. Trying to reload game as turn-based.",
                 FLAGS.game)
    game = pyspiel.load_game_as_turn_based(FLAGS.game)

  gametree = export_gambit(game)  # use default decorators
  if FLAGS.print:
    print(gametree)
  else:
    with open(FLAGS.out, "w") as f:
      f.write(gametree)
    logging.info("Game tree for %s saved to file: %s", FLAGS.game, FLAGS.out)


if __name__ == "__main__":
  app.run(main)
