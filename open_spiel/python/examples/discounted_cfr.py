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

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 500, "Number of iterations")
flags.DEFINE_string(
    "game",
    "turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,players=2,points_order=descending))",
    "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 10, "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  discounted_cfr_solver = discounted_cfr.DCFRSolver(game)

  for i in range(FLAGS.iterations):
    discounted_cfr_solver.evaluate_and_update_policy()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.exploitability(
          game, discounted_cfr_solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
