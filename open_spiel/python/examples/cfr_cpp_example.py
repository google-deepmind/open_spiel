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

import pickle
from absl import app
from absl import flags

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr"], "CFR solver")
flags.DEFINE_integer("iterations", 20, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")


def main(_):
  game = pyspiel.load_game(
      FLAGS.game,
      {"players": FLAGS.players},
  )

  if FLAGS.solver == "cfr":
    solver = pyspiel.CFRSolver(game)
  elif FLAGS.solver == "cfrplus":
    solver = pyspiel.CFRPlusSolver(game)
  elif FLAGS.solver == "cfrbr":
    solver = pyspiel.CFRBRSolver(game)

  for i in range(int(FLAGS.iterations / 2)):
    solver.evaluate_and_update_policy()
    print("Iteration {} exploitability: {:.6f}".format(
        i, pyspiel.exploitability(game, solver.average_policy())))

  print("Persisting the model...")
  with open("{}_solver.pickle".format(FLAGS.solver), "wb") as file:
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

  print("Loading the model...")
  with open("{}_solver.pickle".format(FLAGS.solver), "rb") as file:
    loaded_solver = pickle.load(file)
  print("Exploitability of the loaded model: {:.6f}".format(
      pyspiel.exploitability(game, loaded_solver.average_policy())))

  for i in range(int(FLAGS.iterations / 2)):
    loaded_solver.evaluate_and_update_policy()
    print("Iteration {} exploitability: {:.6f}".format(
        int(FLAGS.iterations / 2) + i,
        pyspiel.exploitability(game, loaded_solver.average_policy())))


if __name__ == "__main__":
  app.run(main)
