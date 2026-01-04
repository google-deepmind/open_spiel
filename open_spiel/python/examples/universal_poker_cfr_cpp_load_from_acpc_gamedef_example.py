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

import pickle
import sys
from absl import app
from absl import flags

import pyspiel

universal_poker = pyspiel.universal_poker

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr"], "CFR solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 100, "Number of iterations")

CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
GAMEDEF
limit
numPlayers = 2
numRounds = 1
blind = 2 4
raiseSize = 4 4 8
firstPlayer = 1
maxRaises = 2 2 2
numSuits = 2
numRanks = 5
numHoleCards = 1
numBoardCards = 0 2 1
stack = 20
END GAMEDEF
"""


def main(_):
  game = universal_poker.load_universal_poker_from_acpc_gamedef(
      CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF
  )

  solver = None
  if FLAGS.solver == "cfr":
    solver = pyspiel.CFRSolver(game)
  elif FLAGS.solver == "cfrplus":
    solver = pyspiel.CFRPlusSolver(game)
  elif FLAGS.solver == "cfrbr":
    solver = pyspiel.CFRBRSolver(game)
  else:
    print("Unknown solver")
    sys.exit(0)

  for i in range(int(_ITERATIONS.value / 2)):
    solver.evaluate_and_update_policy()
    print("Iteration {} exploitability: {:.6f}".format(
        i, pyspiel.exploitability(game, solver.average_policy())))

  filename = "/tmp/{}_solver.pickle".format(FLAGS.solver)
  print("Persisting the model...")
  with open(filename, "wb") as file:
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

  print("Loading the model...")
  with open(filename, "rb") as file:
    loaded_solver = pickle.load(file)
  print("Exploitability of the loaded model: {:.6f}".format(
      pyspiel.exploitability(game, loaded_solver.average_policy())))

  for i in range(int(_ITERATIONS.value / 2)):
    loaded_solver.evaluate_and_update_policy()
    tabular_policy = loaded_solver.tabular_average_policy()
    print(f"Tabular policy length: {len(tabular_policy)}")
    print(
        "Iteration {} exploitability: {:.6f}".format(
            int(_ITERATIONS.value / 2) + i,
            pyspiel.exploitability(game, loaded_solver.average_policy()),
        )
    )


if __name__ == "__main__":
  app.run(main)
