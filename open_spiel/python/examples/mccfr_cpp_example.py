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

"""Example use of the C++ MCCFR algorithms on Kuhn Poker.

This examples calls the underlying C++ implementations via the Python bindings.
Note that there are some pure Python implementations of some of these algorithms
in python/algorithms as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
from absl import flags

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sampling",
    "external",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)
flags.DEFINE_integer("iterations", 50, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")

MODEL_FILE_NAME = "{}_sampling_mccfr_solver.pickle"


def run_iterations(game, solver, start_iteration=0):
  """Run iterations of MCCFR."""
  for i in range(int(FLAGS.iterations / 2)):
    solver.run_iteration()
    policy = solver.average_policy()
    exploitability = pyspiel.exploitability(game, policy)

    # We also compute NashConv to highlight an important API feature:
    # when using Monte Carlo sampling, the policy
    # may not have a table entry for every info state.
    # Therefore, when calling nash_conv, ensure the third argument,
    # "use_state_get_policy" is set to True
    # See https://github.com/deepmind/open_spiel/issues/500
    nash_conv = pyspiel.nash_conv(game, policy, True)

    print("Iteration {} nashconv: {:.6f} exploitability: {:.6f}".format(
        start_iteration + i, nash_conv, exploitability))


def main(_):
  game = pyspiel.load_game(
      FLAGS.game,
      {"players": FLAGS.players},
  )

  if FLAGS.sampling == "external":
    solver = pyspiel.ExternalSamplingMCCFRSolver(
        game,
        avg_type=pyspiel.MCCFRAverageType.FULL,
    )
  elif FLAGS.sampling == "outcome":
    solver = pyspiel.OutcomeSamplingMCCFRSolver(game)

  run_iterations(game, solver)

  print("Persisting the model...")
  with open(MODEL_FILE_NAME.format(FLAGS.sampling), "wb") as file:
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

  print("Loading the model...")
  with open(MODEL_FILE_NAME.format(FLAGS.sampling), "rb") as file:
    loaded_solver = pickle.load(file)
  print("Exploitability of the loaded model: {:.6f}".format(
      pyspiel.exploitability(game, loaded_solver.average_policy())))

  run_iterations(game, solver, start_iteration=int(FLAGS.iterations / 2))


if __name__ == "__main__":
  app.run(main)
