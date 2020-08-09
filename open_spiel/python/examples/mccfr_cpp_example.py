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

"""Example use of the C++ MCCFR algorithms on Kuhn Poker.

This examples calls the underlying C++ implementations via the Python bindings.
Note that there are some pure Python implementations of some of these algorithms
in python/algorithms as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def main(_):
  game = pyspiel.load_game(
      FLAGS.game,
      {"players": pyspiel.GameParameter(FLAGS.players)},
  )

  if FLAGS.sampling == "external":
    solver = pyspiel.ExternalSamplingMCCFRSolver(
        game,
        avg_type=pyspiel.MCCFRAverageType.FULL,
    )
  elif FLAGS.sampling == "outcome":
    solver = pyspiel.OutcomeSamplingMCCFRSolver(game)

  for i in range(FLAGS.iterations):
    solver.run_iteration()
    print("Iteration {} exploitability: {:.6f}".format(
        i, pyspiel.exploitability(game, solver.average_policy())))


if __name__ == "__main__":
  app.run(main)
