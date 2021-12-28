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

"""Example use of the MCCFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sampling",
    "outcome",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)
flags.DEFINE_integer("iterations", 10000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 1000,
                     "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.players})
  if FLAGS.sampling == "external":
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
  else:
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)
  for i in range(FLAGS.iterations):
    cfr_solver.iteration()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
