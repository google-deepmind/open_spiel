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

"""Example of RNR Usage.

We test Rectified Nash Response (See Balduzzi et Al., 2019,
https://arxiv.org/pdf/1901.08106.pdf ) on kuhn poker.

To use RNR, we
1) Create an Oracle object, done through the call to EvolutionaryStrategyOracle.
2) Create the RNR solver, and initialize it with the oracle and other
    parameters.
3) Call the solver's step function as many times as wanted / needed. Each call
    generates new strategies, updating the current space.
4) Look at the solver's nash equilibrium distribution, meta games, or retrieve
    its strategies to analyze them.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from open_spiel.python.algorithms.psro_variations import optimization_oracle
from open_spiel.python.algorithms.psro_variations import rectified_nash_response
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_float("alpha", 0.1, "Evolution parameter.")
flags.DEFINE_float("beta", 0.1, "Evolution parameter.")
flags.DEFINE_integer("sims_per_entry", 2,
                     "Number of simulations to update meta game matrix.")
flags.DEFINE_integer("rnr_iterations", 2, "Number of iterations or RNR.")
flags.DEFINE_integer("n_evolution_tests", 2,
                     "Number of evolved strategies tested.")
flags.DEFINE_integer("number_policies_sampled", 2,
                     "Number of policies sampled for value estimation.")
flags.DEFINE_integer(
    "number_episodes_sampled", 2,
    "Number of episodes per policy sampled for value estimation.")
flags.DEFINE_boolean("rectify_nash", True, "Whether to rectify Nash.")


def main(unused_argv):
  game = pyspiel.load_game(FLAGS.game)

  oracle = optimization_oracle.EvolutionaryStrategyOracle(
      n_evolution_tests=FLAGS.n_evolution_tests,
      number_policies_sampled=FLAGS.number_policies_sampled,
      number_episodes_sampled=FLAGS.number_episodes_sampled,
      alpha=FLAGS.alpha,
      beta=FLAGS.beta)
  rnr_solver = rectified_nash_response.RNRSolver(
      game,
      oracle,
      sims_per_entry=FLAGS.sims_per_entry,
      rectify_training=FLAGS.rectify_nash)
  for a in range(FLAGS.rnr_iterations):
    rnr_solver.iteration()
    nash_probabilities = rnr_solver.get_and_update_meta_strategies()
    print("{} / {}".format(a + 1, FLAGS.rnr_iterations))
    print(nash_probabilities)
  meta_game = rnr_solver.get_meta_game
  nash_probabilities = rnr_solver.get_and_update_meta_strategies()

  print(FLAGS.game + " Nash probabilities")
  print(nash_probabilities)
  print("")

  print(FLAGS.game + " Meta Game Values")
  print(meta_game)
  print("")


if __name__ == "__main__":
  app.run(main)
