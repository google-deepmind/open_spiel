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

"""Example of Generalized PSRO Usage.

We test a generalization of PSRO on kuhn poker. This generalization is capable
of handling N players, assymetric, general sum games. It also supports
Uniform Response, Nash Response and Restricted Nash Response, and new
selection methods can be added easily.

Note : Runtime is slow, because creation of a tabular policy for N player games
takes a very long time : the initialization of the Generalized PSRO solver is
then very slow. Improving that, or changing the initial policy type, could
yield massive speedups.

To use Generalized PSRO, we
1) Create an Oracle object, done through the call to EvolutionaryStrategyOracle.
2) Create the Generalized PSRO solver, and initialize it with the oracle and
    other parameters.
3) Call the solver's step function as many times as wanted / needed. Each call
    generates new strategies, updating the current space.
4) Look at the solver's nash equilibrium distribution, meta games, or retrieve
    its strategies to analyze them.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from absl import app
from absl import flags

from open_spiel.python.algorithms.psro_variations import generalized_psro
from open_spiel.python.algorithms.psro_variations import optimization_oracle
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_integer("num_players", 2, "Number of players")
flags.DEFINE_float("alpha", 0.1, "Evolution parameter.")
flags.DEFINE_float("beta", 0.1, "Evolution parameter.")
flags.DEFINE_integer("sims_per_entry", 2,
                     "Number of simulations to update meta game matrix.")
flags.DEFINE_integer("gen_psro_iterations", 2,
                     "Number of iterations of Generalized PSRO.")
flags.DEFINE_integer("n_evolution_tests", 2,
                     "Number of evolved strategies tested.")
flags.DEFINE_integer("number_policies_sampled", 2,
                     "Number of policies sampled for value estimation.")
flags.DEFINE_integer(
    "number_episodes_sampled", 2,
    "Number of episodes per policy sampled for value estimation.")
flags.DEFINE_boolean("rectify_training", True, "Whether to rectify Nash.")


def main(unused_argv):
  game = pyspiel.load_game(
      FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.num_players)})

  oracle = optimization_oracle.EvolutionaryStrategyOracle(
      n_evolution_tests=FLAGS.n_evolution_tests,
      number_policies_sampled=FLAGS.number_policies_sampled,
      number_episodes_sampled=FLAGS.number_episodes_sampled,
      alpha=FLAGS.alpha,
      beta=FLAGS.beta)
  g_psro_solver = generalized_psro.GenPSROSolver(
      game,
      oracle,
      sims_per_entry=FLAGS.sims_per_entry,
      rectify_training=FLAGS.rectify_training)
  for a in range(FLAGS.gen_psro_iterations):
    g_psro_solver.iteration()
    nash_probabilities = g_psro_solver.get_and_update_meta_strategies()
    logging.info("%s / %s", a + 1, FLAGS.gen_psro_iterations)
    logging.info(nash_probabilities)

  meta_game = g_psro_solver.get_meta_game
  meta_probabilities = g_psro_solver.get_and_update_meta_strategies()

  logging.info("%s meta probabilities", FLAGS.game)
  logging.info(meta_probabilities)
  logging.info("")

  logging.info("%s Meta Game Values", FLAGS.game)
  logging.info(meta_game)
  logging.info("")


if __name__ == "__main__":
  app.run(main)
