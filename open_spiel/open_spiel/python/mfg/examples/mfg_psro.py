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

"""Mean-Field PSRO examples."""

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.mfg.algorithms import correlated_equilibrium
from open_spiel.python.mfg.algorithms import mf_psro
from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import hedge
from open_spiel.python.mfg.algorithms.regret import polynomial_weights
from open_spiel.python.mfg.algorithms.regret import regret_matching
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
from open_spiel.python.mfg.games import dynamic_routing  # pylint: disable=unused-import
from open_spiel.python.mfg.games import normal_form_game  # pylint: disable=unused-import
from open_spiel.python.mfg.games import predator_prey  # pylint: disable=unused-import
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_mfg_predator_prey",
                    "Name of the game.")
flags.DEFINE_integer(
    "regret_steps_per_step",
    1000,
    "number of runs to average value function over.",
)
flags.DEFINE_integer(
    "value_estimation_n", 1, "number of runs to average value function over."
)
flags.DEFINE_string(
    "value_estimator", "sampled", "Best Response type : `ce` or `cce`."
)
flags.DEFINE_string(
    "regret_minimizer",
    "hedge",
    "Which regret minimization algorithm to use : `rm` for"
    "Regret Matching, `hedge` for Hedge, `poly` for Polynomial "
    "Weights.",
)
flags.DEFINE_integer("n_iter", 1000, "Num PSRO iterations.")
flags.DEFINE_integer("compress_every", 1, "Compress every")
flags.DEFINE_float("compress_lbd", 0.0, "Compression lambda.")
flags.DEFINE_float("eta", None, "Polynomial Weight algorithm eta.")
flags.DEFINE_string(
    "best_responder", "cce", "Best Response type : `ce` or `cce`."
)
flags.DEFINE_bool(
    "compute_internal_regret",
    False,
    "Compute internal (Or external if False) regret",
)
flags.DEFINE_bool("compute_ce_gap", False, "Compute `ce_gap`")
flags.DEFINE_integer("seed", 1, "Seed value.")

GAME_SETTINGS = {
    "mfg_crowd_modelling_2d": {
        "only_distribution_reward": False,
        "forbidden_states": "[0|0;0|1]",
        "initial_distribution": "[0|2;0|3]",
        "initial_distribution_value": "[0.5;0.5]",
    }
}


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  mfg_game = pyspiel.load_game(
      FLAGS.game_name, GAME_SETTINGS.get(FLAGS.game_name, {})
  )

  eta = FLAGS.eta
  regret_steps_per_step = FLAGS.regret_steps_per_step

  best_responder = FLAGS.best_responder
  compute_ce_gap = FLAGS.compute_ce_gap
  compute_internal_regret = FLAGS.compute_internal_regret

  if FLAGS.value_estimator == "sampled":
    value_estimator = utils.sample_value
  elif FLAGS.value_estimator == "exact":
    value_estimator = utils.get_exact_value
  else:
    raise NameError(
        "Unknown value estimator {}. Valid names are `sampled`, `exact`."
        .format(FLAGS.value_estimator)
    )

  if FLAGS.regret_minimizer == "hedge":
    regret_minimizer = hedge.Hedge(
        mfg_game,
        [],
        eta,
        regret_steps_per_step,
        compress_nus=True,
        compress_every=FLAGS.compress_every,
        compress_lbd=FLAGS.compress_lbd,
        value_estimator=value_estimator,
        value_estimation_n=FLAGS.value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )
  elif FLAGS.regret_minimizer == "rm":
    regret_minimizer = regret_matching.RegretMatching(
        mfg_game,
        [],
        eta,
        regret_steps_per_step,
        compress_nus=True,
        compress_every=FLAGS.compress_every,
        compress_lbd=FLAGS.compress_lbd,
        value_estimator=value_estimator,
        value_estimation_n=FLAGS.value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )
  elif FLAGS.regret_minimizer == "poly":
    regret_minimizer = polynomial_weights.PolynomialWeightAlgorithm(
        mfg_game,
        [],
        eta,
        regret_steps_per_step,
        compress_nus=True,
        compress_every=FLAGS.compress_every,
        compress_lbd=FLAGS.compress_lbd,
        value_estimator=value_estimator,
        value_estimation_n=FLAGS.value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )
  else:
    raise NameError(
        "Unknown regret minimizer {}.".format(FLAGS.regret_minimizer)
    )

  if best_responder == "cce":
    best_responder = correlated_equilibrium.cce_br
  elif best_responder == "ce":
    best_responder = correlated_equilibrium.ce_br
  elif best_responder == "ce_partial":
    best_responder = correlated_equilibrium.partial_ce_br
  else:
    raise NameError(
        "Unknown best responder {}. Valid names are `cce` and `ce`.".format(
            FLAGS.best_responder
        )
    )

  mfpsro = mf_psro.MeanFieldPSRO(
      mfg_game,
      regret_minimizer,
      regret_steps_per_step,
      best_responder=best_responder,
  )

  for j in range(FLAGS.n_iter):
    logging.info("Iteration {} of MF-PSRO".format(j))  # pylint: disable=logging-format-interpolation
    print("PSRO Step")
    mfpsro.step()

    print("Equilibrium Computation")
    policies, nus, mus, rhos = mfpsro.get_equilibrium()

    print("Welfare Computation")
    average_welfare = correlated_equilibrium.compute_average_welfare(
        mfg_game, policies, mus, rhos, nus
    )

    print("CCE Gap Computation")
    cce_gap_value = correlated_equilibrium.cce_gap(
        mfg_game, policies, rhos, mus, nus, compute_true_rewards=True
    )
    if compute_ce_gap:
      print("CE Gap Computation")
      ce_gap_value = correlated_equilibrium.ce_gap(
          mfg_game, policies, rhos, mus, nus, compute_true_rewards=True
      )
    else:
      ce_gap_value = 0.0

    print("CCE Gap value : {}".format(cce_gap_value))
    print("CE Gap value : {}".format(ce_gap_value))
    print("Average welfare : {}".format(average_welfare))
    print("")


if __name__ == "__main__":
  app.run(main)
