# Copyright 2022 DeepMind Technologies Limited
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

"""Main file to train and evaluate meta-cfr agent, cfr and cfr-plus."""

from typing import Sequence

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.examples.meta_cfr.sequential_games import cfr
from open_spiel.python.examples.meta_cfr.sequential_games import evaluation
from open_spiel.python.examples.meta_cfr.sequential_games import game_tree_utils
from open_spiel.python.examples.meta_cfr.sequential_games import meta_learning
from open_spiel.python.examples.meta_cfr.sequential_games import openspiel_api


FLAGS = flags.FLAGS

flags.DEFINE_integer("random_seed_size", 30, "Number of random seeds to use.")


def main(argv: Sequence[str]) -> None:
  del argv
  config = {"players": FLAGS.players}
  random_seeds_eval = np.random.choice(
      np.array(list(range(1000))), size=FLAGS.random_seed_size, replace=False)

  # Train a meta-cfr agent
  meta_cfr_agent = meta_learning.MetaCFRRegretAgent(
      training_epochs=1,
      meta_learner_training_epochs=FLAGS.meta_learner_training_epochs,
      game_name=FLAGS.game,
      game_config=config,
      perturbation=FLAGS.perturbation,
      seed=FLAGS.random_seed,
      model_type=FLAGS.model_type,
      best_response=True)
  meta_cfr_agent.train()

  cfr_vals = np.zeros((FLAGS.meta_learner_training_epochs,))
  cfr_plus_vals = np.zeros((FLAGS.meta_learner_training_epochs,))

  for seed in list(random_seeds_eval):

    # Evaluate a meta-cfr agent
    world_state = openspiel_api.WorldState(
        FLAGS.game, config, perturbation=True, random_seed=seed)
    meta_cfr_vals = evaluation.CFRBREvaluation(meta_cfr_agent, world_state)

    # Evaluate a cfr plus agent
    game_tree = game_tree_utils.build_game_tree(
        openspiel_api.WorldState(
            FLAGS.game,
            config,
            perturbation=FLAGS.perturbation,
            random_seed=seed))
    _, cfr_plus_vals = cfr.compute_cfr_plus_values(
        game_tree, FLAGS.meta_learner_training_epochs)

    # Evaluate a cfr agent
    game_tree = game_tree_utils.build_game_tree(
        openspiel_api.WorldState(
            FLAGS.game,
            config,
            perturbation=FLAGS.perturbation,
            random_seed=seed))
    _, cfr_vals = cfr.compute_cfr_values(
        game_tree, FLAGS.meta_learner_training_epochs)

  print("Evaluation seed:", random_seeds_eval)
  print("Meta_cfr agent:", meta_cfr_vals)
  print("cfr_plus agent:", cfr_plus_vals)
  print("cfr agent:", cfr_vals)


if __name__ == "__main__":
  app.run(main)
