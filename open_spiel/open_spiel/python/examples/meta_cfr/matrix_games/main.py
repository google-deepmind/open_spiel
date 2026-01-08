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

"""Main file to train and evaluate meta-regret and regret matching agents."""

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.examples.meta_cfr.matrix_games import evaluation
from open_spiel.python.examples.meta_cfr.matrix_games import matrix_dataset
from open_spiel.python.examples.meta_cfr.matrix_games import meta_selfplay_agent
from open_spiel.python.examples.meta_cfr.matrix_games import regret_matching_agent


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_integer("evaluation_steps", 1000, "Number of evaluation steps.")
flags.DEFINE_integer("num_batches", 1,
                     "Number of batches to train a meta optimizer.")
flags.DEFINE_integer("repeats", 10,
                     "Number of training each batch in meta learning.")
flags.DEFINE_integer("seed", 10, "random seed.")
flags.DEFINE_integer("min_val", 0,
                     "minimum value for randomizing a payoff matrix.")
flags.DEFINE_integer("max_val", 10,
                     "maximum value for randomizing a payoff matrix.")
flags.DEFINE_integer("num_actions", 3, "Number of actions an agent can take.")
flags.DEFINE_bool("single_problem", False,
                  "If the matrix dataset generates only a single matrix.")


def selfplay_main(argv):
  """Self play."""
  del argv
  np.random.seed(FLAGS.seed)
  # rock-paper-scissor
  base_matrix = np.array([[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]] *
                         FLAGS.batch_size)
  dataset = matrix_dataset.Dataset(
      base_matrix=base_matrix,
      num_training_batches=FLAGS.num_batches,
      minval=FLAGS.min_val,
      maxval=FLAGS.max_val)
  data_loader = dataset.get_training_batch()
  eval_payoff_batch = dataset.get_eval_batch()

  mr_agent = meta_selfplay_agent.MetaSelfplayAgent(
      repeats=FLAGS.repeats,
      training_epochs=FLAGS.evaluation_steps,
      data_loader=data_loader)
  mr_agent.train()

  mr_agent2 = meta_selfplay_agent.MetaSelfplayAgent(
      repeats=FLAGS.repeats,
      training_epochs=FLAGS.evaluation_steps,
      data_loader=data_loader)
  mr_agent2.train()

  rm_agent = regret_matching_agent.RegretMatchingAgent(
      num_actions=FLAGS.num_actions, data_loader=data_loader)
  rm_agent.train()

  rm_agent2 = regret_matching_agent.RegretMatchingAgent(
      num_actions=FLAGS.num_actions, data_loader=data_loader)
  rm_agent2.train()

  print("Regret matching")
  evaluation.evaluate_in_selfplay(
      agent_x=rm_agent,
      agent_y=rm_agent2,
      payoff_batch=eval_payoff_batch,
      steps_count=FLAGS.evaluation_steps)

  print("Meta regret matching")
  evaluation.evaluate_in_selfplay(
      agent_x=mr_agent,
      agent_y=mr_agent2,
      payoff_batch=eval_payoff_batch,
      steps_count=FLAGS.evaluation_steps)


if __name__ == "__main__":
  app.run(selfplay_main)
