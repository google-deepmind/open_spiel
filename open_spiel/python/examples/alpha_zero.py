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

# Lint as: python3
"""Starting point for playing with the AlphaZero algorithm."""

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn

flags.DEFINE_string("game", "connect_four", "Name of the game.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 300, "How many simulations to run.")
flags.DEFINE_integer("train_batch_size", 2 ** 10, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 2 ** 16,
                     "How many states to store in the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 3,
                     "How many times to learn from each state.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1,
                   "Temperature for final move selection.")
flags.DEFINE_integer("temperature_drop", 10,  # Less than AZ due to short games.
                     "Drop the temperature to 0 after this many moves.")
flags.DEFINE_enum("nn_model", "resnet", model_lib.Model.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_string("path", None, "Where to save checkpoints.")
flags.DEFINE_integer("checkpoint_freq", 100, "Save a checkpoint every N steps.")
flags.DEFINE_integer("actors", 2, "How many actors to run.")
flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_integer("evaluation_window", 100,
                     "How many games to average results over.")
flags.DEFINE_integer(
    "eval_levels", 7,
    ("Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
     " simulations for n in range(eval_levels). Default of 7 means "
     "running mcts with up to 1000 times more simulations."))
flags.DEFINE_integer("max_steps", 0, "How many learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def main(unused_argv):
  config = alpha_zero.Config(
      game=FLAGS.game,
      path=FLAGS.path,
      learning_rate=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay,
      train_batch_size=FLAGS.train_batch_size,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_buffer_reuse=FLAGS.replay_buffer_reuse,
      max_steps=FLAGS.max_steps,
      checkpoint_freq=FLAGS.checkpoint_freq,

      actors=FLAGS.actors,
      evaluators=FLAGS.evaluators,
      uct_c=FLAGS.uct_c,
      max_simulations=FLAGS.max_simulations,
      policy_alpha=FLAGS.policy_alpha,
      policy_epsilon=FLAGS.policy_epsilon,
      temperature=FLAGS.temperature,
      temperature_drop=FLAGS.temperature_drop,
      evaluation_window=FLAGS.evaluation_window,
      eval_levels=FLAGS.eval_levels,

      nn_model=FLAGS.nn_model,
      nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth,
      observation_shape=None,
      output_size=None,

      quiet=FLAGS.quiet,
  )
  alpha_zero.alpha_zero(config)


if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)
