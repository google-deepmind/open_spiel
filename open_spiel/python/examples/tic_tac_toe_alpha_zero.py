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

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,\
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AlphaZero tic tac toe example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import minimax
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel

tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_rounds", 25,
    "The number of rounds of self-play followed by neural net training.")

flags.DEFINE_integer("num_self_play_games", 100,
                     "The number of self-play games to play in a round.")

flags.DEFINE_integer(
    "num_training_epochs", 10,
    "The number of passes over the replay buffer done during training.")

flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")

flags.DEFINE_integer("replay_buffer_capacity", 50000,
                     "The size of the replay buffer.")

flags.DEFINE_integer(
    "evaluation_frequency", 3,
    ("The current net will be evaluated against a minimax player every "
     "evaluation_frequency rounds."))

flags.DEFINE_enum("nn_model", "resnet", model_lib.Model.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 2, "How deep should the network be.")


class MinimaxBot(pyspiel.Bot):
  """A minimax bot."""

  def __init__(self, game):
    pyspiel.Bot.__init__(self)
    self._game = game

  def provides_policy(self):
    return False

  def step(self, state):
    _, action = minimax.alpha_beta_search(self._game, state=state)
    return action


def bot_evaluation(game, bots, num_evaluations):
  """Returns a tuple (wins, losses, draws) for player 2."""
  wins, losses, draws = 0, 0, 0
  for i in range(num_evaluations):
    [_, result] = pyspiel.evaluate_bots(game.new_initial_state(), bots, i)
    if result == 0:
      draws += 1
    elif result == 1:
      wins += 1
    else:
      losses += 1

  return (wins, losses, draws)


def main(_):
  game = pyspiel.load_game("tic_tac_toe")

  # 1. Define a model
  model = model_lib.Model(
      FLAGS.nn_model, game.observation_tensor_shape(),
      game.num_distinct_actions(), nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth, weight_decay=1e-4, learning_rate=0.01, path=None)
  print("Model type: {}({}, {}), size: {} variables".format(
      FLAGS.nn_model, FLAGS.nn_width, FLAGS.nn_depth,
      model.num_trainable_variables))

  # 2. Create an MCTS bot using the model
  evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  bot = mcts.MCTSBot(game,
                     1.,
                     20,
                     evaluator,
                     solve=False,
                     dirichlet_noise=(0.25, 1.))

  # 3. Build an AlphaZero instance
  a0 = alpha_zero.AlphaZero(game,
                            bot,
                            model,
                            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                            action_selection_transition=4)

  # 4. Create a bot using min-max search. It can never lose tic-tac-toe, so
  # a success condition for our AlphaZero bot is to draw all games with it.
  minimax_bot = MinimaxBot(game)

  # 5. Run training loop
  for num_round in range(FLAGS.num_rounds):
    logging.info("------------- Starting round %s out of %s -------------",
                 num_round, FLAGS.num_rounds)

    if num_round % FLAGS.evaluation_frequency == 0:
      num_evaluations = 50
      logging.info("Playing %s games against the minimax player.",
                   num_evaluations)
      (_, losses, draws) = bot_evaluation(game, [minimax_bot, a0.bot],
                                          num_evaluations=50)
      logging.info("Result against Minimax player: %s losses and %s draws.",
                   losses, draws)

    logging.info("Running %s games of self play", FLAGS.num_self_play_games)
    a0.self_play(num_self_play_games=FLAGS.num_self_play_games)

    logging.info("Training the net for %s epochs.", FLAGS.num_training_epochs)
    a0.update(FLAGS.num_training_epochs,
              batch_size=FLAGS.batch_size,
              verbose=True)
    logging.info("Cache: %s", evaluator.cache_info())
    evaluator.clear_cache()


if __name__ == "__main__":
  app.run(main)
