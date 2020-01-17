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
"""AlphaZero example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app
from absl import flags
from absl import logging
import numpy as np
import math

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from open_spiel.python.algorithms import alpha_zero
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import minimax
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import evaluate_bots
import pyspiel

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
    "The current net will be evaluated against a minimax player every \
      evaluation_frequency rounds.")

flags.DEFINE_string(
    "net_type", "mlp",
    "The type of network to use. Can be either 'mlp' or 'resnet.")

flags.DEFINE_string(
    "device", "cpu",
    "Device to evaluate neural nets on. Can be 'cpu', 'tpu' or 'gpu'.")


class MinimaxBot(pyspiel.Bot):

  def __init__(self, game):
    pyspiel.Bot.__init__(self)
    self._game = game

  def provides_policy(self):
    return False

  def step(self, state):
    _, action = minimax.alpha_beta_search(self._game, state=state)
    return action


def bot_evaluation(game, bots, num_evaluations):
  '''Returns a tuple (wins, losses, draws) for player 2.'''
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


def mlp_feature_extractor(state):
  obs = np.array(state.observation_tensor(), dtype=np.float32)
  return np.reshape(obs[9:18] - obs[18:], (1, 9))


def main(_):
  game = pyspiel.load_game("tic_tac_toe")
  num_actions = game.num_distinct_actions()
  observation_shape = game.observation_tensor_shape()

  # 1. Define a keras net
  if FLAGS.net_type == "resnet":
    feature_extractor = None
    net = alpha_zero.keras_resnet(observation_shape,
                                  num_actions=num_actions,
                                  num_residual_blocks=1,
                                  num_filters=256,
                                  data_format='channels_first')
  elif FLAGS.net_type == "mlp":
    # The full length-27 observation_tensor could be trained on. But this
    # demonstrates the use of a custom feature extractor, and the training
    # can be faster with this smaller feature representation.
    feature_extractor = mlp_feature_extractor
    net = alpha_zero.keras_mlp(9, num_actions, num_layers=2, num_hidden=64)
  else:
    raise ValueError(
        "Invalid value for 'net_type'. Must be either 'mlp' or 'resnet', but was %s",
        FLAGS.net_type)

  # 2. Create an MCTS bot using the previous keras net
  evaluator = alpha_zero.AlphaZeroKerasEvaluator(
      net,
      optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
      device=FLAGS.device,
      feature_extractor=feature_extractor)

  bot = mcts.MCTSBot(game,
                     1.,
                     20,
                     evaluator,
                     solve=False,
                     dirichlet_noise=(0.25, 1.))

  # 3. Build an AlphaZero instance
  a0 = alpha_zero.AlphaZero(game,
                            bot,
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


if __name__ == "__main__":
  app.run(main)
