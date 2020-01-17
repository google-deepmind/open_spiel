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


def evaluate_bots():
  print("Score RandomBot0: ",
        mean_bot_score(game, [rand_bot0, a0.bot], num_evaluations=40))


def mean_bot_score(game, bots, num_evaluations=100):
  results = np.array([
      pyspiel.evaluate_bots(game.new_initial_state(), bots, iteration)
      for iteration in range(num_evaluations)
  ])
  return np.mean(results, axis=0)


def mean_losses(losses):
  total_loss, policy_loss, value_loss, l2_loss = 0, 0, 0, 0
  for l in losses:
    t, p, v, l2 = l
    total_loss += t
    policy_loss += p
    value_loss += v
    l2_loss += l2
  return alpha_zero.LossValues(total=float('%.3g' % (total_loss / len(losses))),
                               policy=float('%.3g' %
                                            (policy_loss / len(losses))),
                               value=float('%.3g' % (value_loss / len(losses))),
                               l2=float('%.3g' % (l2_loss / len(losses))))


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

  bot = mcts.MCTSBot(
      game,
      # (1.25, 19652),
      1.,
      20,
      evaluator,
      solve=False,
      #  child_selection_fn=alpha_zero.alpha_zero_ucb_score,
      dirichlet_noise=(0.25, 1.))

  # 3. Build an AlphaZero instance
  a0 = alpha_zero.AlphaZero(game,
                            bot,
                            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                            action_selection_transition=4,
                            num_self_play_games=FLAGS.num_self_play_games,
                            batch_size=FLAGS.batch_size)

  # 4. Create a bot using min-max search. It can never lose tic-tac-toe, so 
  # a success condition for our AlphaZero bot is to draw all games with it.
  minimax_bot = MinimaxBot(game)

  # 5. Run training loop
  for num_round in range(FLAGS.num_rounds):
    logging.info("------------- Starting round %s out of %s -------------",
                 num_round, FLAGS.num_rounds)

    # 5.a: 
    if num_round % 3 == 0:
      print("Result against Minimax player: %s draws and %s losses.",
            mean_bot_score(game, [minimax_bot, a0.bot], num_evaluations=50))

    logging.info("Running %s games of self play", FLAGS.num_self_play_games)
    a0.self_play()

    num_training_updates = math.ceil(
        len(a0.replay_buffer) / float(a0.batch_size))
    logging.info("Training net on replay buffer for %s epochs",
                 FLAGS.num_training_epochs)

    for epoch in range(FLAGS.num_training_epochs):
      losses = []
      for _ in range(1, num_training_updates + 1):
        batch_losses = a0.update()
        losses.append(batch_losses)

      m_losses = mean_losses(losses)
      logging.info("     [%s/%s] %s", epoch, FLAGS.num_training_epochs,
                   m_losses)


if __name__ == "__main__":
  app.run(main)
