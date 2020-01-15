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

# import logging
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
    "num_rounds", 400,
    "The number of rounds of self-play followed by neural net training.")

flags.DEFINE_integer("num_self_play_games", 150,
                     "The number of self-play games to play in a round.")

flags.DEFINE_integer(
    "num_training_epochs", 10,
    "The number of passes over the replay buffer done during training.")

flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")

flags.DEFINE_integer("replay_buffer_capacity", 150000,
                     "The size of the replay buffer.")

flags.DEFINE_float("dirichlet_noise_alpha", 1., "Dirichlet noise used in MCTS.")

flags.DEFINE_string(
    "device", "cpu",
    "Device to evaluate neural nets on. Can be 'cpu', 'tpu' or 'gpu'.")


class AlphaBetaBot(pyspiel.Bot):

  def __init__(self, game):
    pyspiel.Bot.__init__(self)
    self._game = game

  def provides_policy(self):
    return False

  def step(self, state):
    _, action = minimax.alpha_beta_search(self._game, state=state)
    return action


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


def tic_tac_toe_feature_extractor(state):
  obs = np.array(state.observation_tensor(), dtype=np.float32).reshape(
      (1, 3, 3, 3))
  return np.transpose(obs, (0, 2, 3, 1))


def main(_):
  game = pyspiel.load_game("tic_tac_toe")

  net = alpha_zero.keras_resnet((3, 3, 3),
                                num_actions=9,
                                num_residual_blocks=1,
                                num_filters=256)

  evaluator = alpha_zero.AlphaZeroKerasEvaluator(
      net,
      optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
      device=FLAGS.device,
      feature_extractor=tic_tac_toe_feature_extractor)

  bot = mcts.MCTSBot(game, (1.25, 19652),
                     25,
                     evaluator,
                     solve=False,
                     random_state=np.random.RandomState(42),
                     child_selection_fn=alpha_zero.alpha_zero_ucb_score,
                     dirichlet_noise=(0.25, FLAGS.dirichlet_noise_alpha),
                     verbose=False)

  a0 = alpha_zero.AlphaZero(game,
                            bot,
                            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                            action_selection_transition=2,
                            num_self_play_games=FLAGS.num_self_play_games,
                            batch_size=FLAGS.batch_size)

  # Create bots to tetst against
  ab_bot = AlphaBetaBot(game)
  rand_bot0 = uniform_random.UniformRandomBot(0, np.random.RandomState(3))

  for num_round in range(FLAGS.num_rounds):
    logging.info("------------- Starting round %s out of %s -------------",
                 num_round, FLAGS.num_rounds)
    logging.info("Running %s games of self play", FLAGS.num_self_play_games)
    values = a0.self_play()
    print("Self-Play Values: ", np.mean(values, axis=0))

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

    if num_round % 10 == 0:
      print("Score AlphaBeta: ",
            mean_bot_score(game, [ab_bot, a0.bot], num_evaluations=100))
      print("Score RandomBot0: ",
            mean_bot_score(game, [rand_bot0, a0.bot], num_evaluations=100))


if __name__ == "__main__":
  app.run(main)
