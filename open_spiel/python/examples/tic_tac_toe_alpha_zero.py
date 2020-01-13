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

flags.DEFINE_integer("num_self_play_games", 200,
                     "The number of self-play games to play in a round.")
flags.DEFINE_integer(
    "num_training_updates", 2000,
    "The number of neural net training updates to carry out per round.")

flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")

flags.DEFINE_integer(
    "report_loss_num", 100,
    "The number of training updates for which the loss is averaged over and then reported."
)

flags.DEFINE_integer(
    "window_size", 200 * 10 * 10,
    "The number of training updates for which the loss is averaged over and then reported."
)

flags.DEFINE_float("dirichlet_noise_alpha", 1., "Dirichlet noise used in MCTS.")

flags.DEFINE_string(
    "device", "cpu",
    "Device to evaluate neural nets on. Can be \'cpu\', \'tpu\' or \'gpu\'.")
# flags.DEFINE_integer("mcts_max_simulations", 10, "Number of train episodes.")


class AlphaBetaBot(pyspiel.Bot):

  def __init__(self, game):
    pyspiel.Bot.__init__(self)
    self._game = game

  def provides_policy(self):
    return False

  def step(self, state):
    _, action = minimax.alpha_beta_search(self._game, state=state)
    return action
d

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


def main(_):
  game = pyspiel.load_game("tic_tac_toe")

  net = alpha_zero.keras_mlp(27, game.num_distinct_actions(), num_hidden=64)

  evaluator = alpha_zero.AlphaZeroKerasEvaluator(
      net,
      optimizer=tf.train.AdamOptimizer(),
      # optimizer=tf.train.MomentumOptimizer(2e-2, momentum=0.9),
      device=FLAGS.device)

  bot = mcts.MCTSBot(game, 
      # (1.25, 19652),
      1.4,
                     25,
                     evaluator,
                     solve=False,
                     random_state=np.random.RandomState(42),
                    #  child_selection_fn=alpha_zero.alpha_zero_ucb_score,
                     dirichlet_noise=(0.25, FLAGS.dirichlet_noise_alpha),
                     verbose=False)

  a0 = alpha_zero.AlphaZero(game,
                            bot,
                            replay_buffer_capacity=FLAGS.window_size,
                            action_selection_transition=4,
                            num_self_play_games=FLAGS.num_self_play_games,
                            batch_size=FLAGS.batch_size)

  ab_bot = AlphaBetaBot(game)
  rand_bot1 = uniform_random.UniformRandomBot(1, np.random.RandomState(3)) 
  rand_bot0 = uniform_random.UniformRandomBot(0, np.random.RandomState(3)) 

  for num_round in range(FLAGS.num_rounds):
    logging.info("------------- Starting round %s out of %s -------------",
                 num_round, FLAGS.num_rounds)
    logging.info("Running %s games of self play", FLAGS.num_self_play_games)
    a0.self_play()
    optim = a0.bot.evaluator.optimizer
    tf.variables_initializer(optim.variables())

    logging.info("Training net on replay buffer for %s iterations",
                 FLAGS.num_training_updates)

    losses = []
    for train_num in range(1, FLAGS.num_training_updates + 1):

      if (train_num % FLAGS.report_loss_num) == 0:
        m_losses = mean_losses(losses)
        logging.info("     [%s/%s] %s", train_num, FLAGS.num_training_updates,
                     m_losses)
        losses = []
        print("cleaned")
      batch_losses = a0.update()
      # print(batch_losses)
      losses.append(batch_losses)

    # loss_total, loss_policy, loss_value, loss_l2 = a0.train()
    # print("Total: ", np.mean(loss_total), "Policy", np.mean(loss_policy), "Value", np.mean(loss_value), "L2", np.mean(loss_l2))
    # print("Score Random: ", mean_bot_score(game, [rand_bot0, a0.bot], num_evaluations=40))

    print("Score SelfPlay: ",
          mean_bot_score(game, [a0.bot, a0.bot], num_evaluations=40))
    # print("Score AlphaBeta: ",
    #       mean_bot_score(game, [ab_bot, a0.bot], num_evaluations=40))
    print("Score RandomBot0: ",
      mean_bot_score(game, [rand_bot0, a0.bot], num_evaluations=40))
    print("Score RandomBot1: ",
          mean_bot_score(game, [a0.bot, rand_bot1], num_evaluations=40))

if __name__ == "__main__":
  app.run(main)
