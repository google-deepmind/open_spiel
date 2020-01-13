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
"""AlphaZero Bot implemented in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import math

import tensorflow.compat.v1 as tf

import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import masked_softmax
from open_spiel.python.algorithms import dqn

MCTSResult = collections.namedtuple("MCTSResult",
                                    "observation target_value target_policy")

LossValues = collections.namedtuple("LossValues", "total policy value l2")


class AlphaZero(object):
  """AlphaZero implementation following the pseudocode AlphaZero implementation
  given in the paper with DOI 10.1126/science.aar6404."""

  def __init__(self,
               game,
               bot,
               replay_buffer_capacity=int(1e6),
               action_selection_transition=30,
               num_self_play_games=5000,
               num_training_rounds=10,
               batch_size=4096,
               random_state=None):
    """
    Args:
      game: a pyspiel.Game object
      bot: an MCTSBot object.
      replay_buffer_capacity: the size of the replay buffer in which the results 
        of self-play games are stored.

      num_self_play_games: the number of self-play games to play before each
        training round.
      num_training_rounds: the number rounds of alternating training and
        and self-play.
      num_training_updates: the 
      batch_size: the number of examples used for a single training update. Note
        that this batch size must be small enough for the neural net training 
        update to fit into device memory.
      action_selection_transition: an integer representing the move number in a 
        game of self-play when greedy action selection is used. Before this,
        actions are sampled from the MCTS policy.
      random_state: An optional numpy RandomState to make it deterministic.

    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    game_info = game.get_type()
    if game.num_players() != 2:
      raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
      raise ValueError("The game must be a Deterministic one, not {}".format(
          game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise ValueError(
          "The game must be a perfect information one, not {}".format(
              game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("The game must be turn-based, not {}".format(
          game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
      raise ValueError("The game must be 0-sum, not {}".format(game.utility))
    if game.num_players() != 2:
      raise ValueError("Game must have exactly 2 players.")

    self.bot = bot
    self.game = game
    self.buffer = dqn.ReplayBuffer(replay_buffer_capacity)
    self.num_self_play_games = num_self_play_games
    self._action_selection_transition = action_selection_transition
    self.batch_size = batch_size

  def update(self):
    data = self.buffer.sample(self.batch_size, replace=True)
    (total_loss, policy_loss, value_loss, l2_loss) = self.bot.evaluator.update(data)
    return LossValues(total=total_loss, policy=policy_loss, value=value_loss, l2=l2_loss)

  def self_play(self):
    for _ in range(self.num_self_play_games):
      self._self_play_single()

  def _self_play_single(self):
    state = self.game.new_initial_state()
    input_shape = [1] + self.game.observation_tensor_shape()
    policy_target, observations = [], []

    while not state.is_terminal():
      root_node = self.bot.mcts_search(state)
      # compute and store the policy and value targets obtained from MCTS
      observations.append(
          np.array(state.observation_tensor(),
                   dtype=np.float32).reshape(input_shape))
      target_policy = np.zeros(self.game.num_distinct_actions(),
                               dtype=np.float32)
      for child in root_node.children:
        target_policy[child.action] = child.explore_count
      target_policy /= sum(target_policy)
      policy_target.append(target_policy)
      # take the MCTS actions
      action = self._select_action(root_node.children, len(state.history()))
      state.apply_action(action)

    terminal_rewards = state.rewards()
    for i, (obs, pol) in enumerate(zip(observations, policy_target)):
      value = terminal_rewards[i % 2]
      self.buffer.add(
          MCTSResult(observation=obs, target_policy=pol, target_value=value))

  def _select_action(self, children, game_history_len):
    explore_counts = [(child.explore_count, child.action) for child in children]
    if game_history_len < self._action_selection_transition:
      probs = np_softmax(np.array([i[0] for i in explore_counts]))
      action_index = np.random.choice(range(len(probs)), p=probs)
      action = explore_counts[action_index][1]
    else:
      _, action = max(explore_counts)
    return action


def alpha_zero_ucb_score(child, parent_explore_count, params):
  c_init, c_base = params
  if child.outcome is not None:
    return child.outcome[child.player]

  c = math.log((parent_explore_count + c_base + 1) / c_base) + c_init
  c *= math.sqrt(parent_explore_count) / (child.explore_count + 1)

  prior_score = c * child.prior
  value_score = child.explore_count and child.total_reward / child.explore_count

  return prior_score + value_score


def np_softmax(logits):
  max_logit = np.amax(logits, axis=-1, keepdims=True)
  exp_logit = np.exp(logits - max_logit)
  return exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)


class AlphaZeroResNetEvaluator(mcts.TrainableEvaluator):
  """Implements a parameterized AlphaZero ResNet architecture used in the Science 
	paper 'A general reinforcement learning algorithm that masters chess, shogi, 
	and Go through self-play'.

  This evaluator supports games in which the input features are a 3-Tensor and
	the action space is represented by a vector. Other action representations, 
	such as the 8x8x73-Tensor representation used by AlphaZero for Chess, are
	not supported.
  """

  def __init__(self,
               input_shape,
               num_actions,
               cache_size=None,
               num_residual_blocks=19,
               num_filters=256,
               value_head_hidden_size=256,
               l2_regularization=1e-4,
               optimizer=tf.train.MomentumOptimizer(2e-2, momentum=0.9),
               device='cpu'):
    super().__init__(cache_size=cache_size)

    if len(input_shape) != 3:
      raise ValueError("The input_shape must be a rank-3 tensor.")
    self.input_shape = [1] + input_shape  # add batch dimension
    self.num_actions = num_actions

    self.l2_regularization = l2_regularization
    self.optimizer = optimizer

    if device == 'gpu':
      if not tf.test.is_gpu_available():
        raise ValueError("GPU support is unavailable.")
      self.device = tf.device("gpu:0")
    elif device == 'cpu':
      self.device = tf.device("cpu:0")
    else:
      self.device = device

    with self.device:
      inputs = tf.keras.Input(shape=input_shape, name='input')
      resnet_body = alpha_zero_resnet_body(
          inputs,
          num_filters=num_filters,
          num_residual_blocks=num_residual_blocks)
      value_head = alpha_zero_value_head(resnet_body,
                                         hidden_size=value_head_hidden_size)
      policy_head = alpha_zero_mlp_policy_head(resnet_body, num_actions)
      self.model = tf.keras.Model(inputs=inputs,
                                  outputs=[value_head, policy_head])

  def value_and_prior(self, state):
    tensor_state = np.array(state.observation_tensor(),
                            dtype=np.float32).reshape(self.input_shape)
    with self.device:
      value, policy = self.model(tensor_state)

    # renormalize policy over legal actions
    policy = np.array(policy)[0]
    mask = np.array(state.legal_actions_mask())
    policy = masked_softmax.np_masked_softmax(policy, mask)
    policy = [(action, policy[action]) for action in state.legal_actions()]

    # value is required to be array over players
    value = np.array(value)[0, 0]
    value = np.array([value, -value])

    return (value, policy)

  def update(self, training_examples):
    # As AlphaZero uses very large batch sizes that might not fit into memory,
    # this implementation uses gradient accumulation to only compute one
    # one example at a time

    observations = np.vstack([o for (o, _, _) in training_examples])
    target_values = np.vstack([v for (_, v, _) in training_examples])
    target_policies = np.vstack([p for (_, _, p) in training_examples])

    with self.device:
      with tf.GradientTape() as tape:
        values, policy_logits = self.model(observations, training=True)
        loss_value = tf.losses.mean_squared_error(
            values, tf.stop_gradient(target_values))
        loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=tf.stop_gradient(target_policies))
        loss_policy = tf.reduce_mean(loss_policy)

        loss_l2 = 0
        for weights in self.model.trainable_variables:
          loss_l2 += self.l2_regularization * tf.nn.l2_loss(weights)
        loss = loss_policy + loss_value + loss_l2

      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(
          zip(grads, self.model.trainable_variables),
          global_step=tf.train.get_or_create_global_step())

    return LossValues(total=float(loss),
                      policy=float(loss_policy),
                      value=float(loss_value),
                      l2=float(loss_l2))


# This ResNet implementation copies as closely as possible the
# description found in the Methods section of the AlphaGo Zero Nature paper.
# It is mentioned in the AlphaZero Science paper supplementary material that
# "AlphaZero uses the same network architecture as AlphaGo Zero". Whether to use
# biases in convolutions and the initialization scheme used are not mentioned.


def alpha_zero_residual_layer(inputs, num_filters, kernel_size, activation,
                              use_bias):
  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_normal',
                             use_bias=use_bias)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_normal',
                             use_bias=use_bias)(x)
  return tf.keras.layers.BatchNormalization()(x)


def alpha_zero_residual_tower(inputs, num_res_blocks, num_filters, kernel_size,
                              activation, use_bias):
  x = inputs
  for _ in range(num_res_blocks):
    y = alpha_zero_residual_layer(x, num_filters, kernel_size, activation,
                                  use_bias)
    y = alpha_zero_residual_layer(x, num_filters, kernel_size, activation,
                                  use_bias)
    x = tf.keras.layers.add([x, y])
    x = tf.keras.layers.Activation(activation)(x)

  return x


def alpha_zero_resnet_body(inputs,
                           num_residual_blocks=19,
                           num_filters=256,
                           kernel_size=3,
                           activation='relu',
                           use_bias=False):

  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_normal',
                             use_bias=use_bias)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = alpha_zero_residual_tower(x, num_residual_blocks, num_filters,
                                kernel_size, activation, use_bias)
  return x


def alpha_zero_value_head(inputs, hidden_size=256, activation='relu'):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=1,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_normal')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(hidden_size,
                            activation=activation,
                            kernel_initializer='he_normal')(x)
  x = tf.keras.layers.Dense(1,
                            activation='tanh',
                            kernel_initializer='he_normal',
                            name='value')(x)
  return x


def alpha_zero_mlp_policy_head(inputs, num_classes, activation='relu'):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=2,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_normal')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_classes,
                            kernel_initializer='he_normal',
                            name='policy')(x)
  return x
