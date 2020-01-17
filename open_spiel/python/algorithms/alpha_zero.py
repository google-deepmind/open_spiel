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
                                    "state_feature target_value target_policy")

LossValues = collections.namedtuple("LossValues", "total policy value l2")


class AlphaZero(object):
  """AlphaZero implementation following the pseudocode AlphaZero implementation
  given in the paper DOI:10.1126/science.aar6404."""

  def __init__(self,
               game,
               bot,
               replay_buffer_capacity=int(1e6),
               action_selection_transition=30,
               random_state=None):
    """
    Args:
      game: a pyspiel.Game object
      bot: an MCTSBot object.
      replay_buffer_capacity: the size of the replay buffer in which the results 
        of self-play games are stored.
      action_selection_transition: an integer representing the move number in a 
        game of self-play when greedy action selection is used. Before this,
        actions are sampled from the MCTS policy.
      batch_size: the number of examples used for a single training update. Note
        that this batch size must be small enough for the neural net training 
        update to fit into device memory.
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
    self.replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity)
    self.action_selection_transition = action_selection_transition

  def update(self, num_training_epochs=10, batch_size=128, verbose=False):
    """Trains the neural net by randomly sampling data (with replacement) from 
      the replay buffer. An update resets the optimizer state.

    Args:
      num_training_epochs: An epoch represents one pass over the training data.
        The total number training iterations this corresponds to is 
        num_training_epochs * len(replay_buffer)/batch_size.
      batch_size: the number of examples sampled from the replay buffer and 
        used for each net training iteration.
      verbose: whether to print training metrics during training.

    Returns:
      A list of length num_training_epochs. Each element of this list is 
        another list containing LossValues tuples, one for every training
        iteration.
    """
    # The AlphaZero pseudocode resets the optimizer state before training.
    optim = self.bot.evaluator.optimizer
    tf.variables_initializer(optim.variables())
    tf.train.get_or_create_global_step().assign(0)

    num_epoch_iters = math.ceil(len(self.replay_buffer) / float(batch_size))
    losses = []
    for epoch in range(num_training_epochs):
      epoch_losses = []
      for _ in range(num_epoch_iters):
        train_data = self.replay_buffer.sample(batch_size, replace=True)
        epoch_losses.append(self.bot.evaluator.update(train_data))

      losses.append(epoch_losses)
      if verbose:
        self._print_mean_epoch_losses(epoch, epoch_losses)

    return losses

  def self_play(self, num_self_play_games=5000):
    """Uses the current state of the net with MCTS to play full games against  

    Args:
      num_self_play_games: the number of self-play games to play using the 
        current net and MCTS.
    """
    for _ in range(num_self_play_games):
      self._self_play_single()

  def _self_play_single(self):
    state = self.game.new_initial_state()
    policy_targets, state_features = [], []

    while not state.is_terminal():
      root_node = self.bot.mcts_search(state)
      state_features.append(self.bot.evaluator.feature_extractor(state))
      target_policy = np.zeros(self.game.num_distinct_actions(),
                               dtype=np.float32)
      for child in root_node.children:
        target_policy[child.action] = child.explore_count
      target_policy /= sum(target_policy)
      policy_targets.append(target_policy)

      action = self._select_action(root_node.children, len(state.history()))
      state.apply_action(action)

    terminal_rewards = state.rewards()
    for i, (feature, pol) in enumerate(zip(state_features, policy_targets)):
      value = terminal_rewards[i % 2]
      self.replay_buffer.add(
          MCTSResult(state_feature=feature,
                     target_policy=pol,
                     target_value=value))

  def _select_action(self, children, game_history_len):
    explore_counts = [(child.explore_count, child.action) for child in children]
    if game_history_len < self.action_selection_transition:
      probs = np_softmax(np.array([i[0] for i in explore_counts]))
      action_index = np.random.choice(range(len(probs)), p=probs)
      action = explore_counts[action_index][1]
    else:
      _, action = max(explore_counts)
    return action

  def _print_mean_epoch_losses(self, epoch, losses):
    total_loss, policy_loss, value_loss, l2_loss = 0, 0, 0, 0
    for l in losses:
      t, p, v, l2 = l
      total_loss += t
      policy_loss += p
      value_loss += v
      l2_loss += l2

    loss_str = "Epoch {0} mean losses. Total: {1:.3g}, Policy: {2:.3g}, Value: {3:.3g}, L2: {4:.3g}"
    n = float(len(losses))
    print(
        loss_str.format(epoch, total_loss / n, policy_loss / n, value_loss / n,
                        l2_loss / n))


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


class AlphaZeroKerasEvaluator(mcts.TrainableEvaluator):

  def __init__(self,
               keras_model,
               l2_regularization=1e-4,
               optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
               device='cpu',
               feature_extractor=None,
               cache_size=None):
    """
    Args:
      keras_model: a Keras Model object.
      l2_regularization: the amount of l2 regularization to use during training.
      optimizer: a TensorFlow optimizer object.
      device: The device used to run the keras_model during evaluation and 
        training. Possible values are 'cpu', 'gpu', or a tf.device(...) object.
      feature_extractor: a function which takes as argument the game state and 
        returns a numpy tensor which the keras_model can accept as input. If 
        None, then the default features will be used, which is the 
        observation_tensor() state method, reshaped to match the keras_model 
        input shape (if possible). The keras_model is always evaluated on the
        output of this function.
        cache_size: Whether to cache the result of the net evaluation. Calling 
        the update method automatically resets the cache. Set to 0 to turn it 
        off, and None for an unbounded cache size.
    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    super().__init__(cache_size=cache_size)

    self.model = keras_model

    # TODO: validate user-supplied keras_model
    self.input_shape = list(self.model.input_shape)
    self.input_shape[0] = 1  # Keras sets the batch dim to None
    _, (_, self.num_actions) = self.model.output_shape

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

    if feature_extractor == None:
      self.feature_extractor = _create_default_feature_extractor(
          self.input_shape)
    else:
      self.feature_extractor = feature_extractor

  def value_and_prior(self, state):
    state_feature = self.feature_extractor(state)
    with self.device:
      value, policy = self.model(state_feature)

    # renormalize policy over legal actions
    policy = np.array(policy)[0]
    mask = np.array(state.legal_actions_mask())
    policy = masked_softmax.np_masked_softmax(policy, mask)
    policy = [(action, policy[action]) for action in state.legal_actions()]

    # value is required to be array over players
    value = value[0, 0].numpy()
    if state.current_player() == 0:
      values = np.array([value, -value])
    else:
      values = np.array([-value, value])

    return (values, policy)

  def update(self, training_examples):
    state_features = np.vstack([f for (f, _, _) in training_examples])
    value_targets = np.vstack([v for (_, v, _) in training_examples])
    policy_targets = np.vstack([p for (_, _, p) in training_examples])

    with self.device:
      with tf.GradientTape() as tape:
        values, policy_logits = self.model(state_features, training=True)
        loss_value = tf.losses.mean_squared_error(
            values, tf.stop_gradient(value_targets))
        loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=tf.stop_gradient(policy_targets))
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


def _create_default_feature_extractor(shape):

  def feature_extractor(state):
    obs = state.observation_tensor()
    return np.array(obs, dtype=np.float32).reshape(shape)

  return feature_extractor


def keras_resnet(input_shape,
                 num_actions,
                 num_residual_blocks=19,
                 num_filters=256,
                 value_head_hidden_size=256,
                 activation='relu',
                 data_format="channels_last"):
  """
  This ResNet implementation copies as closely as possible the
  description found in the Methods section of the AlphaGo Zero Nature paper.
  It is mentioned in the AlphaZero Science paper supplementary material that
  "AlphaZero uses the same network architecture as AlphaGo Zero". Note that
  this implementation only supports flat policy distributions.

  Arguments:
    input_shape: A tuple of 3 integers specifying the non-batch dimensions of 
      input tensor shape.
    num_actions: The determines the output size of the policy head.
    num_residual_blocks: The number of residual blocks. Can be 0.
    num_filters: the number of convolution filters to use in the residual blocks.
    value_head_hidden_size: the number of hidden units in the value head dense layer.
    activation: the activation function to use in the net. Does not affect the 
      final tanh activation in the value head.
    data_format: Can take values 'channels_first' or 'channels_last' (default).
      Which input dimension to interpret as the channel dimension. The input
      is (1, channel, width, height) with (1, width, height, channel)
  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  inputs = tf.keras.Input(shape=input_shape, name='input')
  x = inputs
  # Note: Keras with TensorFlow 1.15 does not support the data_format arg on CPU
  # for convolutions. Hence why this transpose is needed.
  if data_format == "channels_first":
    x = tf.keras.backend.permute_dimensions(inputs, (0, 2, 3, 1))
  x = _resnet_body(x,
                   num_filters=num_filters,
                   num_residual_blocks=num_residual_blocks,
                   kernel_size=3,
                   activation=activation)
  value_head = _resnet_value_head(x,
                                  hidden_size=value_head_hidden_size,
                                  activation=activation)
  policy_head = _resnet_mlp_policy_head(x, num_actions, activation=activation)
  return tf.keras.Model(inputs=inputs, outputs=[value_head, policy_head])


def _residual_layer(inputs, num_filters, kernel_size, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  return tf.keras.layers.BatchNormalization(axis=-1)(x)


def _residual_tower(inputs, num_res_blocks, num_filters, kernel_size,
                    activation):
  x = inputs
  for _ in range(num_res_blocks):
    y = _residual_layer(x, num_filters, kernel_size, activation)
    y = _residual_layer(x, num_filters, kernel_size, activation)
    x = tf.keras.layers.add([x, y])
    x = tf.keras.layers.Activation(activation)(x)

  return x


def _resnet_body(inputs, num_residual_blocks, num_filters, kernel_size,
                 activation):
  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = _residual_tower(x, num_residual_blocks, num_filters, kernel_size,
                      activation)
  return x


def _resnet_value_head(inputs, hidden_size, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=1,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(hidden_size,
                            activation=activation,
                            kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.Dense(1,
                            activation='tanh',
                            kernel_initializer='he_uniform',
                            name='value')(x)
  return x


def _resnet_mlp_policy_head(inputs, num_classes, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=2,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_classes,
                            kernel_initializer='he_uniform',
                            name='policy')(x)
  return x


def keras_mlp(input_size,
              num_actions,
              num_layers=2,
              num_hidden=128,
              activation='relu'):
  """
  A simple MLP implementation with both a value and policy head.

  Arguments:
    input_size: An integer specifying the size of the input vector.
    num_actions: The determines the output size of the policy head.
    num_layers: The number of dense layers before the policy and value heads.
    num_hidden: the number of hidden units in the dense layers.
    activation: the activation function to use in the net. Does not affect the 
      final tanh activation in the value head.

  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  inputs = tf.keras.Input(shape=(input_size,), name='input')
  x = inputs
  for _ in range(num_layers):
    x = tf.keras.layers.Dense(num_hidden,
                              kernel_initializer='he_uniform',
                              activation=activation)(x)
  policy = tf.keras.layers.Dense(num_actions,
                                 kernel_initializer='he_uniform',
                                 name='policy')(x)
  value = tf.keras.layers.Dense(1,
                                kernel_initializer='he_uniform',
                                activation='tanh',
                                name='value')(x)
  return tf.keras.Model(inputs=inputs, outputs=[value, policy])
