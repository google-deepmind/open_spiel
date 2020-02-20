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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AlphaZero Bot implemented in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import masked_softmax
from open_spiel.python.algorithms import mcts
import pyspiel

MCTSResult = collections.namedtuple("MCTSResult",
                                    "state_feature target_value target_policy")

LossValues = collections.namedtuple("LossValues", "total policy value l2")


class AlphaZero(object):
  """AlphaZero implementation.

  Follows the pseudocode AlphaZero implementation given in the paper
  DOI:10.1126/science.aar6404.
  """

  def __init__(self,
               game,
               bot,
               model,
               replay_buffer_capacity=int(1e6),
               action_selection_transition=30):
    """AlphaZero constructor.

    Args:
      game: a pyspiel.Game object
      bot: an MCTSBot object.
      model: A Model.
      replay_buffer_capacity: the size of the replay buffer in which the results
        of self-play games are stored.
      action_selection_transition: an integer representing the move number in a
        game of self-play when greedy action selection is used. Before this,
        actions are sampled from the MCTS policy.

    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    game_info = game.get_type()
    if game.num_players() != 2:
      raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
      raise ValueError("The game must be a Deterministic one, not {}".format(
          game.chance_mode))
    if (game_info.information !=
        pyspiel.GameType.Information.PERFECT_INFORMATION):
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

    self.game = game
    self.bot = bot
    self.model = model
    self.replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity)
    self.action_selection_transition = action_selection_transition

  def update(self, num_training_epochs=10, batch_size=128, verbose=False):
    """Trains the neural net.

    Randomly sampls data from the replay buffer. An update resets the optimizer
    state.

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
    num_epoch_iters = math.ceil(len(self.replay_buffer) / float(batch_size))
    losses = []
    for epoch in range(num_training_epochs):
      epoch_losses = []
      for _ in range(num_epoch_iters):
        train_data = self.replay_buffer.sample(batch_size)
        epoch_losses.append(self.model.update(train_data))

      losses.append(epoch_losses)
      if verbose:
        self._print_mean_epoch_losses(epoch, epoch_losses)

    return losses

  def self_play(self, num_self_play_games=5000):
    """Uses the current state of the net with MCTS to play full games against.

    Args:
      num_self_play_games: the number of self-play games to play using the
        current net and MCTS.
    """
    for _ in range(num_self_play_games):
      self._self_play_single()

  def _self_play_single(self):
    """Play a single game and add it to the replay buffer."""
    state = self.game.new_initial_state()
    policy_targets, state_features = [], []

    while not state.is_terminal():
      root_node = self.bot.mcts_search(state)
      state_features.append(state.observation_tensor())
      target_policy = np.zeros(self.game.num_distinct_actions(),
                               dtype=np.float32)
      for child in root_node.children:
        target_policy[child.action] = child.explore_count
      target_policy /= sum(target_policy)
      policy_targets.append(target_policy)

      action = self._select_action(root_node.children, len(state.history()))
      state.apply_action(action)

    terminal_rewards = state.rewards()
    for feature, pol in zip(state_features, policy_targets):
      self.replay_buffer.add(
          MCTSResult(state_feature=feature,
                     target_policy=pol,
                     target_value=terminal_rewards[0]))

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
    n = len(losses)
    print(("Epoch {0} mean losses. Total: {1:.3g}, Policy: {2:.3g}, "
           "Value: {3:.3g}, L2: {4:.3g}").format(
               epoch, total_loss / n, policy_loss / n, value_loss / n,
               l2_loss / n))


def np_softmax(logits):
  max_logit = np.amax(logits, axis=-1, keepdims=True)
  exp_logit = np.exp(logits - max_logit)
  return exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)


class AlphaZeroKerasEvaluator(mcts.Evaluator):
  """An AlphaZero MCTS Evaluator."""

  def __init__(self, game, model):
    """An AlphaZero MCTS Evaluator."""
    self.model = model
    self._input_shape = game.observation_tensor_shape()
    self._num_actions = game.num_distinct_actions()

  @functools.lru_cache(maxsize=2**12)
  def value_and_prior(self, state):
    # Make a singleton batch
    obs = np.expand_dims(state.observation_tensor(), 0)
    mask = np.expand_dims(state.legal_actions_mask(), 0)
    value, policy = self.model.inference(obs, mask)
    return value[0, 0], policy[0]  # Unpack batch

  def evaluate(self, state):
    value, _ = self.value_and_prior(state)
    return np.array([value, -value])

  def prior(self, state):
    _, policy = self.value_and_prior(state)
    return [(action, policy[action]) for action in state.legal_actions()]


class Model(object):
  """A wrapper around a keras model, and optimizer."""

  def __init__(self, keras_model, l2_regularization, learning_rate, device):
    """A wrapper around a keras model, and optimizer.

    Args:
      keras_model: a Keras Model object.
      l2_regularization: the amount of l2 regularization to use during training.
      learning_rate: a learning rate for the adam optimizer.
      device: The device used to run the keras_model during evaluation and
        training. Possible values are 'cpu', 'gpu', or a tf.device(...) object.
    """
    if device == "gpu":
      if not tf.test.is_gpu_available():
        raise ValueError("GPU support is unavailable.")
      self._device = tf.device("gpu:0")
    elif device == "cpu":
      self._device = tf.device("cpu:0")
    else:
      self._device = device
    self._keras_model = keras_model
    self._optimizer = tf.train.AdamOptimizer(learning_rate)
    self._l2_regularization = l2_regularization

  def inference(self, obs, mask):
    with self._device:
      value, policy = self._keras_model(obs)
    policy = masked_softmax.np_masked_softmax(np.array(policy), np.array(mask))
    return value, policy

  def update(self, training_examples):
    """Run an update step."""
    state_features = np.stack([f for (f, _, _) in training_examples])
    value_targets = np.vstack([v for (_, v, _) in training_examples])
    policy_targets = np.stack([p for (_, _, p) in training_examples])

    with self._device:
      with tf.GradientTape() as tape:
        values, policy_logits = self._keras_model(state_features, training=True)
        loss_value = tf.losses.mean_squared_error(
            values, tf.stop_gradient(value_targets))
        loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=tf.stop_gradient(policy_targets))
        loss_policy = tf.reduce_mean(loss_policy)
        loss_l2 = 0
        for weights in self._keras_model.trainable_variables:
          loss_l2 += self._l2_regularization * tf.nn.l2_loss(weights)
        loss = loss_policy + loss_value + loss_l2

      grads = tape.gradient(loss, self._keras_model.trainable_variables)

      self._optimizer.apply_gradients(
          zip(grads, self._keras_model.trainable_variables),
          global_step=tf.train.get_or_create_global_step())

    return LossValues(total=float(loss),
                      policy=float(loss_policy),
                      value=float(loss_value),
                      l2=float(loss_l2))


def cascade(x, fns):
  for fn in fns:
    x = fn(x)
  return x


def keras_resnet(input_shape,
                 num_actions,
                 num_residual_blocks=19,
                 num_filters=256,
                 value_head_hidden_size=256,
                 activation="relu",
                 data_format="channels_last"):
  """A ResNet implementation following AlphaGo Zero.

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
    num_filters: the number of convolution filters to use in the residual blocks
    value_head_hidden_size: number of hidden units in the value head dense layer
    activation: the activation function to use in the net. Does not affect the
      final tanh activation in the value head.
    data_format: Can take values 'channels_first' or 'channels_last' (default).
      Which input dimension to interpret as the channel dimension. The input
      is (1, channel, width, height) with (1, width, height, channel)
  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  def residual_layer(inputs, num_filters, kernel_size):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(axis=-1),
        lambda x: tf.keras.layers.add([x, inputs]),
        tf.keras.layers.Activation(activation),
    ])

  def resnet_body(inputs, num_filters, kernel_size):
    x = cascade(inputs, [
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
    ])
    for _ in range(num_residual_blocks):
      x = residual_layer(x, num_filters, kernel_size)
    return x

  def resnet_value_head(inputs, hidden_size):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(filters=1, kernel_size=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_size, activation),
        tf.keras.layers.Dense(1, activation="tanh", name="value"),
    ])

  def resnet_policy_head(inputs, num_classes):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(filters=2, kernel_size=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, name="policy"),
    ])

  input_size = int(np.prod(input_shape))
  inputs = tf.keras.Input(shape=input_size, name="input")
  torso = tf.keras.layers.Reshape(input_shape)(inputs)
  # Note: Keras with TensorFlow 1.15 does not support the data_format arg on CPU
  # for convolutions. Hence why this transpose is needed.
  if data_format == "channels_first":
    torso = tf.keras.backend.permute_dimensions(torso, (0, 2, 3, 1))
  torso = resnet_body(torso, num_filters, 3)
  value_head = resnet_value_head(torso, value_head_hidden_size)
  policy_head = resnet_policy_head(torso, num_actions)
  return tf.keras.Model(inputs=inputs, outputs=[value_head, policy_head])


def keras_mlp(input_shape,
              num_actions,
              num_layers=2,
              num_hidden=128,
              activation="relu"):
  """A simple MLP implementation with both a value and policy head.

  Arguments:
    input_shape: A tuple of 3 integers specifying the non-batch dimensions of
      input tensor shape.
    num_actions: The determines the output size of the policy head.
    num_layers: The number of dense layers before the policy and value heads.
    num_hidden: the number of hidden units in the dense layers.
    activation: the activation function to use in the net. Does not affect the
      final tanh activation in the value head.

  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  input_size = int(np.prod(input_shape))
  inputs = tf.keras.Input(shape=input_size, name="input")
  torso = inputs
  for _ in range(num_layers):
    torso = tf.keras.layers.Dense(num_hidden, activation=activation)(torso)
  policy = tf.keras.layers.Dense(num_actions, name="policy")(torso)
  value = tf.keras.layers.Dense(1, activation="tanh", name="value")(torso)
  return tf.keras.Model(inputs=inputs, outputs=[value, policy])
