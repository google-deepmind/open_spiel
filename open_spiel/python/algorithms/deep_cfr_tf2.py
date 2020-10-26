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

"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game. To train these networks a reservoir buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.

This implementation uses skip connections as described in the paper if two
consecutive layers of the advantage or policy network have the same number
of units, except for the last connection. Before the last hidden layer
a layer normalization is applied.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from open_spiel.python import policy
import pyspiel


# TODO(author3) Refactor into data structures lib.
class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class SkipDense(keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.hidden = keras.layers.Dense(units, kernel_initializer='he_normal')

  def call(self, x):
    return self.hidden(x) + x


class PolicyNetwork(keras.Model):
  def __init__(self, input_size, policy_network_layers, num_actions, activation="leakyrelu", **kwargs):
    super().__init__(**kwargs)
    self._input_size = input_size
    self._num_actions = num_actions
    if activation == 'leakyrelu':
      self.activation = keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'relu':
      self.activation = keras.layers.ReLU()
    else:
      self.activation = activation

    self.softmax = keras.layers.Softmax()

    self.hidden = []
    prevunits = 0
    for units in policy_network_layers[:-1]:
      if prevunits == units:
        self.hidden.append(SkipDense(units))
      else:
        self.hidden.append(keras.layers.Dense(units, kernel_initializer='he_normal'))
      prevunits = units
    self.normalization = keras.layers.LayerNormalization()
    self.lastlayer = keras.layers.Dense(policy_network_layers[-1], kernel_initializer='he_normal')

    self.out_layer = keras.layers.Dense(num_actions)

  @tf.function
  def call(self, inputs):
    x, mask = inputs
    for layer in self.hidden:
      x = layer(x)
      x = self.activation(x)

    x = self.normalization(x)
    x = self.lastlayer(x)
    x = self.activation(x)
    x = self.out_layer(x)
    x = tf.where(mask == 1, x, -10e20)
    x = self.softmax(x)
    return x


class AdvantageNetwork(keras.Model):
  def __init__(self, input_size, adv_network_layers, num_actions, activation="leakyrelu", **kwargs):
    super().__init__(**kwargs)
    self._input_size = input_size
    self._num_actions = num_actions
    if activation == 'leakyrelu':
      self.activation = keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'relu':
      self.activation = keras.layers.ReLU()
    else:
      self.activation = activation

    self.hidden = []
    prevunits = 0
    for units in adv_network_layers[:-1]:
      if prevunits == units:
        self.hidden.append(SkipDense(units))
      else:
        self.hidden.append(keras.layers.Dense(units, kernel_initializer='he_normal'))
      prevunits = units
    self.normalization = keras.layers.LayerNormalization()
    self.lastlayer = keras.layers.Dense(adv_network_layers[-1], kernel_initializer='he_normal')

    self.out_layer = keras.layers.Dense(num_actions)

  @tf.function
  def call(self, inputs):
    x, mask = inputs
    for layer in self.hidden:
      x = layer(x)
      x = self.activation(x)

    x = self.normalization(x)
    x = self.lastlayer(x)
    x = self.activation(x)
    x = self.out_layer(x)
    x = mask * x

    return x


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.
  """

  def __init__(self,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations: int = 100,
               num_traversals: int = 100,
               learning_rate: float = 1e-3,
               batch_size_advantage: int = 2048,
               batch_size_strategy: int = 2048,
               memory_capacity: int = int(1e6),
               policy_network_train_steps: int = 5000,
               advantage_network_train_steps: int = 750,
               reinitialize_advantage_networks: bool = True):
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: Number of iterations.
      num_traversals: Number of traversals per iteration.
      learning_rate: Learning rate.
      batch_size_advantage: (int) Batch size to sample from advantage
        memories.
      batch_size_strategy: (int) Batch size to sample from strategy
        memories.
      memory_capacity: Number of samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (one
        policy training step at the end).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the
        advantage network before training on each iteration.
    """
    all_players = list(range(game.num_players()))
    super(DeepCFRSolver, self).__init__(game, all_players)
    self._game = game
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
      # `_traverse_game_tree` does not take into account this option.
      raise ValueError("Simulatenous games are not supported.")
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._policy_network_train_steps = policy_network_train_steps
    self._advantage_network_train_steps = advantage_network_train_steps
    self._policy_network_layers = policy_network_layers
    self._advantage_network_layers = advantage_network_layers
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1
    self._learning_rate = learning_rate

    self._reinitialize_policy_network()
    self._adv_networks = []
    for _ in range(self._num_players):
      self._adv_networks.append(AdvantageNetwork(self._embedding_size, self._advantage_network_layers, self._num_actions))

    self._create_memories(memory_capacity)

    self._loss_policy = keras.losses.MeanSquaredError()

    self._loss_advantages = []
    self._optimizer_advantages = []
    for _ in range(self._num_players):
      self._loss_advantages.append(keras.losses.MeanSquaredError())
      self._optimizer_advantages.append(keras.optimizers.Adam(learning_rate=learning_rate))

  def _reinitialize_policy_network(self):
    """Reinitalize policy network and optimizer."""
    self._policy_network = PolicyNetwork(self._embedding_size, self._policy_network_layers, self._num_actions)
    self._optimizer_policy = keras.optimizers.Adam(learning_rate=self._learning_rate)

  def _reinitialize_advantage_network(self, player):
    """Reinitalize player's advantage network and optimizer."""
    self._adv_networks[player] = AdvantageNetwork(self._embedding_size, self._advantage_network_layers, self._num_actions)
    self._optimizer_advantages[player] = keras.optimizers.Adam(learning_rate=self._learning_rate)

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def _create_memories(self, memory_capacity):
    """Create memory buffers and associated feature descriptions."""
    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._strategy_feature_description = {
        "info_state": tf.io.FixedLenFeature([self._embedding_size], tf.float32),
        'action_probs': tf.io.FixedLenFeature([self._num_actions], tf.float32),
        "iteration": tf.io.FixedLenFeature([1], tf.float32),
        'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
    }
    self._advantage_feature_description = {
        "info_state": tf.io.FixedLenFeature([self._embedding_size], tf.float32),
        "iteration": tf.io.FixedLenFeature([1], tf.float32),
        'samp_regret': tf.io.FixedLenFeature([self._num_actions], tf.float32),
        'action': tf.io.FixedLenFeature([1], tf.float32),
        'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
    }


  def solve(self):
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for it in range(self._num_iterations):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for player and train from scratch.
          self._reinitialize_advantage_network(p)
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def _serialize_strategy_memory(self, info_state, iteration, strategy_action_probs, legal_actions_mask):
    """Create serialized example to store a strategy entry."""
    FloatList = tf.train.FloatList
    Feature = tf.train.Feature
    example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'info_state': Feature(float_list=FloatList(value=info_state)),
                    'action_probs': Feature(float_list=FloatList(value=strategy_action_probs)),
                    'iteration': Feature(float_list=FloatList(value=[iteration])),
                    'legal_actions': Feature(float_list=FloatList(value=legal_actions_mask))
                }
            )
        )
    return example.SerializeToString()

  def _deserialize_strategy_memory(self, serialized):
    """Deserializes a batch of strategy examples for the train step."""
    tups = tf.io.parse_example(serialized, self._strategy_feature_description)
    return tups['info_state'], tups['action_probs'], tups['iteration'], tups['legal_actions']

  def _serialize_advantage_memory(self, info_state, iteration,
                          samp_regret, action, legal_actions_mask):
    """Create serialized example to store an advantage entry."""
    FloatList = tf.train.FloatList
    Feature = tf.train.Feature
    example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'info_state': Feature(float_list=FloatList(value=info_state)),
                    'iteration': Feature(float_list=FloatList(value=[iteration])),
                    'samp_regret': Feature(float_list=FloatList(value=samp_regret)),
                    'action': Feature(float_list=FloatList(value=[action])),
                    'legal_actions': Feature(float_list=FloatList(value=legal_actions_mask))
                }
            )
        )
    return tf.constant(example.SerializeToString())

  def _deserialize_advantage_memory(self, serialized):
    """Deserializes a batch of advantage examples for the train step."""
    tups = tf.io.parse_example(serialized, self._advantage_feature_description)
    return tups['info_state'], tups['samp_regret'], tups['iteration'], tups['legal_actions']

  def _traverse_game_tree(self, state, player):
    """Performs a traversal of the game tree.

    Over a traversal the advantage and strategy memories are populated with
    computed advantage values and matched regrets respectively.
    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index for this traversal.
    Returns:
      Recursively returns expected payoffs for each action.
    """
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      action = np.random.choice([i[0] for i in state.chance_outcomes()])
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      # Update the policy over the info set & actions via regret matching.
      _, strategy = self._sample_action_from_advantage(state, player)
      exp_payoff = 0 * strategy
      for action in state.legal_actions():
        exp_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      cfv = np.sum(exp_payoff * strategy)
      samp_regret = (exp_payoff - cfv) * state.legal_actions_mask(player)
      self._advantage_memories[player].add(
        self._serialize_advantage_memory(state.information_state_tensor(), self._iteration,
                          samp_regret, action, state.legal_actions_mask(player)))
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = strategy
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add(
          self._serialize_strategy_memory(state.information_state_tensor(other_player), self._iteration,
              strategy, state.legal_actions_mask(other_player)))
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _sample_action_from_advantage(self, state, player):
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.
    Returns:
      1. (np-array) Advantage values for info state actions indexed by action.
      2. (np-array) Matched regrets, prob for actions indexed by action.
    """
    info_state = tf.constant(state.information_state_tensor(player), dtype=tf.float32)
    legal_actions_mask = tf.constant(state.legal_actions_mask(player), dtype=tf.float32)
    advs = self._adv_networks[player]((tf.expand_dims(info_state, axis=0), legal_actions_mask), training=False)[0]
    advs = advs * legal_actions_mask
    advantages = tf.maximum(advs, 0)
    cumulative_regret = tf.reduce_sum(advantages)
    if cumulative_regret > 0:
      matched_regrets = advantages / cumulative_regret
    else:
      matched_regrets = tf.one_hot(tf.argmax( tf.where(legal_actions_mask==1, advs, -10e20)), self._num_actions)
    return advantages.numpy(), matched_regrets.numpy()

  def action_probabilities(self, state):
    """Returns action probabilities dict for a single batch."""
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    legal_actions_mask = tf.constant(state.legal_actions_mask(cur_player), dtype=tf.float32)
    info_state_vector = tf.constant(state.information_state_tensor(), dtype=tf.float32)
    if len(info_state_vector.shape) == 1:
      info_state_vector = tf.expand_dims(info_state_vector, axis=0)
    probs = self._policy_network((info_state_vector, legal_actions_mask), training=False)
    probs = probs.numpy()
    return {action: probs[0][action] for action in legal_actions}


  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.
    Returns:
      The average loss over the advantage network of the last batch.
    """
    @tf.function
    def train_step(info_states, advantages, iterations, masks):
      model = self._adv_networks[player]
      with tf.GradientTape() as tape:
        preds = model((info_states, masks), training=True)
        main_loss = self._loss_advantages[player](advantages, preds, sample_weight=iterations*2/self._iteration)
        loss = tf.add_n([main_loss], model.losses)
      gradients = tape.gradient(loss, model.trainable_variables)
      self._optimizer_advantages[player].apply_gradients(zip(gradients, model.trainable_variables))
      return main_loss

    if self._batch_size_strategy > len(self._advantage_memories[player]):
        ## Skip if there aren't enough samples
        return None

    random.shuffle(self._advantage_memories[player]._data)
    data = tf.data.Dataset.from_tensor_slices(self._advantage_memories[player]._data)
    data = data.shuffle(100000)
    data = data.batch(self._batch_size_advantage, drop_remainder=True)
    data = data.map(self._deserialize_advantage_memory)
    data = data.repeat()
    data = data.prefetch(10)
    
    step = 0
    for d in data:
        main_loss = train_step(*d)
        step += 1
        if step == self._advantage_network_train_steps:
            break
    return main_loss

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on the last training batch of transitions or `None`.
    """
    @tf.function
    def train_step(info_states, action_probs, iterations, masks):
      model = self._policy_network
      with tf.GradientTape() as tape:
        preds = model((info_states, masks), training=True)
        main_loss = self._loss_policy(action_probs, preds, sample_weight=iterations*2/self._iteration)
        loss = tf.add_n([main_loss], model.losses)
      gradients = tape.gradient(loss, model.trainable_variables)
      self._optimizer_policy.apply_gradients(zip(gradients, model.trainable_variables))
      return main_loss

    if self._batch_size_strategy > len(self._strategy_memories):
        ## Skip if there aren't enough samples
        return None
    
    random.shuffle(self._strategy_memories._data)
    data = tf.data.Dataset.from_tensor_slices(self._strategy_memories._data)
    data = data.shuffle(100000)
    data = data.batch(self._batch_size_strategy, drop_remainder=True)
    data = data.map(self._deserialize_strategy_memory)
    data = data.repeat()
    data = data.prefetch(10)
    
    step = 0
    for d in data:
        main_loss = train_step(*d)
        step += 1
        if step == self._policy_network_train_steps:
            break
    return main_loss
