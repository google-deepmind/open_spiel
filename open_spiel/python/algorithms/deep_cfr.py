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

"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game. To train these networks a reservoir buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import simple_nets
import pyspiel

# Temporarily Disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "info_state iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "info_state iteration strategy_action_probs")


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


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.

  Note: batch sizes default to `None` implying that training over the full
        dataset in memory is done by default.  To sample from the memories you
        may set these values to something less than the full capacity of the
        memory.
  """

  def __init__(self,
               session,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations: int = 100,
               num_traversals: int = 20,
               learning_rate: float = 1e-4,
               batch_size_advantage=None,
               batch_size_strategy=None,
               memory_capacity: int = int(1e6),
               policy_network_train_steps: int = 1,
               advantage_network_train_steps: int = 1,
               reinitialize_advantage_networks: bool = True):
    """Initialize the Deep CFR algorithm.

    Args:
      session: (tf.Session) TensorFlow session.
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: Number of iterations.
      num_traversals: Number of traversals per iteration.
      learning_rate: Learning rate.
      batch_size_advantage: (int or None) Batch size to sample from advantage
        memories.
      batch_size_strategy: (int or None) Batch size to sample from strategy
        memories.
      memory_capacity: Number of samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (per
        iteration).
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
    self._session = session
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._policy_network_train_steps = policy_network_train_steps
    self._advantage_network_train_steps = advantage_network_train_steps
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    # TODO(author6) Allow embedding size (and network) to be specified.
    self._embedding_size = len(self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1
    self._environment_steps = 0

    # Create required TensorFlow placeholders to perform the Q-network updates.
    self._info_state_ph = tf.placeholder(
        shape=[None, self._embedding_size],
        dtype=tf.float32,
        name="info_state_ph")
    self._info_state_action_ph = tf.placeholder(
        shape=[None, self._embedding_size + 1],
        dtype=tf.float32,
        name="info_state_action_ph")
    self._action_probs_ph = tf.placeholder(
        shape=[None, self._num_actions],
        dtype=tf.float32,
        name="action_probs_ph")
    self._iter_ph = tf.placeholder(
        shape=[None, 1], dtype=tf.float32, name="iter_ph")
    self._advantage_ph = []
    for p in range(self._num_players):
      self._advantage_ph.append(
          tf.placeholder(
              shape=[None, self._num_actions],
              dtype=tf.float32,
              name="advantage_ph_" + str(p)))

    # Define strategy network, loss & memory.
    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._policy_network = simple_nets.MLP(self._embedding_size,
                                           list(policy_network_layers),
                                           self._num_actions)
    action_logits = self._policy_network(self._info_state_ph)
    # Illegal actions are handled in the traversal code where expected payoff
    # and sampled regret is computed from the advantage networks.
    self._action_probs = tf.nn.softmax(action_logits)
    self._loss_policy = tf.reduce_mean(
        tf.losses.mean_squared_error(
            labels=tf.math.sqrt(self._iter_ph) * self._action_probs_ph,
            predictions=tf.math.sqrt(self._iter_ph) * self._action_probs))
    self._optimizer_policy = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self._learn_step_policy = self._optimizer_policy.minimize(self._loss_policy)

    # Define advantage network, loss & memory. (One per player)
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._advantage_networks = [
        simple_nets.MLP(self._embedding_size, list(advantage_network_layers),
                        self._num_actions) for _ in range(self._num_players)
    ]
    self._advantage_outputs = [
        self._advantage_networks[i](self._info_state_ph)
        for i in range(self._num_players)
    ]
    self._loss_advantages = []
    self._optimizer_advantages = []
    self._learn_step_advantages = []
    for p in range(self._num_players):
      self._loss_advantages.append(
          tf.reduce_mean(
              tf.losses.mean_squared_error(
                  labels=tf.math.sqrt(self._iter_ph) * self._advantage_ph[p],
                  predictions=tf.math.sqrt(self._iter_ph) *
                  self._advantage_outputs[p])))
      self._optimizer_advantages.append(
          tf.train.AdamOptimizer(learning_rate=learning_rate))
      self._learn_step_advantages.append(self._optimizer_advantages[p].minimize(
          self._loss_advantages[p]))

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def reinitialize_advantage_networks(self):
    for p in range(self._num_players):
      self.reinitialize_advantage_network(p)

  def reinitialize_advantage_network(self, player):
    self._session.run(
        tf.group(*[
            var.initializer
            for var in self._advantage_networks[player].variables
        ]))

  def solve(self):
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for player and train from scratch.
          self.reinitialize_advantage_network(p)
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def get_environment_steps(self):
    return self._environment_steps

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
    self._environment_steps += 1
    expected_payoff = collections.defaultdict(float)
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      action = np.random.choice([i[0] for i in state.chance_outcomes()])
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      sampled_regret = collections.defaultdict(float)
      # Update the policy over the info set & actions via regret matching.
      _, strategy = self._sample_action_from_advantage(state, player)
      for action in state.legal_actions():
        expected_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      cfv = 0
      for a_ in state.legal_actions():
        cfv += strategy[a_] * expected_payoff[a_]
      for action in state.legal_actions():
        sampled_regret[action] = expected_payoff[action]
        sampled_regret[action] -= cfv
      sampled_regret_arr = [0] * self._num_actions
      for action in sampled_regret:
        sampled_regret_arr[action] = sampled_regret[action]
      self._advantage_memories[player].add(
          AdvantageMemory(state.information_state_tensor(), self._iteration,
                          sampled_regret_arr, action))
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution dor numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add(
          StrategyMemory(
              state.information_state_tensor(other_player), self._iteration,
              strategy))
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _sample_action_from_advantage(self, state, player):
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.
    Returns:
      1. (list) Advantage values for info state actions indexed by action.
      2. (list) Matched regrets, prob for actions indexed by action.
    """
    info_state = state.information_state_tensor(player)
    legal_actions = state.legal_actions(player)
    advantages_full = self._session.run(
        self._advantage_outputs[player],
        feed_dict={self._info_state_ph: np.expand_dims(info_state, axis=0)})[0]
    advantages = [max(0., advantage) for advantage in advantages_full]
    cumulative_regret = np.sum([advantages[action] for action in legal_actions])
    matched_regrets = np.array([0.] * self._num_actions)

    if cumulative_regret > 0.:
      for action in legal_actions:
        matched_regrets[action] = advantages[action] / cumulative_regret
    else:
      matched_regrets[max(legal_actions, key=lambda a: advantages_full[a])] = 1

    return advantages, matched_regrets

  def action_probabilities(self, state):
    """Returns action probabilities dict for a single batch."""
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = np.array(state.information_state_tensor())
    if len(info_state_vector.shape) == 1:
      info_state_vector = np.expand_dims(info_state_vector, axis=0)
    probs = self._session.run(
        self._action_probs, feed_dict={self._info_state_ph: info_state_vector})
    return {action: probs[0][action] for action in legal_actions}

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.
    Returns:
      The average loss over the advantage network.
    """
    for _ in range(self._advantage_network_train_steps):
      if self._batch_size_advantage:
        if self._batch_size_advantage > len(self._advantage_memories[player]):
          ## Skip if there aren't enough samples
          return None
        samples = self._advantage_memories[player].sample(
            self._batch_size_advantage)
      else:
        samples = self._advantage_memories[player]
      info_states = []
      advantages = []
      iterations = []
      for s in samples:
        info_states.append(s.info_state)
        advantages.append(s.advantage)
        iterations.append([s.iteration])
      # Ensure some samples have been gathered.
      if not info_states:
        return None

      loss_advantages, _ = self._session.run(
          [self._loss_advantages[player], self._learn_step_advantages[player]],
          feed_dict={
              self._info_state_ph: np.array(info_states),
              self._advantage_ph[player]: np.array(advantages),
              self._iter_ph: np.array(iterations),
          })
    return loss_advantages

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    for _ in range(self._policy_network_train_steps):
      if self._batch_size_strategy:
        if self._batch_size_strategy > len(self._strategy_memories):
          ## Skip if there aren't enough samples
          return None
        samples = self._strategy_memories.sample(self._batch_size_strategy)
      else:
        samples = self._strategy_memories
      info_states = []
      action_probs = []
      iterations = []
      for s in samples:
        info_states.append(s.info_state)
        action_probs.append(s.strategy_action_probs)
        iterations.append([s.iteration])

      loss_strategy, _ = self._session.run(
          [self._loss_policy, self._learn_step_policy],
          feed_dict={
              self._info_state_ph: np.array(info_states),
              self._action_probs_ph: np.array(np.squeeze(action_probs)),
              self._iter_ph: np.array(iterations),
          })
    return loss_strategy
