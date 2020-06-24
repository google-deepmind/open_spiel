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
the strategy profiles of the game.  To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import torch
import torch.nn as nn

from open_spiel.python import policy
import pyspiel


AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "info_state iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "info_state iteration strategy_action_probs")


def reset_params(m):
  if isinstance(m, nn.Linear):
    m.reset_parameters()


class FixedSizeRingBuffer(object):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def __init__(self, replay_buffer_capacity):
    self._replay_buffer_capacity = replay_buffer_capacity
    self._data = []
    self._next_entry_index = 0

  def add(self, element):
    """Adds `element` to the buffer.

    If the buffer is full, the oldest element will be replaced.

    Args:
      element: data to be added to the buffer.
    """
    if len(self._data) < self._replay_buffer_capacity:
      self._data.append(element)
    else:
      self._data[self._next_entry_index] = element
      self._next_entry_index += 1
      self._next_entry_index %= self._replay_buffer_capacity

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
    self._next_entry_index = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class MLP(nn.Module):
  def __init__(self, layers):
    """Initialize a MLP in PyTorch.

    Args:
      layers: (list[int]) Layer sizes.
    """
    super(MLP, self).__init__()
    layers = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers[:-1]))]
    self._model = nn.Sequential(*layers)
    
  def forward(self, input):
    return self._model(input)
  
  def reset(self):
    self._model.apply(reset_params)


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm with PyTorch.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.

  Note: batch sizes default to `None` implying that training over the full
        dataset in memory is done by default.  To sample from the memories you
        may set these values to something less than the full capacity of the
        memory.
  """

  def __init__(self,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations=100,
               num_traversals=20,
               learning_rate=1e-4,
               batch_size_advantage=None,
               batch_size_strategy=None,
               memory_capacity=int(1e6)):
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: (int) Number of training iterations.
      num_traversals: (int) Number of traversals per iteration.
      learning_rate: (float) Learning rate.
      batch_size_advantage: (int or None) Batch size to sample from advantage
        memories.
      batch_size_strategy: (int or None) Batch size to sample from strategy
        memories.
      memory_capacity: Number af samples that can be stored in memory.
    """
    all_players = list(range(game.num_players()))
    super(DeepCFR, self).__init__(game, all_players)
    self._game = game
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(
        self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1


    # Define strategy network, loss & memory.
    self._strategy_memories = FixedSizeRingBuffer(memory_capacity)
    self._policy_network = MLP([self._embedding_size] + list(policy_network_layers) + [self._num_actions])
    # Illegal actions are handled in the traversal code where expected payoff
    # and sampled regret is computed from the advantage networks.
    self._policy_sm = nn.Softmax(dim=-1)
    self._loss_policy = nn.MSELoss()
    self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(), lr=learning_rate)


    # Define advantage network, loss & memory. (One per player)
    self._advantage_memories = [
        FixedSizeRingBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._advantage_networks = [
        MLP([self._embedding_size] + list(advantage_network_layers) + [self._num_actions])
        for _ in range(self._num_players)
    ]
    self._optimizer_advantages = []
    for p in range(self._num_players):
      self._loss_advantages = nn.MSELoss(reduction='mean')
      self._optimizer_advantages.append(
          torch.optim.Adam(self._advantage_networks[p].parameters(), lr=learning_rate))

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
      self._advantage_networks[p].reset()

  def solve(self):
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        self.reinitialize_advantage_networks()
        # Re-initialize advantage networks and train from scratch.
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

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
      advantages, strategy = self._sample_action_from_advantage(state, player)
      for action in state.legal_actions():
        expected_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      for action in state.legal_actions():
        sampled_regret[action] = expected_payoff[action]
        for a_ in state.legal_actions():
          sampled_regret[action] -= strategy[a_] * expected_payoff[a_]
        self._advantage_memories[player].add(
            AdvantageMemory(state.information_state_tensor(),
                            self._iteration, advantages, action))
      return max(expected_payoff.values())
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution dor numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add(
          StrategyMemory(
              state.information_state_tensor(other_player),
              self._iteration, strategy))
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
    advantages = self._advantage_networks[player](torch.FloatTensor(np.expand_dims(info_state, axis=0)))[0].detach().numpy()
    advantages = [max(0., advantage) for advantage in advantages]
    cumulative_regret = np.sum([advantages[action] for action in legal_actions])
    matched_regrets = np.array([0.] * self._num_actions)
    for action in legal_actions:
      if cumulative_regret > 0.:
        matched_regrets[action] = advantages[action] / cumulative_regret
      else:
        matched_regrets[action] = 1 / self._num_actions
    return advantages, matched_regrets

  def action_probabilities(self, state):
    """Returns action probabilities dict for a single batch."""
    info_state_vector = np.array(state.information_state_tensor())
    if len(info_state_vector.shape) == 1:
      info_state_vector = np.expand_dims(info_state_vector, axis=0)
    probs = self._policy_sm(self._policy_network(torch.FloatTensor(info_state_vector))).detach().numpy()
    return {i: probs[0][i] for i in range(self._num_actions)}

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      The average loss over the advantage network.
    """
    if self._batch_size_advantage:
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
    advantages = torch.FloatTensor(np.array(advantages))
    iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
    outputs = self._advantage_networks[player](torch.FloatTensor(np.array(info_states)))
    self._optimizer_advantages[player].zero_grad()
    loss_advantages = self._loss_advantages(iters * outputs, iters * advantages)
    loss_advantages.backward()
    self._optimizer_advantages[player].step()
    return loss_advantages.detach().numpy()

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if self._batch_size_strategy:
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

    iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
    ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs)))
    outputs = self._policy_sm(self._policy_network(torch.FloatTensor(np.array(info_states))))
    self._optimizer_policy.zero_grad()
    loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
    loss_strategy.backward()
    self._optimizer_policy.step()
    return loss_strategy.detach().numpy()
