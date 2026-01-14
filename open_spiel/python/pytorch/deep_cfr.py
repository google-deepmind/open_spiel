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
the strategy profiles of the game.  To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

# pylint: disable=g-explicit-length-test

import collections
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import nn
import tree as np_tree

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel


def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdvantageMemory(NamedTuple):
  """Advantage network memory buffer."""

  info_state: np.ndarray
  iteration: np.ndarray
  advantage: np.ndarray


class StrategyMemory(NamedTuple):
  """Stratefy network memory buffer."""

  info_state: np.ndarray
  iteration: np.ndarray
  strategy_action_probs: np.ndarray


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(
      self,
      input_size: int,
      hidden_sizes: Iterable[int],
      output_size: int,
      final_activation: nn.Module = None,
      seed: int = 42,
  ) -> None:
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      final_activation: (nn.Module) an activation for the final later, defaults
        to None
      seed: (int) a random seed
    """

    super().__init__()
    set_seed(seed)

    _layers = []

    def _create_linear_block(in_features, out_features):
      return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    # Input and Hidden layers
    for size in hidden_sizes:
      _layers.append(_create_linear_block(input_size, size))
      input_size = size
    # Output layer
    _layers.append(nn.LayerNorm(input_size))
    _layers.append(nn.Linear(input_size, output_size))
    if final_activation:
      _layers.append(final_activation)
    self.model = nn.Sequential(*_layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)

  def reset(self):

    @torch.no_grad()
    def weight_reset(m: nn.Module):
      reset_parameters = getattr(m, "reset_parameters", None)
      if callable(reset_parameters):
        m.reset_parameters()

    self.apply(fn=weight_reset)


class ReservoirBuffer:
  """Allows uniform sampling over a stream of data.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(
      self, capacity: np.ndarray, experience: AdvantageMemory | StrategyMemory
  ) -> None:
    self.capacity = capacity
    self.experience = experience
    self.add_calls = np.array(0)

  def __len__(self) -> int:
    return min(self.add_calls.item(), self.capacity.item())

  @classmethod
  def init_reservoir(
      cls, capacity: int, experience: AdvantageMemory | StrategyMemory
  ) -> "ReservoirBuffer":
    # Initialize buffer by replicating the structure of the experience
    _experience = np_tree.map_structure(
        lambda x: np.empty((capacity, *x.shape), dtype=x.dtype), experience
    )
    return cls(np.array(capacity), _experience)

  def append_to_reservoir(
      self,
      experience: AdvantageMemory | StrategyMemory,
  ) -> None:
    """Potentially adds `experience` to the reservoir buffer.

    Args:
      experience: data to be added to the reservoir buffer.

    Returns:
      None as the method updated the buffer in-place
    """
    # Determine the insertion index
    # Note: count + 1 because the current item is the (count+1)-th item
    idx = np.random.randint(0, self.add_calls + 1)

    # 2. Logic:
    # If buffer is not full, we always add at 'count'.
    # If buffer is full, we replace at 'idx' ONLY IF idx < capacity.
    is_full = self.add_calls >= self.capacity
    write_idx = np.where(is_full, idx, self.add_calls)
    should_update = write_idx < self.capacity

    def _inplace(arr, idx, val):
      arr[idx] = val

    if should_update:
      np_tree.map_structure(
          lambda buf_leaf, exp_leaf: _inplace(buf_leaf, write_idx, exp_leaf),
          self.experience,
          experience,
      )
    self.add_calls += 1

  def sample(self, num_samples: int) -> AdvantageMemory | StrategyMemory:
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    max_size = len(self)
    if max_size < num_samples:
      raise ValueError(
          "{} elements could not be sampled from size {}".format(
              num_samples, max_size
          )
      )

    indices = np.random.choice(max_size, size=(num_samples,), replace=False)

    return np_tree.map_structure(lambda data: data[indices], self.experience)

  def shuffle(self) -> None:
    """Shuffling the reservoir buffer along the batch axis."""
    np_tree.map_structure(
        lambda x: np.random.shuffle(x[: len(self)]), self.experience
    )


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

  def __init__(
      self,
      game,
      policy_network_layers=(256, 256),
      advantage_network_layers=(128, 128),
      num_iterations: int = 100,
      num_traversals: int = 20,
      learning_rate: float = 1e-4,
      batch_size_advantage: int = None,
      batch_size_strategy: int = None,
      memory_capacity: int = int(1e6),
      policy_network_train_steps: int = 1,
      advantage_network_train_steps: int = 1,
      reinitialize_advantage_networks: bool = True,
      device: str = "cpu",
      seed: int = 42,
      print_nash_convs: bool = False,
  ) -> None:
    """Initialize the Deep CFR algorithm.

    Args:
      game: OpenSpiel game.
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
      policy_network_train_steps: Number of policy network training steps (per
        iteration).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the advantage
        network before training on each iteration.
      device: (str) A pytorch device, defaults to cpu
      seed: (int) A random seed
      print_nash_convs: (bool) print explotability for each iteration, defaults
        to False
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
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(self._root_node.information_state_tensor(0))
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._memory_capacity = int(memory_capacity)
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1
    self._learning_rate = learning_rate
    self._print_nash_convs = print_nash_convs
    self._device = torch.device(device)

    # Define advantage network, loss & memory. (One per player)
    self._advantage_memories = [None] * self._num_players
    self._advantage_networks = [
        MLP(
            self._embedding_size,
            list(advantage_network_layers),
            self._num_actions,
            None,
            seed + p,
        ).to(self._device)
        for p in range(self._num_players)
    ]

    # Define strategy network, loss & memory.
    self._strategy_memories = None
    self._policy_network = MLP(
        self._embedding_size,
        list(policy_network_layers),
        self._num_actions,
        nn.Softmax(-1),
        seed,
    ).to(self._device)

    # Illegal actions are handled in the traversal code where expected payoff
    # and sampled regret is computed from the advantage networks.

    self._loss_policy = nn.MSELoss()
    self._optimizer_policy = None
    self._reinitialize_policy_network()

    self._loss_advantages = nn.MSELoss(reduction="mean")
    self._optimizer_advantages = [None] * self._num_players
    for p in range(self._num_players):
      self._reinitialize_advantage_network(p)

  def _get_buffer_init(
      self, capacity: int, data: AdvantageMemory | StrategyMemory
  ) -> ReservoirBuffer:
    return ReservoirBuffer.init_reservoir(capacity, data)

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def _reinitialize_advantage_network(self, player):
    self._advantage_networks[player].reset()
    self._optimizer_advantages[player] = torch.optim.Adam(
        self._advantage_networks[player].parameters(), lr=self._learning_rate)

  def _reinitialize_policy_network(self):
    self._policy_network.reset()
    self._optimizer_policy = torch.optim.Adam(
        self._policy_network.parameters(), lr=self._learning_rate
    )

  def solve(self) -> tuple[nn.Module, list[list[float]], float]:
    """Solution logic for Deep CFR.

    Traverses the game tree, while storing the transitions for training
    advantage and policy networks.

    Returns:
      1. (nn.Module) Instance of the trained policy network for inference.
      2. (list of floats) Advantage network losses for
         each player during each iteration.
      3. (float) Policy loss.
    """
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
      if self._print_nash_convs:
        policy_loss = self._learn_strategy_network()
        average_policy = policy.tabular_policy_from_callable(
            self._game, self.action_probabilities
        )
        conv = exploitability.nash_conv(self._game, average_policy)
        print(
            f"NashConv @ {self._iteration} = {conv} | Policy loss ="
            f" {policy_loss}"
        )
        self._reinitialize_policy_network()

      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for player and train from scratch.
          self._reinitialize_advantage_network(p)
        # Re-initialize advantage networks and train from scratch.
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def _append_to_stategy_buffer(self, data: StrategyMemory) -> None:
    if self._strategy_memories is None:
      self._strategy_memories = self._get_buffer_init(
          self._memory_capacity, data
      )

    self._strategy_memories.append_to_reservoir(data)

  def _append_to_advantage_buffer(
      self, player: int, data: AdvantageMemory
  ) -> None:
    if self._advantage_memories[player] is None:
      self._advantage_memories[player] = self._get_buffer_init(
          self._memory_capacity, data
      )

    self._advantage_memories[player].append_to_reservoir(data)

  def _traverse_game_tree(self, state, player):
    """Performs a traversal of the game tree.

    Over a traversal the advantage and strategy memories are populated with
    computed advantage values and matched regrets respectively.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index for this traversal.

    Returns:
      (float) Recursively returns expected payoffs for each action.
    """
    expected_payoff = collections.defaultdict(float)
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = np.random.choice(chance_outcome, p=chance_proba)
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

      data = AdvantageMemory(
          np.array(state.information_state_tensor(), dtype=np.float32),
          np.array(self._iteration, dtype=int).reshape(
              1,
          ),
          np.array(sampled_regret_arr, dtype=np.float32),
      )

      self._append_to_advantage_buffer(player, data)
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)

      data = StrategyMemory(
          np.array(
              state.information_state_tensor(other_player), dtype=np.float32
          ),
          np.array(self._iteration, dtype=int).reshape(
              1,
          ),
          np.array(strategy, dtype=np.float32),
      )
      self._append_to_stategy_buffer(data)

      return self._traverse_game_tree(state.child(sampled_action), player)

  @torch.inference_mode()
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
    with torch.no_grad():
      state_tensor = torch.FloatTensor(
          np.expand_dims(info_state, axis=0), device=self._device
      )
      raw_advantages = (
          self._advantage_networks[player](state_tensor)[0].cpu().numpy()
      )
    advantages = [max(0., advantage) for advantage in raw_advantages]
    cumulative_regret = np.sum([advantages[action] for action in legal_actions])
    matched_regrets = np.array([0.] * self._num_actions)
    if cumulative_regret > 0.:
      for action in legal_actions:
        matched_regrets[action] = advantages[action] / cumulative_regret
    else:
      matched_regrets[max(legal_actions, key=lambda a: raw_advantages[a])] = 1
    return advantages, matched_regrets

  @torch.inference_mode()
  def action_probabilities(self, state, player_id=None):
    """Computes action probabilities for the current player in state.

    Args:
      state: (pyspiel.State) The state to compute probabilities for.
      player_id: unused, but needed to implement the Policy API.

    Returns:
      (dict) action probabilities for a single batch.
    """
    del player_id
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = np.array(state.information_state_tensor())
    if len(info_state_vector.shape) == 1:
      info_state_vector = np.expand_dims(info_state_vector, axis=0)
    probs = (
        self._policy_network(
            torch.FloatTensor(info_state_vector, device=self._device)
        )
        .cpu()
        .numpy()
    )
    return {action: probs[0][action] for action in legal_actions}

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      (float) The average loss over the advantage network.
    """
    for _ in range(self._advantage_network_train_steps):

      if self._batch_size_advantage:
        if self._batch_size_advantage > len(self._advantage_memories[player]):
          ## Skip if there aren't enough samples
          return None
        samples = self._advantage_memories[player].sample(
            self._batch_size_advantage)
      else:
        self._advantage_memories[player].shuffle()
        samples = self._advantage_memories[player].experience

      # Ensure some samples have been gathered.
      if len(samples.info_state == 0):
        return None

      self._optimizer_advantages[player].zero_grad()
      iters = torch.FloatTensor(samples.iteration, device=self._device).sqrt()
      outputs = self._advantage_networks[player](
          torch.FloatTensor(samples.info_state, device=self._device)
      )
      advantages = torch.FloatTensor(samples.advantage, device=self._device)
      loss_advantages = self._loss_advantages(
          iters * outputs, iters * advantages
      )
      loss_advantages.backward()
      self._optimizer_advantages[player].step()

    return loss_advantages.detach().cpu().item()

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      (float) The average loss obtained on this batch of transitions or `None`.
    """
    if self._strategy_memories is None:
      return None

    for _ in range(self._policy_network_train_steps):
      if self._batch_size_strategy:
        if self._batch_size_strategy > len(self._strategy_memories):
          ## Skip if there aren't enough samples
          return None
        samples = self._strategy_memories.sample(self._batch_size_strategy)
      else:
        self._strategy_memories.shuffle()
        samples = self._strategy_memories.experience

      # Ensure some samples have been gathered.

      if len(samples.info_state) == 0:
        return None

      self._optimizer_policy.zero_grad()
      iters = torch.FloatTensor(samples.iteration, device=self._device).sqrt()
      outputs = self._policy_network(
          torch.FloatTensor(samples.info_state, device=self._device)
      )
      ac_probs = torch.FloatTensor(
          samples.strategy_action_probs, device=self._device
      ).squeeze()
      loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
      loss_strategy.backward()
      self._optimizer_policy.step()

    return loss_strategy.detach().cpu().item()
