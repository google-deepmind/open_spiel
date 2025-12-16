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

This implementation uses skip connections as described in the paper if two
consecutive layers of the advantage or policy network have the same number
of units, except for the last connection. Before the last hidden layer
a layer normalization is applied.

NOTE: the deep_cfr_jax_test.py is no longer run on github CI as TF1 is no
longer supported yet still required in this file.
"""

import collections
import random

import jax
import flax.nnx as nn
import jax.numpy as jnp
import numpy as np
import optax
import chex
from typing import Iterable, NamedTuple, Callable
from tqdm.auto import tqdm
from copy import deepcopy

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel

class AdvantageMemory(NamedTuple):
  info_state: chex.Array
  iteration: chex.Array
  advantage: chex.Array
  action: chex.Array
  legal_mask: chex.Array

class StrategyMemory(NamedTuple):
  info_state: chex.Array
  iteration: chex.Array
  strategy_action_probs: chex.Array
  legal_mask: chex.Array

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

  @property
  def data(self):
    return self._data

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
      raise ValueError('{} elements could not be sampled from size {}'.format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

  @property
  def data(self):
    return self._data

  def shuffle_data(self):
    random.shuffle(self._data)

class MLP(nn.Module):
  def __init__(self,
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int,
    final_activation: Callable = lambda x: x,
    seed: int = 0
  ) -> None:
    
    _layers = []
    def _create_linear_block(in_features, out_features, act=nn.relu):
      return nn.Sequential(
        nn.Linear(in_features, out_features,
          kernel_init=nn.initializers.glorot_uniform(),
          rngs=nn.Rngs(seed)),
        act,
      )
    # Input and Hidden layers
    for size in hidden_sizes:
      _layers.append(_create_linear_block(input_size, size, act=nn.relu))
      input_size = size
    # Output layer
    _layers.append(nn.LayerNorm(input_size, rngs=nn.Rngs(seed)))
    _layers.append(_create_linear_block(input_size, output_size, act=lambda x: x))
    if final_activation:
      _layers.append(final_activation)
    self.model = nn.Sequential(*_layers)

  def __call__(self, x: chex.Array, mask: chex.Array = None):
    outputs = self.model(x)
    if mask is not None:
      outputs *= mask
    return outputs
  
  # NOTE: reset is done a bit differently

@nn.vmap(in_axes=(None, 0, 0), out_axes=0)
def forward(model, x, mask):
  return model(x, mask)

class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.
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
    policy_network_train_steps: int = 5000,
    advantage_network_train_steps: int = 750,
    reinitialize_advantage_networks: bool = True,
    seed: int = 42, 
    print_nash_convs: bool = False
  ) -> None:
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: Number of iterations.
      num_traversals: Number of traversals per iteration.
      learning_rate: Learning rate.
      batch_size_advantage: (int) Batch size to sample from advantage memories.
      batch_size_strategy: (int) Batch size to sample from strategy memories.
      memory_capacity: Number of samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (one
        policy training iteration at the end).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the advantage
        network before training on each iteration.
      seed: A random seed
    """
    all_players = list(range(game.num_players()))
    super().__init__(game, all_players)
    
    self._game = game
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
      # `_traverse_game_tree` does not take into account this option.
      raise ValueError('Simulatenous games are not supported.')
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
    self._rngkey = jax.random.key(seed)
    self._print_nash_convs = print_nash_convs

    # Initialize networks and memory buffers
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._advantage_networks = [
      MLP(
        self._embedding_size, 
        list(advantage_network_layers),
        self._num_actions, 
        lambda x: x, 
        seed + p
      ) for p in range(self._num_players)
    ]
    self._empty_advantage_states = [
      nn.state(self._advantage_networks[p])
      for p in range(self._num_players)
    ]

    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._policy_network = MLP(
      self._embedding_size,
      list(policy_network_layers),
      self._num_actions,
      lambda x: x,
      seed
    )
    self._empty_police_state = nn.state(self._policy_network)

    # initialise losses
    self._advantage_loss = self._policy_loss = jax.vmap(optax.l2_loss)

    # initialise optimizers
    self._reinitialize_policy_network()

    self._advantage_opt = [None] * self._num_players
    for p in range(self._num_players):
      self._reinitialize_advantage_network(p)

    # jit param updates and matched regrets calculations
    self._jitted_matched_regrets = self._get_jitted_matched_regrets()
    self._jitted_adv_update = self._get_jitted_adv_update()
    self._jitted_policy_update = self._get_jitted_policy_update()

  def _get_jitted_adv_update(self) -> Callable:
    """get jitted advantage update function."""

    @nn.jit
    def update(
      advantage_model: nn.Module,
      optimiser: nn.Optimizer, 
      info_states: chex.Array, 
      samp_regrets: chex.Array,
      masks: chex.Array, 
      iterations: chex.Array,
      total_iterations: chex.Array
    ) -> chex.Array:
      
      def _loss_adv(
        advantage_model: nn.Module, 
        info_states: chex.Array, 
        samp_regrets: chex.Array, 
        masks: chex.Array,
        iterations: chex.Array, 
        total_iterations: chex.Array
      ) -> chex.Array:
        """Loss function for our advantage network."""
        preds = forward(advantage_model, info_states, masks)
        loss_values = self._advantage_loss(preds, samp_regrets) * iterations
        return loss_values.mean()
    
      self._adv_grads = nn.value_and_grad(_loss_adv)
      main_loss, grads = self._adv_grads(
        advantage_model, info_states, samp_regrets, masks, iterations, total_iterations)
      optimiser.update(advantage_model, grads)
      return main_loss

    return update

  def _get_jitted_policy_update(self) -> Callable:
    """get jitted policy update function."""

    @nn.jit
    def update(
      policy_model: nn.Module, 
      optimiser: nn.Optimizer, 
      info_states: chex.Array, 
      action_probs: chex.Array, 
      masks: chex.Array, 
      iterations: chex.Array, 
      total_iterations: chex.Array
    ) -> chex.Array:
      
      def _loss_policy(
        policy_model: nn.Module, 
        info_states: chex.Array, 
        action_probs: chex.Array, 
        masks: chex.Array,
        iterations: chex.Array, 
        total_iterations: chex.Array
      ) -> chex.Array:
        """Loss function for our policy network."""
        preds = forward(policy_model, info_states, None)
        # # masking illegal actions and normalising
        preds = jnp.where(masks, preds, -1e-21)
        preds = nn.softmax(preds)

        loss_values = self._policy_loss(preds, action_probs) * iterations
        return loss_values.mean()

      _policy_grad_fn = nn.value_and_grad(_loss_policy)
      
      main_loss, grads = _policy_grad_fn(
        policy_model, info_states, action_probs, masks, iterations, total_iterations)
      optimiser.update(policy_model, grads)
      return main_loss

    return update

  def _get_jitted_matched_regrets(self) -> Callable:
    """get jitted regret matching function."""

    @nn.jit
    def get_matched_regrets(
      advantage_model: nn.Module, 
      info_state: chex.Array, 
      legal_actions_mask: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
      advs = advantage_model(info_state, legal_actions_mask)
      advantages = nn.relu(advs)
      summed_regret = jnp.sum(advantages)
      matched_regrets = jax.lax.cond(
          summed_regret > 0, 
          lambda: advantages / summed_regret,
          lambda: jax.nn.one_hot(  # pylint: disable=g-long-lambda
              jnp.argmax(jnp.where(legal_actions_mask == 1, advs, -1e21)), 
              self._num_actions
            )
          )
      return advantages, matched_regrets

    return get_matched_regrets

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _reinitialize_policy_network(self) -> None:
    """Reinitalize policy network and optimizer for training."""
    nn.update(self._policy_network, self._empty_police_state)
    self._policy_opt = nn.Optimizer(self._policy_network, optax.adam(self._learning_rate), wrt=nn.Param)
  
  def _reinitialize_advantage_network(self, player: int) -> None:
    """Reinitalize player's advantage network and optimizer for training."""
    nn.update(self._advantage_networks[player], self._empty_advantage_states[player])
    self._advantage_opt[player] = nn.Optimizer(self._advantage_networks[player], optax.adam(self._learning_rate), wrt=nn.Param)

  @property
  def advantage_buffers(self) -> list[ReservoirBuffer]:
    return self._advantage_memories

  @property
  def strategy_buffer(self) -> ReservoirBuffer:
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def solve(self) -> tuple[nn.Module, dict, chex.Numeric]:
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in tqdm(range(self._num_iterations), total=self._num_iterations):
      
      if self._print_nash_convs:
        policy_loss = self._learn_strategy_network()
        average_policy = policy.tabular_policy_from_callable(self._game, self.action_probabilities)
        conv = exploitability.nash_conv(self._game, average_policy)
        print(f"NashConv @ {self._iteration} = {conv}")
        self._reinitialize_policy_network()

      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for p and train from scratch.
          self._reinitialize_advantage_network(p)
        advantage_losses[p].append(self._learn_advantage_network(p))
      
      self._iteration += 1

    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def _traverse_game_tree(self, state, player: int):
    """Performs a traversal of the game tree using external sampling.

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
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = jax.random.choice(self._next_rng_key(), jnp.array(chance_outcome), p=jnp.asarray(chance_proba))
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      # Update the policy over the info set & actions via regret matching.
      _, strategy = self._sample_action_from_advantage(state, player)
      strategy = np.array(strategy)
      exp_payoff = 0 * strategy
      for action in state.legal_actions():
        exp_payoff[action] = self._traverse_game_tree(
            state.child(action), player)
      cfv = np.sum(exp_payoff * strategy)
      samp_regret = (exp_payoff - cfv) * state.legal_actions_mask(player)
      self._advantage_memories[player].add(
        AdvantageMemory(
          jnp.array(state.information_state_tensor()), 
          jnp.array(self._iteration, dtype=int),
          jnp.array(samp_regret), 
          jnp.array(action),
          jnp.array(state.legal_actions_mask(player))
        )
      )
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = jnp.array(strategy)
      probs /= probs.sum()
      sampled_action = jax.random.choice(self._next_rng_key(), jnp.arange(self._num_actions), p=probs)
      self._strategy_memories.add(
          StrategyMemory(
          jnp.array(state.information_state_tensor(other_player)), 
          jnp.array(self._iteration),
          jnp.array(strategy),
          jnp.array(state.legal_actions_mask(other_player))
        )
      )
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
    self._advantage_networks[player].eval()
    info_state = jnp.array(
        state.information_state_tensor(player), dtype=jnp.float32)
    legal_actions_mask = jnp.array(
        state.legal_actions_mask(player), dtype=jnp.bool)
    advantages, matched_regrets = self._jitted_matched_regrets(
      self._advantage_networks[player], info_state, legal_actions_mask)
    return advantages, matched_regrets

  def action_probabilities(self, state, player_id: int=None):
    """Returns action probabilities dict for a single batch."""
    del player_id  # unused
    self._policy_network.eval()
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = jnp.array(
        state.information_state_tensor(), dtype=jnp.float32)
    
    probs = self._policy_network(info_state_vector)
    legal_actions_mask = jnp.array(state.legal_actions_mask(cur_player), dtype=jnp.bool)
    probs = jnp.where(legal_actions_mask==1, probs, -1e-21)
    probs = nn.softmax(probs)

    return {action: probs[action] for action in legal_actions}

  def _learn_advantage_network(self, player: int) -> chex.Numeric:
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      The average loss over the advantage network of the last batch.
    """
    
    net = deepcopy(self._advantage_networks[player])
    opt = deepcopy(self._advantage_opt[player])
    net.train()
    cached_update = nn.cached_partial(self._jitted_adv_update, net, opt)
    
    for _ in range(self._advantage_network_train_steps):

      if self._batch_size_advantage:
        if self._batch_size_advantage > len(self._advantage_memories[player]):
          ## Skip if there aren't enough samples
          return None
        samples = self._advantage_memories[player].sample(
            self._batch_size_advantage)
      else:
        samples = self._advantage_memories[player]
      data = []
      
      for s in samples:
        data.append(
          (
            jnp.array(s.info_state),
            jnp.array(s.advantage), 
            jnp.array(s.legal_mask), 
            jnp.array([s.iteration])
          )  
        )

      # Ensure some samples have been gathered.
      if not data[0]:
        print("Not enough samples gathered")
        return None
      
      data = jax.tree.map(lambda *x: jnp.stack(x), *data)

      main_loss = cached_update(
        *data, 
        jnp.array(self._iteration)
      )

    self._advantage_networks[player] = net
    self._advantage_opt[player] = opt

    return main_loss

  def _learn_strategy_network(self) -> chex.Numeric:
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on the last training batch of transitions
      or `None`.
    """

    net = deepcopy(self._policy_network)
    opt = deepcopy(self._policy_opt)
    net.train()

    cached_update = nn.cached_partial(self._jitted_policy_update, net, opt)
    if self._batch_size_strategy > len(self._strategy_memories):
      ## Skip if there aren't enough samples
      return None
    
    for _ in range(self._policy_network_train_steps):
      
      if self._batch_size_strategy:
        samples = self._strategy_memories.sample(
            self._batch_size_strategy)
      else:
        samples = self._strategy_memories
      
      data = []
      
      for s in samples:
        data.append(
          (
            jnp.array(s.info_state),
            jnp.array(s.strategy_action_probs), 
            jnp.array(s.legal_mask), 
            jnp.array([s.iteration])
          )  
        )

      # Ensure some samples have been gathered.
      if not data[0]:
        print("Not enough samples gathered")
        return None
      
      data = jax.tree.map(lambda *x: jnp.stack(x), *data)

      main_loss = cached_update(
        *data, 
        jnp.array(self._iteration)
      )

    self._policy_network = net
    self._policy_opt = opt

    return main_loss
