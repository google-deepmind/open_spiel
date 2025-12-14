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


from open_spiel.python import policy
import pyspiel

class AdvantageMemory(NamedTuple):
  info_state: chex.Array
  iteration: chex.Array
  advantage: chex.Array
  action: chex.Array

class StrategyMemory(NamedTuple):
  info_state: chex.Array
  iteration: chex.Array
  strategy_action_probs: chex.Array

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
    
    self._layers = []
    def _create_linear_block(in_features, out_features):
      return nn.Sequential(
        nn.Linear(in_features, out_features, kernel_init=nn.initializers.glorot_uniform(), rngs=nn.Rngs(seed)),
        nn.relu
      )
    # Input and Hidden layers
    for size in hidden_sizes:
      self._layers.append(_create_linear_block(input_size, size))
      input_size = size
    # Output layer
    self._layers.append(nn.LayerNorm(input_size, rngs=nn.Rngs(seed)))
    self._layers.append(_create_linear_block(input_size, output_size))
    if final_activation:
      self._layers.append(final_activation)
    self.model = nn.Sequential(*self._layers)

  def __call__(self, x: chex.Array, mask: chex.Array = None):
    outputs = self.model(x)
    if mask is not None:
      outputs = jnp.where(mask == 1, outputs, jnp.zeros_like(outputs))

    return outputs

  
  # NOTE: reset is done a bit differently
    
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
    seed: int = 42
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
      nn.state(self._advantage_networks[p], nn.Param)
      for p in range(self._num_players)
    ]

    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._policy_network = MLP(
      self._embedding_size,
      list(policy_network_layers),
      self._num_actions,
      nn.softmax,
      seed
    )
    self._empty_police_state = nn.state(self._policy_network, nn.Param)

    # initialise losses and grads
    self._advantage_loss = self._policy_loss = lambda pred, tgt: jax.vmap(optax.l2_loss)(pred, tgt)

    # initialise optimizers
    self._policy_opt = self._reinitialize_policy_network()
    self._advantage_opt = [
        self._reinitialize_advantage_network(p)
        for p in range(self._num_players)
    ]

    # jit param updates and matched regrets calculations
    self._jitted_matched_regrets = self._get_jitted_matched_regrets()
    self._jitted_adv_update = self._get_jitted_adv_update()
    self._jitted_policy_update = self._get_jitted_policy_update()

  def _get_jitted_adv_update(self):
    """get jitted advantage update function."""

    def _loss_adv(
        advantage_model, 
        info_states, 
        samp_regrets, 
        iterations, 
        # masks,
        total_iterations
      ):
      """Loss function for our advantage network."""
      preds = advantage_model(info_states)
      loss_values = jnp.mean(self._advantage_loss(preds, samp_regrets), axis=-1)
      loss_values = loss_values * iterations * 2 / total_iterations
      return loss_values.mean()
    
    self._adv_grads = nn.value_and_grad(_loss_adv)

    @nn.jit
    def update(
      advantage_model: nn.Module,
      optimiser: nn.Optimizer, 
      info_states: chex.Array, 
      samp_regrets: chex.Array, 
      iterations: chex.Array,
      # masks: chex.Array, 
      total_iterations: chex.Array
    ):
      main_loss, grads = self._adv_grads(
        advantage_model, info_states, samp_regrets, iterations, total_iterations)
      optimiser.update(grads)
      return advantage_model, optimiser, main_loss

    return update

  def _get_jitted_policy_update(self):
    """get jitted policy update function."""

    def _loss_policy(
      policy_model: nn.Module, 
      info_states: chex.Array, 
      action_probs: chex.Array, 
      iterations: chex.Array, 
      # masks: chex.Array, 
      total_iterations: chex.Array
    ) -> chex.Array:
      """Loss function for our policy network."""
      preds = policy_model(info_states)
      loss_values = jnp.mean(self._policy_loss(preds, action_probs), axis=-1)
      loss_values = loss_values * iterations * 2 / total_iterations
      return loss_values.mean()

    _policy_grad_fn = nn.value_and_grad(_loss_policy)

    @nn.jit
    def update(
      policy_model: nn.Module, 
      optimiser: nn.Optimizer, 
      info_states: chex.Array, 
      action_probs: chex.Array, 
      iterations: chex.Array, 
      # masks: chex.Array, 
      total_iterations: chex.Array
    ):
      main_loss, grads = _policy_grad_fn(
        policy_model, info_states, action_probs, iterations, total_iterations)
      optimiser.update(grads)
      return policy_model, optimiser, main_loss

    return update

  def _get_jitted_matched_regrets(self):
    """get jitted regret matching function."""

    @nn.jit
    def get_matched_regrets(advantage_model, info_state, legal_actions_mask):
      advs = advantage_model(info_state, legal_actions_mask)
      advantages = jnp.maximum(advs, 0)
      summed_regret = jnp.sum(advantages)
      matched_regrets = jax.lax.cond(
          summed_regret > 0, 
          lambda: advantages / summed_regret,
          lambda: jax.nn.one_hot(  # pylint: disable=g-long-lambda
              jnp.argmax(jnp.where(legal_actions_mask == 1, advs, -10e20)), 
              self._num_actions
            )
          )
      return advantages, matched_regrets

    return get_matched_regrets

  def _next_rng_key(self):
    """Get the next rng subkey from class rngkey."""
    #TODO: replace sampling with jax.random sample for reproducibility
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _reinitialize_policy_network(self) -> tuple[nn.Optimizer]:
    """Reinitalize policy network and optimizer for training."""
    nn.update(self._policy_network, self._empty_police_state)
    optimiser = nn.Optimizer(self._policy_network, optax.adam(self._learning_rate), wrt=nn.Param)
    return optimiser
  
  def _reinitialize_advantage_network(self, player):
    """Reinitalize player's advantage network and optimizer for training."""
    nn.update(self._advantage_networks[player], self._empty_advantage_states[player])
    optimiser = nn.Optimizer(self._advantage_networks[player], optax.adam(self._learning_rate), wrt=nn.Param)
    return optimiser

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def solve(self):
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for p and train from scratch.
          self._advantage_opt[p] = self._reinitialize_advantage_network(p)
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss

  def _traverse_game_tree(self, state, player):
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
      action = np.random.choice(chance_outcome, p=chance_proba)
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
          jnp.array(action)
        )
      )
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories.add(
          StrategyMemory(
          jnp.array(state.information_state_tensor(other_player)), 
          jnp.array(self._iteration),
          jnp.array(strategy)
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
    info_state = jnp.array(
        state.information_state_tensor(player), dtype=jnp.float32)
    legal_actions_mask = jnp.array(
        state.legal_actions_mask(player), dtype=jnp.bool)
    advantages, matched_regrets = self._jitted_matched_regrets(self._advantage_networks[player], info_state, legal_actions_mask)
    return advantages, matched_regrets

  def action_probabilities(self, state, player_id=None):
    """Returns action probabilities dict for a single batch."""
    del player_id  # unused
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = jnp.array(
        state.information_state_tensor(), dtype=jnp.float32)
    legal_actions_mask = jnp.array(
        state.legal_actions_mask(cur_player), dtype=jnp.bool)
    legal_actions_mask = jnp.where(
      legal_actions_mask,
      legal_actions_mask,
      1e-21
    )
    probs = self._policy_network(info_state_vector)
    return {action: probs[action] for action in legal_actions}

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      The average loss over the advantage network of the last batch.
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
      data = []
      
      for s in samples:
        data.append(
          (
            jnp.array(s.info_state),
            jnp.array(s.advantage), 
            jnp.array([s.iteration])
          )  
        )

      # Ensure some samples have been gathered.
      if not data[0]:
        print("Not enough samples gathered")
        return None
      
      data = jax.tree.map(lambda *x: jnp.stack(x), *data)

      (self._advantage_networks[player], self._advantage_opt[player],
       main_loss) = self._jitted_adv_update(
        self._advantage_networks[player],
        self._advantage_opt[player],
        *data, 
        jnp.array(self._iteration)
      )

    return main_loss

  def _learn_strategy_network(self):
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on the last training batch of transitions
      or `None`.
    """
    for _ in range(self._policy_network_train_steps):
      if self._batch_size_strategy:
        if self._batch_size_strategy > len(self._strategy_memories):
          ## Skip if there aren't enough samples
          return None
        
        samples = self._strategy_memories.sample(self._batch_size_strategy)
      else:
        samples = self._strategy_memories
      data = []
     
      for s in samples:
        data.append(
          (
            jnp.array(s.info_state),
            jnp.array(s.strategy_action_probs), 
            jnp.array([s.iteration])
          )  
        )
     
      data = jax.tree.map(lambda *x: jnp.stack(x), *data)


      (self._policy_network, self._policy_opt, main_loss) = self._jitted_policy_update(
        self._policy_network,
        self._policy_opt,
        *data, 
        jnp.array(self._iteration)
      )

    return main_loss
