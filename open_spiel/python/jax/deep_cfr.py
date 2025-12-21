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

"""

import collections
from typing import Iterable, NamedTuple, Callable
from functools import partial

import jax
import optax
import chex
import flax.nnx as nn
import jax.numpy as jnp
import numpy as np
import tree as np_tree # Use dm-tree for easy pytree handling in NumPy
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel

ILLEGAL_ACTION_LOGITS_PENALTY = -1e21

class AdvantageMemory(NamedTuple):
  """Advantage network memory buffer
  """
  info_state: chex.Array
  iteration: chex.Array
  advantage: chex.Array
  legal_mask: chex.Array

class StrategyMemory(NamedTuple):
  """Strategy network memory buffer
  """
  info_state: chex.Array
  iteration: chex.Array
  strategy_action_probs: chex.Array
  legal_mask: chex.Array

class ReservoirBufferState(NamedTuple):

  experience: chex.ArrayTree
  capacity: chex.Numeric
  add_calls: chex.Array

  def __len__(self) -> int:
    return min(self.add_calls, self.capacity)
  

def get_tree_shape_prefix(tree: chex.ArrayTree, n_axes: int = 1) -> chex.Shape:
  """Get the shape of the leading axes (up to n_axes) of a pytree. This assumes all
    leaves have a common leading axes size (e.g. a common batch size).
  """
  flat_tree, tree_def = jax.tree_util.tree_flatten(tree)
  leaf = flat_tree[0]
  leading_axis_shape = leaf.shape[0:n_axes]
  chex.assert_tree_shape_prefix(tree, leading_axis_shape)
  return leading_axis_shape

class ReservoirBuffer:
    """Allows uniform sampling over a stream of data.
      See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """   
    def __init__(self, capacity: chex.Numeric, experience: chex.ArrayTree) -> None:
      self.capacity = capacity
      self.experience = experience
      self.add_calls = np.array(0)

    def __len__(self) -> int:
      return min(self.add_calls.item(), self.capacity.item())
    
    @classmethod
    def init_reservoir(cls, capacity: chex.Numeric, experience: chex.ArrayTree) -> "ReservoirBuffer":
      # Initialize buffer by replicating the structure of the experience

      _experience = np_tree.map_structure(
        lambda x: np.empty((capacity, *x.shape), dtype=x.dtype),
        experience
      )
      return cls(np.array(capacity), _experience)
    
    def to_state(self) -> ReservoirBufferState:
      return ReservoirBufferState(
        capacity=self.capacity, experience=self.experience, add_calls=self.add_calls+1
      )
    
    @staticmethod
    @partial(jax.jit, static_argnames=("capacity",))
    def init_reservoir_jitted(capacity: chex.Numeric, experience: chex.ArrayTree) -> ReservoirBufferState:
      # Set experience value to be empty.
      experience = jax.tree.map(jnp.empty_like, experience)
      # Broadcast to [add_batch_size, ...]
      experience = jax.tree.map(
        lambda x: jnp.broadcast_to(
            x[jnp.newaxis, ...], (capacity, *x.shape)
        ),
        experience,
      )
      return ReservoirBufferState(
        capacity=capacity, experience=experience, add_calls=jnp.array(0))

    def append_to_reservoir(
      self, 
      experience: chex.ArrayTree, 
      rng: chex.PRNGKey
    ) -> None:
      """Potentially adds `experience` to the reservoir buffer.
      Args:
        experience: data to be added to the reservoir buffer.
        rng: `chex.PRNGKey`, a random seed

      Returns:
        None as the method updated the buffer in-place
      """
    
      # Determine the insertion index
      # Note: count + 1 because the current item is the (count+1)-th item
      idx = jax.random.randint(rng, (), 0, self.add_calls + 1)
      
      # 2. Logic: 
      # If buffer is not full, we always add at 'count'.
      # If buffer is full, we replace at 'idx' ONLY IF idx < capacity.
      is_full = self.add_calls >= self.capacity
      write_idx = jnp.where(is_full, idx, self.add_calls)
      should_update = write_idx < self.capacity
      
      def _inplace(arr, idx, val):
        arr[idx] = val

      if should_update:
        np_tree.map_structure(
          lambda buf_leaf, exp_leaf: _inplace(buf_leaf, write_idx, exp_leaf),
          # lambda buf_leaf, exp_leaf: np.copyto(buf_leaf[write_idx], exp_leaf),
          self.experience, 
          experience
        )
          
      self.add_calls += 1

    @staticmethod    
    @partial(jax.jit, donate_argnums=(0,))
    def append_to_reservoir_jittable(
      state: ReservoirBufferState, 
      experience: chex.ArrayTree, 
      rng: chex.PRNGKey
    ) -> ReservoirBufferState:
      """Potentially adds `experience` to the reservoir buffer.

      Args:
        state: `ReservoirBufferState`, current state of the buffer
        experience: data to be added to the reservoir buffer.
        rng: `chex.PRNGKey`, a random seed

      Returns:
        An updated `ReservoirBufferState` 
      """
      
      # Note: count + 1 because the current item is the (count+1)-th item
      idx = jax.random.randint(rng, (), 0, state.add_calls + 1)
      
      # 2. Logic: 
      # If buffer is not full, we always add at 'count'.
      # If buffer is full, we replace at 'idx' ONLY IF idx < capacity.
      is_full = state.add_calls >= state.capacity
      write_idx = jnp.where(is_full, idx, state.add_calls)
      should_update = write_idx < state.capacity

      def update_leaf(buffer_leaf, exp_leaf):
        new_val = jnp.where(should_update, exp_leaf, buffer_leaf[write_idx])
        return buffer_leaf.at[write_idx].set(new_val)

      new_experience = jax.tree.map(update_leaf, state.experience, experience)

      return ReservoirBufferState(
        capacity=state.capacity, experience=new_experience, add_calls=state.add_calls+1)

    def to_jax(self) -> ReservoirBufferState:
      """Converts numpy buffer to jax.
      """

      return ReservoirBufferState(
        capacity=self.capacity, 
        experience=jax.tree.map(lambda x: jnp.array(x), self.experience), 
        add_calls=self.add_calls
      )

    @staticmethod
    def get_batch_sampler(batch_size: int | None, max_size: int) -> Callable:
      """Choose if we batch or just shuffle the data
      Args:
        num_samples: `int`, number of samples to draw, could be None if the whole buffer is used
        max_size: `int`, current size of the buffer, could be less than its capacity
      """
      @partial(jax.jit, static_argnames=("num_samples"))
      def sample(rng: chex.PRNGKey, state: ReservoirBufferState, num_samples: int):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          rng: `chex.PRNGKey`, a random state
          state: `ReservoirBufferState`, a buffer state
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          AssertionError: If there are less than `num_samples` elements in the buffer
        """
        # capacity = get_tree_shape_prefix(state.experience)[0]
        # chex.assert_equal(num_samples < capacity, True)

        indices = jax.random.choice(
          rng, jnp.arange(max_size), shape=(num_samples,), replace=False)
        return jax.tree.map(lambda x: x[indices], state.experience)

      @partial(jax.jit, static_argnames=("num_samples"))
      def shuffle_data(rng: chex.PRNGKey, state: ReservoirBufferState, num_samples: int):
        """Returns shuffled buffer along the batch axis.

        Args:
          rng: `chex.PRNGKey`, a random state
          state: `ReservoirBufferState`, a buffer state
          num_samples: `int`, number of samples to draw (UNUSED). 
          max_size: `int`, current size of the buffer, could be less than its capacity

        Returns:
          The iterable buffer.
        """
        return jax.tree.map(
          lambda arr: jax.random.permutation(rng, arr[:max_size], axis=0), state.experience
        )
      
      if batch_size is not None:
        if batch_size < max_size:
          return sample
      return shuffle_data

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

  def __call__(self, x: chex.Array, mask: chex.Array = None) -> chex.Array:
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
            seed: (int) A random seed
      print_nash_convs: (bool) print explotability for each iteration, defaults to False
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
    self._memory_capacity = int(memory_capacity)

    self._backend = jax.default_backend()

    # Initialize networks and memory buffers
    self._advantage_memories = [None] * self._num_players
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

    self._strategy_memories = None
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
    self._jittable_matched_regrets = self._get_jittable_matched_regrets()
    self._jittable_adv_update = self._get_jittable_adv_update()
    self._jittable_policy_update = self._get_jittable_policy_update()

  
  def _get_buffer_init(
      self, 
      capacity: int, 
      data: AdvantageMemory | StrategyMemory
    ) -> ReservoirBuffer | ReservoirBufferState:
      if self._backend == "cpu":
        return ReservoirBuffer.init_reservoir(capacity, data)
      data = jax.tree.map(jnp.array, data)
      return ReservoirBuffer.init_reservoir_jitted(capacity, data)

  def _get_jittable_adv_update(self) -> Callable:
    """get jittable advantage update function."""

    def update(
      advantage_model: nn.Module,
      optimiser: nn.Optimizer, 
      batch: AdvantageMemory,
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
        advantage_model, 
        batch.info_state, 
        batch.advantage, 
        batch.legal_mask, 
        batch.iteration, 
        total_iterations
      )
      optimiser.update(advantage_model, grads)
      return main_loss

    return update

  def _get_jittable_policy_update(self) -> Callable:
    """get jittable policy update function."""

    def update(
      policy_model: nn.Module, 
      optimiser: nn.Optimizer, 
      batch: StrategyMemory,
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
        # masking illegal actions and normalising
        preds = jnp.where(masks, preds, -1e-21)
        preds = nn.softmax(preds)

        loss_values = self._policy_loss(preds, action_probs) * iterations
        return loss_values.mean()

      _policy_grad_fn = nn.value_and_grad(_loss_policy)
      
      main_loss, grads = _policy_grad_fn(
        policy_model, 
        batch.info_state, 
        batch.strategy_action_probs, 
        batch.legal_mask, 
        batch.iteration, 
        total_iterations
      )
      optimiser.update(policy_model, grads)
      return main_loss

    return update

  def _get_jittable_matched_regrets(self) -> Callable:
    """get jittable regret matching function."""

    @jax.jit
    def get_matched_regrets(
      graphdef: nn.GraphDef, 
      state: nn.State, 
      info_state: chex.Array, 
      legal_actions_mask: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
      advantage_model = nn.merge(graphdef, state)
      advs = advantage_model(info_state, legal_actions_mask)
      advantages = nn.relu(advs)
      summed_regret = jnp.sum(advantages)
      matched_regrets = jnp.where(
        summed_regret > 0, 
        advantages / summed_regret,
        jax.nn.one_hot(  # pylint: disable=g-long-lambda
          jnp.argmax(jnp.where(legal_actions_mask, advs, ILLEGAL_ACTION_LOGITS_PENALTY)), 
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
  def advantage_buffers(self) -> ReservoirBufferState:
    return self._advantage_memories

  @property
  def strategy_buffer(self) -> ReservoirBufferState:
    return self._strategy_memories
  
  def _append_to_advantage_buffer(self, player: int, data: AdvantageMemory) -> None:
    if self._advantage_memories[player] is None:
      self._advantage_memories[player] = self._get_buffer_init(self._memory_capacity, data)

    if self._backend == "cpu":
      self._advantage_memories[player].append_to_reservoir(data, self._next_rng_key())
    else:
      data = jax.tree.map(jnp.array, data)
      self._advantage_memories[player] = ReservoirBuffer.append_to_reservoir_jitted(
        self._advantage_memories[player], data, self._next_rng_key())


  def _append_to_stategy_buffer(self, data: StrategyMemory) -> None:
    if self._strategy_memories is None:
      self._strategy_memories = self._get_buffer_init(self._memory_capacity, data)
    
    if self._backend == "cpu":
      self._strategy_memories.append_to_reservoir(data, self._next_rng_key())
    else:
      data = jax.tree.map(jnp.array, data)
      self._strategy_memories = ReservoirBuffer.append_to_reservoir_jitted(
        self._strategy_memories, data, self._next_rng_key())

  def solve(self) -> tuple[nn.Module, dict, chex.Numeric]:
    """Solution logic for Deep CFR."""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
      
      if self._print_nash_convs:
        policy_loss = self._learn_strategy_network()
        average_policy = policy.tabular_policy_from_callable(self._game, self.action_probabilities)
        conv = exploitability.nash_conv(self._game, average_policy)
        print(f"NashConv @ {self._iteration} = {conv} | Policy loss = {policy_loss}")
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

  def _traverse_game_tree(self, state, player: int) -> chex.Array:
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
      data = AdvantageMemory(
        np.array(state.information_state_tensor(), dtype=np.float32), 
        np.array(self._iteration, dtype=int).reshape(1,),
        np.array(samp_regret, dtype=np.float32), 
        np.array(state.legal_actions_mask(player), dtype=np.bool)
      )
      self._append_to_advantage_buffer(player, data)
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = jnp.array(strategy)
      probs /= probs.sum()
      sampled_action = jax.random.choice(self._next_rng_key(), jnp.arange(self._num_actions), p=probs)
      
      data = StrategyMemory(
        np.array(state.information_state_tensor(other_player), dtype=np.float32), 
        np.array(self._iteration, dtype=int).reshape(-1,),
        np.array(strategy, dtype=np.float32),
        np.array(state.legal_actions_mask(other_player), dtype=np.bool)
      )
      self._append_to_stategy_buffer(data)
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _sample_action_from_advantage(self, state, player: int) -> tuple[chex.Array, chex.Array]:
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.

    Returns:
      1. (jnp-array) Advantage values for info state actions indexed by action.
      2. (jnp-array) Matched regrets, prob for actions indexed by action.
    """
    self._advantage_networks[player].eval()
    info_state = jnp.array(
        state.information_state_tensor(player), dtype=jnp.float32)
    legal_actions_mask = jnp.array(
        state.legal_actions_mask(player), dtype=jnp.bool)
    
    graphdef, state = nn.split(self._advantage_networks[player])
    advantages, matched_regrets = self._jittable_matched_regrets(
      graphdef, state, info_state, legal_actions_mask)
    
    return advantages, matched_regrets

  def action_probabilities(self, state, player_id: int=None) -> dict[chex.Numeric, chex.Array]:
    """Returns action probabilities dict for a single batch."""
    del player_id  # unused
    self._policy_network.eval()
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = jnp.array(
        state.information_state_tensor(), dtype=jnp.float32)
    
    probs = self._policy_network(info_state_vector)
    legal_actions_mask = jnp.array(state.legal_actions_mask(cur_player), dtype=jnp.bool)
    probs = jnp.where(legal_actions_mask, probs, ILLEGAL_ACTION_LOGITS_PENALTY)
    probs = nn.softmax(probs)

    return {action: probs[action] for action in legal_actions}
  
  def _make_train_step(
      self, 
      update_fn: Callable, 
      batch_sampler: Callable, 
      batch_size: int
    ) -> Callable:
      """Utility method to avoid boilerplate"""
      
      def _train_step(it: chex.Array, carry: nn.Carry) -> nn.Carry:
        graphdef, state, buffer_state, rng, iteration, _ = carry

        # merge at the beginning of the function
        model, optimiser = nn.merge(graphdef, state)
        rng, _rng = jax.random.split(rng)

        batch = batch_sampler(rng, buffer_state, batch_size)

        loss = update_fn(
          model, 
          optimiser,
          batch, 
          jnp.array(iteration)
        )
        
        state = nn.state((model, optimiser))
        return (graphdef, state, buffer_state, _rng, iteration, loss)  
      
      return _train_step

  def _learn_advantage_network(self, player: int) -> chex.Numeric:
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      The average loss over the advantage network of the last batch.
    """

    self._advantage_networks[player].train()

    batch_size = self._batch_size_advantage
    max_size = min(
      self._advantage_memories[player].add_calls.item(), 
      self._advantage_memories[player].capacity.astype(int).item()
    )

    sampler_fn = ReservoirBuffer.get_batch_sampler(batch_size, max_size)

    if batch_size is not None:
      if batch_size > max_size:
        # Not enough samples to train on
        return None
      
    # Training loops are inspired by https://flax.readthedocs.io/en/stable/guides/performance.html#functional-training-loop
    graphdef, state = nn.split((self._advantage_networks[player], self._advantage_opt[player]))
    buffer_state = self._advantage_memories[player].to_jax() if self._backend == "cpu" else self._advantage_memories[player]
    update_fn = self._make_train_step(self._jittable_adv_update, sampler_fn, batch_size)
    
    (_, state, _, _, _, main_loss) = jax.lax.fori_loop(
      0, self._advantage_network_train_steps,
      update_fn,
      (graphdef, state, buffer_state, self._next_rng_key(), self._iteration, jnp.array(0))
    )
        
    nn.update((self._advantage_networks[player], self._advantage_opt[player]), state)

    return main_loss

  def _learn_strategy_network(self) -> chex.Numeric:
    """Compute the loss over the strategy network.

    Returns:
      The average loss obtained on the last training batch of transitions
      or `None`.
    """
    self._policy_network.train()

    if self._strategy_memories is None:
      return None
    
    if self._batch_size_strategy > len(self._strategy_memories):
      ## Skip if there aren't enough samples
      return None
  
    batch_size = self._batch_size_advantage
    max_size = min(
      self._strategy_memories.add_calls.item(), 
      self._strategy_memories.capacity.astype(int).item()
    )
    sampler_fn = ReservoirBuffer.get_batch_sampler(batch_size, max_size)

    # Training loops are inspired by https://flax.readthedocs.io/en/stable/guides/performance.html#functional-training-loop
    graphdef, state = nn.split((self._policy_network, self._policy_opt))
    buffer_state = self._strategy_memories.to_jax() if self._backend == "cpu" else self._strategy_memories
    update_fn = self._make_train_step(self._jittable_policy_update, sampler_fn, batch_size)
    
    (_, state, _, _, _, main_loss) = jax.lax.fori_loop(
      0, self._advantage_network_train_steps,
      update_fn,
      (graphdef, state, buffer_state, self._next_rng_key(), self._iteration, jnp.array(0))
    )
    
    nn.update((self._policy_network, self._policy_opt), state)

    return main_loss
