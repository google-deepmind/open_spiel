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
"""DQN agent implemented in JAX."""

from typing import Iterable, NamedTuple, Callable, Any
from functools import partial
from enum import StrEnum

import flax.nnx as nn
import jax.numpy as jnp

import jax
import optax
import chex

import etils.epath as epath
import orbax.checkpoint as ocp

from open_spiel.python import rl_agent

ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min

class Transition(NamedTuple):
  """Data structure for the Replay buffer"""
  info_state: chex.Array
  action: chex.Array
  reward: chex.Array
  next_info_state: chex.Array
  is_final_step: chex.Array
  legal_actions_mask: chex.Array

class ReplayBufferState(NamedTuple):

  experience: chex.ArrayTree
  capacity: chex.Numeric
  entry_index: chex.Array
  is_full: bool

  def __len__(self) -> int:
    return jnp.where(self.is_full, self.capacity, self.entry_index).item()

class ReplayBuffer:
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  @staticmethod
  @partial(jax.jit, static_argnames=("capacity",))
  def init(capacity: chex.Numeric, experience: chex.ArrayTree) -> ReplayBufferState:
    # Set experience value to be empty.

    experience = jax.tree.map(jnp.empty_like, experience)
    # Broadcast to [add_batch_size, ...]
    experience = jax.tree.map(
      lambda x: jnp.broadcast_to(
        x[jnp.newaxis, ...], (capacity, *x.shape)
      ),
      experience,
    )
    return ReplayBufferState(
      capacity=capacity, experience=experience, entry_index=jnp.array(0), is_full=jnp.array(False, dtype=jnp.bool))
  
  @staticmethod    
  @partial(jax.jit, donate_argnums=(0,))
  def append(
    state: ReplayBufferState, 
    experience: chex.ArrayTree, 
  ) -> ReplayBufferState:
    """Potentially adds `experience` to the replay buffer.
    Args:
      state: `ReplayBufferState`, current state of the buffer
      experience: data to be added to the reservoir buffer.
    Returns:
      An updated `ReplayBufferState` 
    """

    chex.assert_trees_all_equal_dtypes(experience, state.experience)

    index = state.entry_index % state.capacity

    def update_leaf(buffer_leaf, exp_leaf):
      return buffer_leaf.at[index].set(exp_leaf)

    new_experience = jax.tree.map(update_leaf, state.experience, experience)

    new_entry_index = state.entry_index + 1
    new_is_full = state.is_full | (new_entry_index >= state.capacity)
    new_entry_index = new_entry_index % state.capacity

    return ReplayBufferState(
      capacity=state.capacity, 
      experience=new_experience, 
      entry_index=new_entry_index, 
      is_full=new_is_full
    )
  
  @staticmethod    
  @partial(jax.jit, static_argnames=("num_samples",))
  def sample(rng: chex.PRNGKey, state: ReplayBufferState, num_samples: int) -> Transition:
    """Returns `num_samples` uniformly sampled from the buffer.
    Args:
      rng: `chex.PRNGKey`, a random state
      state: `ReplayBufferState`, a buffer state
      num_samples: `int`, number of samples to draw.
    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      AssertionError: If there are less than `num_samples` elements in the buffer
    """

    # When full, the max time index is max_length_time_axis otherwise it is current index.
    max_size = jnp.where(state.is_full, state.capacity, state.entry_index)

    indices = jax.random.randint(
      rng, shape=(num_samples,), minval=0, maxval=max_size)
    
    return jax.tree.map(lambda x: x[indices], state.experience)


class MLP(nn.Module):
  def __init__(self,
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int,
    final_activation: Callable = lambda x: x,
    seed: int = 0
  ) -> None:

    _layers = []
    def _create_linear_block(in_features, out_features, act=nn.relu, scale=jnp.sqrt(2)):
      return nn.Sequential(
        nn.Linear(              
          in_features,
          out_features,
          kernel_init=nn.initializers.glorot_uniform(),
          rngs=nn.Rngs(seed)
        ),
        act,
      )
    # Input and Hidden layers
    for size in hidden_sizes:
      _layers.append(_create_linear_block(input_size, size, act=nn.relu))
      input_size = size
    # Output layer
    _layers.append(_create_linear_block(input_size, output_size, act=lambda x: x, scale=jnp.array(1)))
    if final_activation:
      _layers.append(final_activation)
    self.model = nn.Sequential(*_layers)

  def __call__(self, x: chex.Array) -> chex.Array:
    return self.model(x)

@nn.vmap(in_axes=(None, 0), out_axes=0)
def forward(model, x: chex.Array) -> chex.Array:
  return model(x)

class Loss(StrEnum):
  MSE="mse"
  HUBER="huber"

class Optimiser(StrEnum):
  SGD="sgd"
  RMSPROP="rmsprop"
  ADAM="adam"

class EpsilonDecaySchedule(StrEnum):
  LINEAR="linear"
  EXP="exp" 

# EPSILON DECAY SCHEDULES
def exponential_schedule(start_e: float, end_e: float, duration: float) -> Callable:
    @jax.jit
    def _call(t: int) -> float:
      return end_e + (start_e - end_e) * jnp.exp(-1. * t / duration)
    return _call

def linear_schedule(start_e: float, end_e: float, duration: int) -> Callable:
    slope = (end_e - start_e) / duration
    @jax.jit
    def _call(t: int) -> float:
      return max(slope * t + start_e, end_e)
    return _call

class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in JAX."""

  def __init__(
    self,
    player_id: int,
    state_representation_size: chex.Shape,
    num_actions: int,
    hidden_layers_sizes: Iterable[int]= (128,),
    batch_size: int=128,
    replay_buffer_class: object=ReplayBuffer,
    replay_buffer_capacity: int=10000,
    learning_rate: float=0.01,
    update_target_network_every: int=1000,
    weight_update_coeff: float = .995,
    learn_every: int=10,
    discount_factor: float=1.0,
    min_buffer_size_to_learn: int=1000,
    epsilon_start: float=1.0,
    epsilon_end: float =0.1,
    epsilon_decay_duration: int=int(1e6),
    epsilon_decay_schedule_str: EpsilonDecaySchedule = "exp",
    optimizer_str: Optimiser="sgd",
    loss_str: Loss="mse",
    huber_loss_parameter: float=1.0,
    seed: int = 42,
    gradient_clipping: float | None=None,
    allow_checkpointing: bool = True
  ) -> None:
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()
    chex.assert_type(
      [num_actions, seed, batch_size, min_buffer_size_to_learn, replay_buffer_capacity, learn_every
      ], int)
    chex.assert_type(discount_factor, float)

    self._rngkey = jax.random.key(seed)
    
    self.player_id = player_id
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor
    assert discount_factor >=0 and discount_factor <= 1

    self._replay_buffer_class = replay_buffer_class
    assert hasattr(replay_buffer_class, "init")
    assert hasattr(replay_buffer_class, "append")
    assert hasattr(replay_buffer_class, "sample")

    self._replay_buffer_capacity = int(replay_buffer_capacity)
    self._replay_buffer = None
    
    self._tau = weight_update_coeff
    assert weight_update_coeff >=0 and weight_update_coeff <= 1.0

    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._iteration = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create the Q-network instances
    self._q_network = MLP(
      state_representation_size, 
      self._layer_sizes, 
      num_actions, 
      seed=seed
    )

    self._target_q_network = MLP(
      state_representation_size, 
      self._layer_sizes,
      num_actions, 
      seed=seed
    )

    # compilation warmup
    updated_state = self._copy_weights(
      nn.state(self._q_network, nn.Param), nn.state(self._target_q_network, nn.Param), 1.
    )
    nn.update(self._target_q_network, updated_state)

    if loss_str == Loss.MSE:
      self.loss_func = jax.vmap(optax.l2_loss)
    elif loss_str == Loss.HUBER:
      self.loss_func = jax.vmap(partial(optax.huber_loss, delta=huber_loss_parameter))
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")
    
    assert (epsilon_start > 0 and epsilon_end >= 0) and (epsilon_start >= epsilon_end)
    if epsilon_decay_schedule_str == EpsilonDecaySchedule.EXP:
      self.epsilon_schedule = exponential_schedule(
        epsilon_start, epsilon_end, epsilon_decay_duration
      )
    elif epsilon_decay_schedule_str == EpsilonDecaySchedule.LINEAR:
      self.epsilon_schedule = linear_schedule(
        epsilon_start, epsilon_end, epsilon_decay_duration
      )
    else:
      raise ValueError("Not implemented, choose from 'linear', 'exp'.")

    if optimizer_str == Optimiser.ADAM:
      optimizer = optax.adam(learning_rate)
    elif optimizer_str == Optimiser.SGD:
      optimizer = optax.sgd(learning_rate)
    elif optimizer_str == Optimiser.RMSPROP:
      optimizer = optax.rmsprop(learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam', 'rmsprop', and 'sgd'.")
    
    self._checkpointer = None
    if allow_checkpointing:
      self._checkpointer = ocp.StandardCheckpointer()

    # Clipping the gradients prevent divergence and allow more stable training.
    if gradient_clipping:
      optimizer = optax.chain(
        optimizer,
        optax.clip_by_global_norm(gradient_clipping)
      )
  
    self._optimizer = nn.Optimizer(self._q_network, optimizer, wrt=nn.Param)
    
    self._graphdef_q_network_opt = nn.graphdef((self._q_network, self._optimizer))
    self._graphdef_q_network = nn.graphdef(self._q_network)
    self._graphdef_target_q_network = nn.graphdef(self._target_q_network)

    self._jittable_update = self._get_jitted_update()
    self._jittable_inference = self._get_jitted_inference()
   
  def _get_jitted_inference(self) -> Callable:
    """Get jitted Q-network inference function."""

    @jax.jit
    def infer(state: nn.State, info_state: chex.Array) -> chex.Array:
      model = nn.merge(self._graphdef_q_network, state, copy=True)
      q_values = model(info_state)
      return q_values
    
    return infer

  def _get_jitted_update(self) -> Callable:
    """Get jitted Q-network update function."""

    def _loss_fn(
        q_network: nn.Module, 
        target_q_network: nn.Module, 
        info_states: chex.Array,
        actions: chex.Array,
        rewards: chex.Array,
        next_info_states: chex.Array,
        are_final_steps: chex.Array,
        legal_actions_mask: chex.Array
      ) -> chex.Array:
        """Loss function for the Q-network."""
        q_values = forward(q_network, info_states)
        next_q_values = forward(target_q_network, next_info_states)
        next_q_values = jnp.where(
          legal_actions_mask,
          next_q_values,
          ILLEGAL_ACTION_LOGITS_PENALTY
        ).max(-1)

        targets = jax.lax.stop_gradient(
          rewards + jnp.logical_not(are_final_steps) * self._discount_factor * next_q_values)
        predictions = q_values[jnp.arange(q_values.shape[0]), actions]

        loss_values = self.loss_func(predictions, targets)

        return loss_values.mean()

    @jax.jit
    def update(
      q_network_opt_state: nn.State,
      target_q_network_state: nn.State, 
      batch: Transition
    ) -> tuple[chex.Numeric, nn.State]:
      
      q_network, optimiser = nn.merge(self._graphdef_q_network_opt, q_network_opt_state, copy=True)
      target_q_network = nn.merge(self._graphdef_target_q_network, target_q_network_state)
      
      main_loss, grads = nn.value_and_grad(_loss_fn)(
        q_network,
        target_q_network, 
        batch.info_state, 
        batch.action, 
        batch.reward, 
        batch.next_info_state,
        batch.is_final_step,
        batch.legal_actions_mask 
      )

      optimiser.update(q_network, grads)

      return main_loss, nn.state((q_network, optimiser))
  
    return update
  
  def select_action(self, time_step: chex.ArrayTree, greedy: bool = True) -> rl_agent.StepOutput:
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    action = None
    probs = []
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()
      ):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      epsilon = jnp.where(greedy, jnp.asarray(0.0), self.epsilon_schedule(self._iteration))
      action, probs = self.act_epsilon_greedy(
        info_state, 
        legal_actions, 
        epsilon
      )
  
    return rl_agent.StepOutput(action=action, probs=probs)
  
  def act_epsilon_greedy(self, info_state, legal_actions, epsilon):
    return self._act_epsilon_greedy(
        nn.state(self._q_network), 
        jnp.asarray(info_state), 
        jnp.zeros(self._num_actions, dtype=jnp.bool).at[jnp.asarray(legal_actions, dtype=jnp.int32)].set(True),  
        self._next_rng_key(), 
        epsilon
      )

  def step(self, time_step, is_evaluation=False) -> rl_agent.StepOutput:
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # rng, train_rng = jax.random.split(self._next_rng_key())
    if is_evaluation:
      self._q_network.eval()
      return self.select_action(time_step, True)
    
    self._q_network.train()
    # Act step: don't act at terminal info states or if its not our turn.
    action, probs = self.select_action(time_step, False)

    self._iteration += 1

    if self._iteration % self._learn_every == 0:
      self._last_loss_value = self.learn()

    if self._iteration % self._update_target_network_every == 0:
      updated_state = self._copy_weights(
        nn.state(self._q_network, nn.Param), nn.state(self._target_q_network, nn.Param), self._tau
      )

      nn.update(self._target_q_network, updated_state)

    if self._prev_timestep is not None and self._prev_action is not None:
      # We may omit record adding here if it's done elsewhere.
      self._add_transition(self._prev_timestep, self._prev_action, time_step)

    if time_step.last():  # prepare for the next episode.
      self._prev_timestep = None
      self._prev_action = None
      return None
    else:
      self._prev_timestep = time_step
      self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)
  
  def _add_transition(
    self, 
    prev_time_step: chex.ArrayTree | None, 
    prev_action: chex.Array | None, 
    time_step: chex.ArrayTree
  ) -> None:
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    legal_actions = jnp.asarray(time_step.observations["legal_actions"][self.player_id], dtype=jnp.int32)
    transition = Transition(
      info_state=jnp.asarray(prev_time_step.observations["info_state"][self.player_id], dtype=jnp.float32),
      action=jnp.asarray(prev_action, dtype=int),
      reward=jnp.asarray(time_step.rewards[self.player_id], dtype=jnp.float32),
      next_info_state=jnp.asarray(time_step.observations["info_state"][self.player_id], dtype=jnp.float32),
      is_final_step=jnp.asarray(time_step.last(), dtype=jnp.bool),
      legal_actions_mask=jnp.zeros(self._num_actions, dtype=jnp.bool).at[jnp.asarray(legal_actions, dtype=jnp.int32)].set(True),  
    )
    
    if self._replay_buffer is None:
      self._replay_buffer = self._replay_buffer_class.init(
        self._replay_buffer_capacity, transition
      )
      
    self._replay_buffer = self._replay_buffer_class.append(
      self._replay_buffer, transition)

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  @partial(jax.jit, static_argnums=(0,))
  def _act_epsilon_greedy(
    self, 
    network_state: nn.State,
    info_state: chex.Array, 
    legal_actions: chex.Array, 
    rng: chex.PRNGKey,
    epsilon: float
  ) -> tuple[chex.Array, chex.Array]:
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """

    q_values = self._jittable_inference(network_state, info_state)
    masked_q = jnp.where(legal_actions, q_values, ILLEGAL_ACTION_LOGITS_PENALTY)
    greedy_action = masked_q.argmax()
    
    random_probs = legal_actions / legal_actions.sum()
    greedy_probs = jax.nn.one_hot(greedy_action, self._num_actions)
    
    # Combined: (1 - eps) * greedy + (eps) * random
    mixed_probs = (1.0 - epsilon) * greedy_probs + epsilon * random_probs
    action = jax.random.choice(rng, self._num_actions, p=mixed_probs)
    
    return action, mixed_probs    
  
  def learn(self) -> float:
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (
      len(self._replay_buffer) < self._batch_size
      or len(self._replay_buffer) < self._min_buffer_size_to_learn
    ):
      return None
    
    transitions = self._replay_buffer_class.sample(
      self._next_rng_key(), self._replay_buffer, self._batch_size
    )
    q_network_state = nn.state((self._q_network, self._optimizer))
    target_q_network_state = nn.state(self._target_q_network)

    loss_val, new_state = self._jittable_update(q_network_state, target_q_network_state, transitions)
    nn.update((self._q_network, self._optimizer), new_state)
    return loss_val

  @property
  def q_values(self) -> chex.Array | None:
    return self._q_values

  @property
  def replay_buffer(self) -> Any:
    return self._replay_buffer

  @property
  def loss(self) -> float | None:
    return self._last_loss_value

  @property
  def prev_timestep(self) -> chex.Array | None:
    return self._prev_timestep

  @property
  def prev_action(self) -> int | None:
    return self._prev_action

  @property
  def step_counter(self) -> int:
    return self._iteration
  
  def _copy_weights(
    self, 
    q_network_state: nn.State, 
    target_q_network_state: nn.State, 
    tau: chex.Array
  ) -> nn.Param:
    """Soft update of the target network's weights
      θ′ ← τ θ + (1 - τ )θ′
    """
    updated_state = jax.jit(optax.incremental_update)(
      q_network_state,   
      target_q_network_state,    
      tau        
    )
    return updated_state
    
  def save(self, checkpoint_dir: epath.Path, save_optimiser: bool=True) -> None:
    """Saves the RL agent's q-network.

    Args:
      checkpoint_dir (epath.Path): directory from which checkpoints will be restored.
      save_optimiser (bool, optional): whether save only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    assert self._checkpointer, "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = epath.Path(checkpoint_dir)
    if save_optimiser:
      self._checkpointer.save(checkpoint_dir / 'optimiser', nn.state((self._q_network, self._optimizer)), force=True)
    else:
      self._checkpointer.save(checkpoint_dir / 'state', nn.state(self._q_network))
    self._checkpointer.wait_until_finished()

  def load(self, checkpoint_dir, load_optimiser: bool=True) -> None:
    """Restores the RL agent's q-network.

    Args:
      checkpoint_dir (epath.Path): directory from which checkpoints will be restored.
      load_optimiser (bool, optional): whether load only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    assert self._checkpointer, "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = epath.Path(checkpoint_dir)
    checkpoint_dir = epath.Path(checkpoint_dir)

    if load_optimiser:
      state_restored = self._checkpointer.restore(checkpoint_dir / 'optimiser', nn.state((self._q_network, self._optimizer)))
      nn.update((self._q_network, self._optimizer), state_restored)
    
    else:
      state_restored = self._checkpointer.restore(checkpoint_dir / 'state', nn.state(self._q_network))
      nn.update(self._q_network, state_restored)
    self._checkpointer.wait_until_finished()
    

