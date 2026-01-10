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

"""Neural Fictitious Self-Play (NFSP) agent implemented in Jax.

The code is around 4x slower than the TF implementation at the moment. Future
PRs improving the runtime are welcome.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

from typing import NamedTuple, Iterable, Callable
from enum import Enum, StrEnum
import etils.epath as epath
from functools import partial
import contextlib


import flax.nnx as nn
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

import numpy as np
import optax
import chex

from open_spiel.python import rl_agent
from open_spiel.python.jax import dqn

class Transition(NamedTuple):
  info_state: chex.Array 
  action_probs: chex.Array 
  legal_actions_mask: chex.Array

class MODE(Enum):
  best_response=0
  average_policy=1


class Optimiser(StrEnum):
  SGD="sgd"
  RMSPROP="rmsprop"
  ADAM="adam"

class ReservoirBufferState(NamedTuple):

  experience: chex.ArrayTree
  capacity: chex.Numeric
  add_calls: chex.Array
  is_full: chex.Array

  def __len__(self) -> int:
    return min(self.add_calls, self.capacity)
  
class ReservoirBuffer:
  """Allows uniform sampling over a stream of data.
    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """   
  def __init__(self, capacity: chex.Numeric, experience: chex.ArrayTree) -> None:
    self.capacity = capacity
    self.experience = experience
    self.add_calls = jnp.array(0)

  def __len__(self) -> int:
    return min(self.add_calls.item(), self.capacity.item())

  @staticmethod
  @partial(jax.jit, static_argnames=("capacity",))
  def init(capacity: chex.Numeric, experience: chex.ArrayTree) -> ReservoirBufferState:
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
      capacity=capacity, 
      experience=experience, 
      add_calls=jnp.array(0), 
      is_full=jnp.array(False, dtype=jnp.bool)
    )
  
  @staticmethod    
  @partial(jax.jit, donate_argnums=(0,))
  def append(
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
    is_full = state.is_full | state.add_calls >= state.capacity
    write_idx = jnp.where(is_full, idx, state.add_calls)
    should_update = write_idx < state.capacity

    def update_leaf(buffer_leaf, exp_leaf):
      new_val = jnp.where(should_update, exp_leaf, buffer_leaf[write_idx])
      return buffer_leaf.at[write_idx].set(new_val)

    new_experience = jax.tree.map(update_leaf, state.experience, experience)

    return ReservoirBufferState(
      capacity=state.capacity, experience=new_experience, add_calls=state.add_calls+1, is_full=is_full)
  
  @staticmethod    
  @partial(jax.jit, static_argnames=("num_samples",))
  def sample(rng: chex.PRNGKey, state: ReservoirBufferState, num_samples: int) -> Transition:
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

    # When full, the max time index is max_length_time_axis otherwise it is current index.
    max_size = jnp.where(state.is_full, state.capacity, state.add_calls)

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
    def _create_linear_block(in_features, out_features, act=nn.relu):
      return nn.Sequential(
        nn.Linear(in_features, out_features,
          rngs=nn.Rngs(seed)),
        act,
      )
    # Input and Hidden layers
    for size in hidden_sizes:
      _layers.append(_create_linear_block(input_size, size, act=nn.relu))
      input_size = size
    # Output layer
    _layers.append(_create_linear_block(input_size, output_size, act=lambda x: x))
    if final_activation:
      _layers.append(final_activation)
    self.model = nn.Sequential(*_layers)

  def __call__(self, x: chex.Array) -> chex.Array:
    return self.model(x)

@nn.jit
@nn.vmap(in_axes=(None, 0), out_axes=0)
def forward(model, x: chex.Array) -> chex.Array:
  return model(x)

class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in JAX.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

  def __init__(
    self,
    player_id: int,
    state_representation_size: chex.Shape,
    num_actions: int,
    hidden_layers_sizes: Iterable[int],
    reservoir_buffer_capacity: int,
    anticipatory_param: float,
    replay_buffer_class: object = ReservoirBuffer,
    batch_size: int=128,
    rl_learning_rate: float=0.01,
    sl_learning_rate: float=0.01,
    min_buffer_size_to_learn: int=1000,
    learn_every: int=64,
    optimizer_str="sgd",
    seed: int = 42,
    allow_checkpointing: bool = True,
    **kwargs
  ) -> None:
    """Initialize the `NFSP` agent."""

    chex.assert_type(
      [num_actions, seed, batch_size, min_buffer_size_to_learn, reservoir_buffer_capacity, learn_every
      ], int)
    chex.assert_type(anticipatory_param, float)

    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._replay_buffer_class = replay_buffer_class
    assert hasattr(replay_buffer_class, "init")
    assert hasattr(replay_buffer_class, "append")
    assert hasattr(replay_buffer_class, "sample")

    self._reservoir_buffer_capacity = int(reservoir_buffer_capacity)
    self._reservoir_buffer = None
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._iteration = 0

    # Inner RL agent
    kwargs.update({
      "batch_size": batch_size,
      "learning_rate": rl_learning_rate,
      "learn_every": learn_every,
      "min_buffer_size_to_learn": min_buffer_size_to_learn,
      "optimizer_str": optimizer_str,
    })
    self._rl_agent = dqn.DQN(
      player_id, 
      state_representation_size,
      num_actions, 
      hidden_layers_sizes, 
      allow_checkpointing=allow_checkpointing,
      **kwargs
    )

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._sl_loss_fn = optax.softmax_cross_entropy
    self._last_sl_loss_value = None

    self._rngkey = jax.random.PRNGKey(seed)

    # Average policy network.
    self._avg_network = MLP(
      state_representation_size, 
      self._layer_sizes, 
      num_actions, 
      seed=seed+1
    )

    if optimizer_str == Optimiser.ADAM:
      optimiser = optax.adam(sl_learning_rate)
    elif optimizer_str == Optimiser.SGD:
      optimiser = optax.sgd(sl_learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")
    
    self._avg_network_optimiser = nn.Optimizer(self._avg_network, optimiser, wrt=nn.Param)
  
    self._sample_episode_policy(self._next_rng_key())
    self._jit_update = self._get_jitted_sl_upate()
    self._avg_network_inference = self._get_jitted_sl_inference()
    
    self._checkpointer = None
    if allow_checkpointing:
      self._checkpointer = ocp.StandardCheckpointer()

  @contextlib.contextmanager
  def temp_mode_as(self, mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._mode
    self._mode = mode
    yield
    self._mode = previous_mode

  def _get_jitted_sl_upate(self) -> Callable:
    """Get jitted average policy network update function.
    """
    def _loss_fn(
      avg_network: nn.Module,
      info_states: chex.Array,
      legal_actions_mask: chex.Array,
      action_probs: chex.Array
    ) -> chex.Numeric:
      
      avg_actions_logits = forward(avg_network, info_states)
      avg_actions_logits = jnp.where(
        legal_actions_mask,
        avg_actions_logits,
        jnp.full_like(avg_actions_logits, jnp.finfo(jnp.float32).min)
      )
      loss_values = self._sl_loss_fn(avg_actions_logits, action_probs)
      return loss_values.mean()

    grad_fn = nn.value_and_grad(_loss_fn)
    graphdef = nn.graphdef((self._avg_network, self._avg_network_optimiser))

    
    @jax.jit
    def update(
      avg_network_state: nn.State,
      batch: Transition
    ) -> tuple[chex.Numeric, nn.State]:
      
      avg_network, optimiser = nn.merge(graphdef, avg_network_state)

      main_loss, grads = grad_fn(
        avg_network,
        batch.info_state, 
        batch.legal_actions_mask,
        batch.action_probs,
      )
      optimiser.update(avg_network, grads)

      return main_loss, nn.state((avg_network, optimiser))
  
    return update
    
  def _get_jitted_sl_inference(self) -> Callable:
    """Get jitted average policy network inference function."""

    graphdef = nn.graphdef(self._avg_network)

    def infer(
      avg_network_state: nn.State, 
      info_state: np.ndarray, 
    ) -> tuple[chex.Array, chex.Array]:
      avg_network = nn.merge(graphdef, avg_network_state)
      action_values = avg_network(info_state)
      action_probs = nn.softmax(action_values, axis=-1)
      return action_values, action_probs

    return infer
  
  @property
  def step_counter(self) -> int:
    return self._iteration
  
  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _sample_episode_policy(self, rng: chex.PRNGKey) -> None:
    if jax.random.uniform(rng) < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  @partial(jax.jit, static_argnums=(0,))
  def _act(self, network_state: nn.State, rng: chex.PRNGKey, info_state: chex.Array, legal_actions: chex.Array):
    action_values, action_probs = self._avg_network_inference(
      network_state, info_state
    )
    # Remove illegal actions, normalize probs
    probs = jnp.where(legal_actions, action_probs, jnp.full_like(action_probs, jnp.finfo(jnp.float32).min))
    probs = nn.softmax(probs, axis=-1)
    action = jax.random.choice(rng, jnp.arange(len(probs)), p=jnp.asarray(probs))
    return action_values, action, probs

  @property
  def mode(self):
    return self._mode

  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value())

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if self._mode == MODE.best_response:
      agent_output = self._rl_agent.step(time_step, is_evaluation)
      if not is_evaluation and not time_step.last():
        self._add_transition(time_step, agent_output)

    elif self._mode == MODE.average_policy:
      # Act step: don't act at terminal info states.
      if not time_step.last():
        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        action_values, action, probs = self._act(
          nn.state(self._avg_network), 
          self._next_rng_key(), 
          jnp.asarray(info_state), 
          jax.nn.one_hot(legal_actions, self._num_actions).sum(0).astype(bool)
        )
        self._last_action_values = action_values

        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent._add_transition(self._prev_timestep, self._prev_action, time_step)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    if not is_evaluation:
      self._iteration += 1

      if self._iteration % self._learn_every == 0:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if self._mode == MODE.average_policy:
          self._rl_agent.learn()

      # Prepare for the next episode.
      if time_step.last():
        self._sample_episode_policy(self._next_rng_key())
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action
    return agent_output

  def _add_transition(self, time_step, agent_output: rl_agent.StepOutput) -> None:
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
      info_state=jnp.asarray(time_step.observations["info_state"][self.player_id], dtype=jnp.float32),
      action_probs=jnp.asarray(agent_output.probs, dtype=jnp.float32),
      legal_actions_mask=jnp.asarray(legal_actions_mask, dtype=jnp.bool)
    )
    
    if self._reservoir_buffer is None:
      self._reservoir_buffer = self._replay_buffer_class.init(self._reservoir_buffer_capacity, transition)

    self._reservoir_buffer = self._replay_buffer_class.append(
      self._reservoir_buffer, transition, self._next_rng_key()
    )


  def _learn(self) -> None:
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (len(self._reservoir_buffer) < self._batch_size or
        len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer_class.sample(self._next_rng_key(), self._reservoir_buffer, self._batch_size)
    avg_network_state = nn.state((self._avg_network, self._avg_network_optimiser))
 
    loss_val, new_state = self._jit_update(avg_network_state, transitions)
    nn.update((self._avg_network, self._avg_network_optimiser), new_state)

    return loss_val

  def save(self, checkpoint_dir: epath.Path, save_optimiser: bool = True) -> None:
    """Saves the average policy network and the inner RL agent's q-network.

    Args:
      checkpoint_dir (epath.Path): directory from which checkpoints will be restored.
      save_optimiser (bool, optional): whether save only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    assert self._checkpointer, "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    checkpoint_dir = epath.Path(checkpoint_dir)

    self._rl_agent.save(checkpoint_dir / "q_network", save_optimiser)
    if save_optimiser:
      self._checkpointer.save(checkpoint_dir / 'optimiser', nn.state((self._avg_network, self._avg_network_optimiser)), force=True)
    else:
      self._checkpointer.save(checkpoint_dir / 'state', nn.state(self._avg_network))
    self._checkpointer.wait_until_finished()


  def restore(self, checkpoint_dir: epath.Path, load_optimiser: bool = True) -> None:
    """Restores the average policy network and the inner RL agent's q-network.

    Args:
      checkpoint_dir (epath.Path): directory from which checkpoints will be restored.
      load_optimiser (bool, optional): whether load only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    assert self._checkpointer, "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    checkpoint_dir = epath.Path(checkpoint_dir)

    self._rl_agent.load(checkpoint_dir / "q_network", load_optimiser)

    if load_optimiser:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / 'optimiser', nn.state((self._avg_network, self._avg_network_optimiser))
      )
      nn.update((self._avg_network, self._avg_network_optimiser), state_restored)
    
    else:
      state_restored = self._checkpointer.restore(checkpoint_dir / 'state', nn.state(self._avg_network))
      nn.update(self._avg_network, state_restored)

    self._checkpointer.wait_until_finished()
    
