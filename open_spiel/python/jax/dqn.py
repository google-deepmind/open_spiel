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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

# Penalty for illegal actions in action selection. In epsilon-greedy, this will
# prevent them from being selected.
ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in JAX."""

  def __init__(self,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes=128,
               replay_buffer_capacity=10000,
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=0.01,
               update_target_network_every=1000,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e6),
               optimizer_str="sgd",
               loss_str="mse",
               huber_loss_parameter=1.0,
               seed=42,
               gradient_clipping=None):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()

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
    self.huber_loss_parameter = huber_loss_parameter

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration

    # TODO(author6) Allow for optional replay buffer config.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create the Q-network instances

    def network(x):
      mlp = hk.nets.MLP(self._layer_sizes + [num_actions])
      return mlp(x)

    self.hk_network = hk.without_apply_rng(hk.transform(network))
    self.hk_network_apply = jax.jit(self.hk_network.apply)

    rng = jax.random.PRNGKey(seed)
    self._create_networks(rng, state_representation_size)

    if loss_str == "mse":
      self.loss_func = lambda x: jnp.mean(x**2)
    elif loss_str == "huber":
      # pylint: disable=g-long-lambda
      self.loss_func = lambda x: jnp.mean(
          rlax.huber_loss(x, self.huber_loss_parameter))
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    if optimizer_str == "adam":
      optimizer = optax.adam(learning_rate)
    elif optimizer_str == "sgd":
      optimizer = optax.sgd(learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    # Clipping the gradients prevent divergence and allow more stable training.
    if gradient_clipping:
      optimizer = optax.chain(optimizer,
                              optax.clip_by_global_norm(gradient_clipping))

    opt_init, opt_update = optimizer.init, optimizer.update

    self._opt_update_fn = self._get_update_func(opt_update)
    self._opt_state = opt_init(self.params_q_network)
    self._loss_and_grad = jax.value_and_grad(self._loss, has_aux=False)
    self._jit_update = jax.jit(self.get_update())

  def _create_networks(self, rng, state_representation_size):
    """Called to create the networks."""
    x = jnp.ones([1, state_representation_size])
    self.params_q_network = self.hk_network.init(rng, x)
    self.params_target_q_network = self.hk_network.init(rng, x)

  def _get_update_func(self, opt_update):

    def update(params, opt_state, gradient):
      """Learning rule (stochastic gradient descent)."""
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    return update

  def _get_action_probs(self, info_state, legal_actions, is_evaluation=False):
    """Returns a selected action and the probabilities of legal actions."""
    epsilon = self._get_epsilon(is_evaluation)
    return self._epsilon_greedy(info_state, legal_actions, epsilon)

  def step(self, time_step, is_evaluation=False, add_transition_record=True):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (time_step.is_simultaneous_move() or
                                   self.player_id
                                   == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs = self._get_action_probs(
          info_state, legal_actions, is_evaluation=is_evaluation)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_loss_value = self.learn()

      if self._step_counter % self._update_target_network_every == 0:
        # state_dict method returns a dictionary containing a whole state of the
        # module.
        self.params_target_q_network = jax.tree_multimap(
            lambda x: x.copy(), self.params_q_network)

      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        self.add_transition(self._prev_timestep, self._prev_action, time_step)

      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    legal_actions = (time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask)
    self._replay_buffer.add(transition)

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    legal_one_hot = np.zeros(self._num_actions)
    legal_one_hot[legal_actions] = 1
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      info_state = np.reshape(info_state, [1, -1])
      q_values = self.hk_network_apply(self.params_q_network, info_state)
      legal_q_values = q_values[0] + (
          1 - legal_one_hot) * ILLEGAL_ACTION_LOGITS_PENALTY
      action = int(np.argmax(legal_q_values))
      probs[action] = 1.0
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def _loss(self, param, param_target, info_states, actions, rewards,
            next_info_states, are_final_steps, legal_actions_mask):

    q_values = self.hk_network.apply(param, info_states)
    target_q_values = self.hk_network.apply(param_target, next_info_states)
    # Sum a large negative constant to illegal action logits before taking the
    # max. This prevents illegal action values from being considered as target.
    max_next_q = jnp.max(
        target_q_values +
        (1 - legal_actions_mask) * ILLEGAL_ACTION_LOGITS_PENALTY,
        axis=-1)
    max_next_q = jax.numpy.where(
        1 - are_final_steps, x=max_next_q, y=jnp.zeros_like(max_next_q))
    target = (
        rewards + (1 - are_final_steps) * self._discount_factor * max_next_q)
    target = jax.lax.stop_gradient(target)
    predictions = jnp.sum(q_values * actions, axis=-1)
    loss_value = self.loss_func(predictions - target)
    return loss_value

  def get_update(self):

    def update(param, param_target, opt_state, info_states, actions, rewards,
               next_info_states, are_final_steps, legal_actions_mask):
      loss_val, grad_val = self._loss_and_grad(param, param_target, info_states,
                                               actions, rewards,
                                               next_info_states,
                                               are_final_steps,
                                               legal_actions_mask)
      new_param, new_opt_state = self._opt_update_fn(param, opt_state, grad_val)
      return new_param, new_opt_state, loss_val

    return update

  def _to_one_hot(self, a):
    a_one_hot = np.zeros(self._num_actions)
    a_one_hot[a] = 1.0
    return a_one_hot

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer.sample(self._batch_size)
    info_states = np.asarray([t.info_state for t in transitions])
    actions = np.asarray([self._to_one_hot(t.action) for t in transitions])
    rewards = np.asarray([t.reward for t in transitions])
    next_info_states = np.asarray([t.next_info_state for t in transitions])
    are_final_steps = np.asarray([t.is_final_step for t in transitions])
    legal_actions_mask = np.asarray([t.legal_actions_mask for t in transitions])

    self.params_q_network, self._opt_state, loss_val = self._jit_update(
        self.params_q_network, self.params_target_q_network, self._opt_state,
        info_states, actions, rewards, next_info_states, are_final_steps,
        legal_actions_mask)

    return loss_val

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter
