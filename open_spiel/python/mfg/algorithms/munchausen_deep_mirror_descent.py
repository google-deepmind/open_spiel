# Copyright 2022 DeepMind Technologies Limited
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

# TODO(sertan): Add link to the reference paper.
"""Munchausen DQN Agent and deep online mirror descent implementation."""

import collections
from typing import Any, Callable, Dict, Optional

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from open_spiel.python import rl_agent
from open_spiel.python import rl_agent_policy
from open_spiel.python.mfg.algorithms import distribution as distribution_std
from open_spiel.python.utils.replay_buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action legal_one_hots reward next_info_state is_final_step "
    "next_legal_one_hots")

# Penalty for illegal actions in action selection. In epsilon-greedy, this will
# prevent them from being selected and in soft-max the probabilities will be
# (close to) 0.
ILLEGAL_ACTION_PENALTY = -1e9
# Lower bound for action probabilities to prevent NaNs in log terms.
MIN_ACTION_PROB = 1e-6


def _copy_params(params):
  """Returns a copy of the params."""
  return jax.tree_multimap(lambda x: x.copy(), params)


class MunchausenDQN(rl_agent.AbstractAgent):
  """Munchausen DQN Agent implementation in JAX."""

  def __init__(
      self,
      player_id,
      state_representation_size,
      num_actions,
      # Training options.
      batch_size: int = 128,
      learn_every: int = 64,
      epsilon_start: float = 0.1,
      epsilon_end: float = 0.1,
      epsilon_decay_duration: int = int(20e6),
      epsilon_power: float = 1.0,
      discount_factor: float = 1.0,
      # Replay buffer options.
      replay_buffer_capacity: int = int(2e5),
      min_buffer_size_to_learn: int = 1000,
      replay_buffer_class=ReplayBuffer,
      # Loss and optimizer options.
      optimizer: str = "sgd",
      learning_rate: float = 0.01,
      loss: str = "mse",
      huber_loss_parameter: float = 1.0,
      # Network options.
      update_target_network_every: int = 19200,
      hidden_layers_sizes=128,
      qnn_params_init=None,
      # Munchausen options.
      tau=0.05,
      alpha=0.9,
      reset_replay_buffer_on_update: bool = True,
      gradient_clipping: Optional[float] = None,
      with_munchausen: bool = True,
      seed: int = 42):
    """Initialize the Munchausen DQN agent."""
    self.player_id = int(player_id)
    self._num_actions = num_actions

    self._batch_size = batch_size
    self._learn_every = learn_every
    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration
    self._epsilon_power = epsilon_power
    self._discount_factor = discount_factor
    self._reset_replay_buffer_on_update = reset_replay_buffer_on_update

    self._tau = tau
    self._alpha = alpha

    # If true, the target uses Munchausen penalty terms.
    self._with_munchausen = with_munchausen

    self._prev_action = None
    self._prev_legal_action = None
    self._prev_time_step = None

    # Used to select actions.
    self._rs = np.random.RandomState(seed)

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create the replay buffer.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    # Create the Q-network.
    self._update_target_network_every = update_target_network_every

    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]

    def network(x):
      mlp = hk.nets.MLP(hidden_layers_sizes + [num_actions])
      return mlp(x)

    self.hk_network = hk.without_apply_rng(hk.transform(network))
    self.hk_network_apply = jax.jit(self.hk_network.apply)

    if qnn_params_init:
      self._params_q_network = _copy_params(qnn_params_init)
      self._params_target_q_network = _copy_params(qnn_params_init)
      self._params_prev_q_network = _copy_params(qnn_params_init)
    else:
      rng = jax.random.PRNGKey(seed)
      x = jnp.ones([1, state_representation_size])
      self._params_q_network = self.hk_network.init(rng, x)
      self._params_target_q_network = self.hk_network.init(rng, x)
      self._params_prev_q_network = self.hk_network.init(rng, x)

    # Create the loss function and the optimizer.
    if loss == "mse":
      self._loss_func = lambda x: jnp.mean(x**2)
    elif loss == "huber":
      self._loss_func = lambda x: jnp.mean(  # pylint: disable=g-long-lambda
          rlax.huber_loss(x, huber_loss_parameter))
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    if optimizer == "adam":
      optimizer = optax.adam(learning_rate)
    elif optimizer == "sgd":
      optimizer = optax.sgd(learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    # Clipping the gradients prevent divergence and allow more stable training.
    if gradient_clipping:
      optimizer = optax.chain(optimizer,
                              optax.clip_by_global_norm(gradient_clipping))

    opt_init, opt_update = optimizer.init, optimizer.update

    def _stochastic_gradient_descent(params, opt_state, gradient):
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    self._opt_update_fn = _stochastic_gradient_descent
    self._opt_state = opt_init(self._params_q_network)
    self._loss_and_grad = jax.value_and_grad(self._loss, has_aux=False)
    self._jit_update = jax.jit(self._get_update())

  def step(self,
           time_step,
           is_evaluation=False,
           add_transition_record=True,
           use_softmax=False):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.
      use_softmax: Uses soft-max action selection.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (time_step.is_simultaneous_move() or
                                   self.player_id == int(
                                       time_step.current_player())):
      # Act according to epsilon-greedy or soft-max for current Q-network.
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      if use_softmax:
        action, probs = self._softmax(info_state, legal_actions)
      else:
        epsilon = self._get_epsilon(is_evaluation)
        action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_loss_value = self.learn()

      if self._step_counter % self._update_target_network_every == 0:
        self._params_target_q_network = _copy_params(self._params_q_network)

      if self._prev_time_step and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        self.add_transition(self._prev_time_step, self._prev_action,
                            self._prev_legal_action, time_step)

      if time_step.last():  # prepare for the next episode.
        self._prev_time_step = None
        self._prev_action = None
        self._prev_legal_action = None
      else:
        self._prev_time_step = time_step
        self._prev_action = action
        self._prev_legal_action = legal_actions

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, prev_legal_actions,
                     time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_time_step` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      prev_legal_actions: Previous legal actions.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    next_legal_actions = (
        time_step.observations["legal_actions"][self.player_id])
    next_legal_one_hots = self._to_one_hot(next_legal_actions)
    # Added for deep OMD: keep previous action mask.
    prev_legal_one_hots = self._to_one_hot(prev_legal_actions)

    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        legal_one_hots=prev_legal_one_hots,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        next_legal_one_hots=next_legal_one_hots)
    self._replay_buffer.add(transition)

  def _get_action_probs(self, params, info_states, legal_one_hots):
    """Returns the soft-max action probability distribution."""
    q_values = self.hk_network.apply(params, info_states)
    legal_q_values = q_values + (1 - legal_one_hots) * ILLEGAL_ACTION_PENALTY
    return jax.nn.softmax(legal_q_values / self._tau)

  def _loss(self, params, params_target, params_prev, info_states, actions,
            legal_one_hots, rewards, next_info_states, are_final_steps,
            next_legal_one_hots):
    """Returns the Munchausen loss."""
    # Target with 2 parts: reward and value for next state; each part is
    # modified according to the Munchausen trick.
    q_values = self.hk_network.apply(params, info_states)
    target_q_values = self.hk_network.apply(params_target, next_info_states)

    r_term = rewards
    if self._with_munchausen:
      probs = self._get_action_probs(params_prev, info_states, legal_one_hots)
      prob_prev_action = jnp.sum(probs * actions, axis=-1)
      penalty_pi = jnp.log(jnp.clip(prob_prev_action, MIN_ACTION_PROB))
      r_term += self._alpha * self._tau * penalty_pi

    if self._with_munchausen:
      # Average value over actions + extra log term.
      # We clip the probabilities to avoid NaNs in the log term.
      next_probs = self._get_action_probs(params_prev, next_info_states,
                                          next_legal_one_hots)
      q_term_values = next_probs * (
          target_q_values -
          self._tau * jnp.log(jnp.clip(next_probs, MIN_ACTION_PROB)))
      q_term = jnp.sum(q_term_values, axis=-1)
    else:
      # Maximum value.
      max_next_q = jnp.max(
          target_q_values + (1 - legal_one_hots) * ILLEGAL_ACTION_PENALTY,
          axis=-1)
      max_next_q = jax.numpy.where(
          1 - are_final_steps, x=max_next_q, y=jnp.zeros_like(max_next_q))
      q_term = max_next_q

    target = (r_term + (1 - are_final_steps) * self._discount_factor * q_term)
    target = jax.lax.stop_gradient(target)

    predictions = jnp.sum(q_values * actions, axis=-1)

    return self._loss_func(predictions - target)

  def _get_update(self):
    """Returns the gradient update function."""

    def update(params, params_target, params_prev, opt_state, info_states,
               actions, legal_one_hots, rewards, next_info_states,
               are_final_steps, next_legal_one_hots):
      loss_val, grad_val = self._loss_and_grad(params, params_target,
                                               params_prev, info_states,
                                               actions, legal_one_hots, rewards,
                                               next_info_states,
                                               are_final_steps,
                                               next_legal_one_hots)
      new_params, new_opt_state = self._opt_update_fn(params, opt_state,
                                                      grad_val)
      return new_params, new_opt_state, loss_val

    return update

  def _to_one_hot(self, a, value=1.0):
    """Returns the one-hot encoding of the action."""
    a_one_hot = np.zeros(self._num_actions)
    a_one_hot[a] = value
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
    legal_one_hots = np.asarray([t.legal_one_hots for t in transitions])
    rewards = np.asarray([t.reward for t in transitions])
    next_info_states = np.asarray([t.next_info_state for t in transitions])
    are_final_steps = np.asarray([t.is_final_step for t in transitions])
    next_legal_one_hots = np.asarray(
        [t.next_legal_one_hots for t in transitions])

    self._params_q_network, self._opt_state, loss_val = self._jit_update(
        self._params_q_network, self._params_target_q_network,
        self._params_prev_q_network, self._opt_state, info_states, actions,
        legal_one_hots, rewards, next_info_states, are_final_steps,
        next_legal_one_hots)

    return loss_val

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and action probabilities.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and action probabilities.
    """
    if self._rs.rand() < epsilon:
      action = self._rs.choice(legal_actions)
      probs = self._to_one_hot(legal_actions, value=1.0 / len(legal_actions))
      return action, probs

    info_state = np.reshape(info_state, [1, -1])
    q_values = self.hk_network_apply(self._params_q_network, info_state)[0]
    legal_one_hot = self._to_one_hot(legal_actions)
    legal_q_values = q_values + (1 - legal_one_hot) * ILLEGAL_ACTION_PENALTY
    action = int(np.argmax(legal_q_values))
    probs = self._to_one_hot(action)
    return action, probs

  def _get_epsilon(self, is_evaluation):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0

    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**self._epsilon_power)
    return decayed_epsilon

  def _softmax(self, info_state, legal_actions):
    """Returns a valid soft-max action and action probabilities."""
    info_state = np.reshape(info_state, [1, -1])
    q_values = self.hk_network_apply(self._params_q_network, info_state)[0]
    legal_one_hot = self._to_one_hot(legal_actions)
    legal_q_values = q_values + (1 - legal_one_hot) * ILLEGAL_ACTION_PENALTY
    # Apply temperature and subtract the maximum value for numerical stability.
    temp = legal_q_values / self._tau
    unnormalized = np.exp(temp - np.amax(temp))
    probs = unnormalized / unnormalized.sum()
    action = self._rs.choice(legal_actions, p=probs[legal_actions])
    return action, probs

  def update_prev_q_network(self):
    """Updates the parameters of the previous Q-network."""
    self._params_prev_q_network = _copy_params(self._params_q_network)
    if self._reset_replay_buffer_on_update:
      # Also reset the replay buffer to avoid having transitions from the
      # previous policy.
      self._replay_buffer.reset()

  @property
  def loss(self):
    return self._last_loss_value


class SoftMaxMunchausenDQN(rl_agent.AbstractAgent):
  """Wraps a Munchausen DQN agent to use soft-max action selection."""

  def __init__(self, agent: MunchausenDQN):
    self._agent = agent

  def step(self, time_step, is_evaluation=False):
    return self._agent.step(
        time_step, is_evaluation=is_evaluation, use_softmax=True)


class DeepOnlineMirrorDescent(object):
  """The deep online mirror descent algorithm."""

  def __init__(self,
               game,
               envs,
               agents,
               eval_every=200,
               num_episodes_per_iteration=1000,
               logging_fn: Optional[Callable[[int, int, Dict[str, Any]],
                                             None]] = None):
    """Initializes mirror descent.

    Args:
      game: The game,
      envs: RL environment for each player.
      agents: Munchausen DQN agents for each player.
      eval_every: Number of training episodes between two evaluations.
      num_episodes_per_iteration: Number of training episodes for each
        iiteration.
      logging_fn: Callable for logging the metrics. The arguments will be the
        current iteration, episode and a dictionary of metrics to log.
    """
    assert len(envs) == len(agents)
    # Make sure that the agents are all MunchausenDQN.
    for agent in agents:
      assert isinstance(agent, MunchausenDQN)

    self._game = game

    self._eval_every = eval_every
    self._num_episodes_per_iteration = num_episodes_per_iteration

    self._envs = envs
    self._agents = agents
    self._use_observation = envs[0].use_observation

    self._iteration = 0

    if logging_fn is None:
      logging_fn = lambda it, ep, vals: logging.info("%d/%d %r", it, ep, vals)
    self._logging_fn = logging_fn

    # Set the initial policy and distribution.
    self._update_policy_and_distribution()

  def _train_agents(self):
    """Trains the agents.

    This will evaluate the Q-network for current policy and distribution.
    """
    for ep in range(self._num_episodes_per_iteration):
      for env, agent in zip(self._envs, self._agents):
        time_step = env.reset()
        while not time_step.last():
          agent_output = agent.step(time_step, use_softmax=False)
          action_list = [agent_output.action]
          time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        agent.step(time_step, use_softmax=False)

      if (ep + 1) % self._eval_every == 0:
        metrics = {}
        for i, agent in enumerate(self._agents):
          metrics[f"agent{i}/loss"] = agent.loss
        self._logging_fn(self._iteration, ep + 1, metrics)

  def _update_policy_and_distribution(self):
    """Updates the current soft-max policy and the distribution."""
    self._policy = rl_agent_policy.JointRLAgentPolicy(self._game, {
        idx: SoftMaxMunchausenDQN(agent)
        for idx, agent in enumerate(self._agents)
    }, self._use_observation)
    self._distribution = distribution_std.DistributionPolicy(
        self._game, self._policy)

  def iteration(self):
    """An iteration of Mirror Descent."""
    self._train_agents()
    self._update_policy_and_distribution()
    self._iteration += 1
    # Update the distributions of the environments and the previous Q-networks
    # of the agents.
    for env, agent in zip(self._envs, self._agents):
      env.update_mfg_distribution(self.distribution)
      agent.update_prev_q_network()

  @property
  def policy(self):
    return self._policy

  @property
  def distribution(self):
    return self._distribution
