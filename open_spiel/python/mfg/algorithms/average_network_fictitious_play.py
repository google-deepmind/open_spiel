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
"""Implementation of the Deep Average-network Fictitious Play.

Coupled with agents that compute a best-response (BR) at each iteration, instead
of keeping in memory all the BRs from past iterations Deep Average-network
Fictitious Play learns along the way the policy generating the average
distribution. This is done by keeping a buffer of state-action pairs generated
by past BRs and learning the average policy (represented by a neural network) by
minimizing a categorical loss. This approach is inspired by the Neural
Fictitious Self Play (NFSP) method (Heinrich & Silver, 2016), developed
initially for imperfect information games with a finite number of players, and
adapted here to the MFG setting.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python import rl_agent
from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.mfg.algorithms import distribution
import pyspiel
from open_spiel.python.utils import reservoir_buffer
from open_spiel.python.utils import training


@dataclasses.dataclass
class Transition:
  """Transitions stored in the reservoir buffer."""
  info_state: np.ndarray
  action_probs: np.ndarray
  legal_actions_mask: np.ndarray


class AveragePolicy(rl_agent.AbstractAgent):
  """NFSP-like agent that learns an average policy using a single network."""

  def __init__(self,
               player_id: int,
               br_rl_agent: rl_agent.AbstractAgent,
               state_representation_size: int,
               num_actions: int,
               hidden_layers_sizes: List[int],
               params_avg_network: Optional[jnp.ndarray] = None,
               reservoir_buffer_capacity: int = 100000,
               batch_size: int = 128,
               learning_rate: float = 0.01,
               min_buffer_size_to_learn: int = 1000,
               optimizer_str: str = 'sgd',
               gradient_clipping: Optional[float] = None,
               seed: int = 42,
               tau: float = 1.0):
    """Initialize the AveragePolicy agent."""
    self._br_rl_agent = br_rl_agent
    self._player_id = player_id
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._reservoir_buffer = reservoir_buffer.ReservoirBuffer(
        reservoir_buffer_capacity)

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Average policy network.
    def network(x):
      mlp = hk.nets.MLP(hidden_layers_sizes + [num_actions])
      return mlp(x)

    self.avg_network = hk.without_apply_rng(hk.transform(network))

    def avg_network_policy(param, info_state):
      action_values = self.avg_network.apply(param, info_state)
      return jax.nn.softmax(action_values / tau, axis=1)

    self._avg_network_policy = jax.jit(avg_network_policy)

    rng = jax.random.PRNGKey(seed)
    x = jnp.ones([1, state_representation_size])
    # Use the specified parameters if any, or initialize the network with random
    # weights.
    if params_avg_network is None:
      self._params_avg_network = self.avg_network.init(rng, x)
    else:
      self._params_avg_network = jax.tree_util.tree_map(lambda x: x.copy(),
                                                        params_avg_network)
    self._params_avg_network = jax.device_put(self._params_avg_network)

    if optimizer_str == 'adam':
      optimizer = optax.adam(learning_rate)
    elif optimizer_str == 'sgd':
      optimizer = optax.sgd(learning_rate)
    else:
      raise ValueError('Not implemented, choose from "adam" and "sgd".')

    if gradient_clipping:
      optimizer = optax.chain(optimizer,
                              optax.clip_by_global_norm(gradient_clipping))

    opt_init, opt_update = optimizer.init, optimizer.update

    def opt_update_fn(params, opt_state, gradient):
      """Learning rule (stochastic gradient descent)."""
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    self._opt_update_fn = opt_update_fn
    self._opt_state = opt_init(self._params_avg_network)
    self._loss_and_grad = jax.value_and_grad(self._loss_avg, has_aux=False)

    self._jit_update = jax.jit(self._get_update_fn())

  def _get_update_fn(self):
    """Returns the function that updates the parameters."""

    def update(param_avg, opt_state_avg, info_states, action_probs):
      loss_val, grad_val = self._loss_and_grad(param_avg, info_states,
                                               action_probs)
      new_param_avg, new_opt_state_avg = self._opt_update_fn(
          param_avg, opt_state_avg, grad_val)
      return new_param_avg, new_opt_state_avg, loss_val

    return update

  def _act(self, info_state, legal_actions) -> Tuple[int, np.ndarray]:
    """Returns an action and the action probabilities."""
    info_state = np.reshape(info_state, [1, -1])
    action_probs = self._avg_network_policy(self._params_avg_network,
                                            info_state)
    # Remove illegal actions and normalize probs
    probs = np.zeros(self._num_actions)
    action_probs = np.asarray(action_probs)
    probs[legal_actions] = action_probs[0][legal_actions]
    probs /= sum(probs)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  @property
  def loss(self) -> Optional[float]:
    """Return the latest loss."""
    return self._last_loss_value

  def step(self,
           time_step: rl_environment.TimeStep,
           is_evaluation: bool = True) -> Optional[rl_agent.StepOutput]:
    """Returns the action to be taken by following the average network policy.

    Note that unlike most other algorithms, this method doesn't train the agent.
    Instead, we add new samples to the reservoir buffer and the training happens
    at a later stage.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Prepare for the next episode.
    if time_step.last():
      return

    if is_evaluation:
      # Use the average policy network.
      info_state = time_step.observations['info_state'][self._player_id]
      legal_actions = time_step.observations['legal_actions'][self._player_id]
      action, probs = self._act(info_state, legal_actions)
      return rl_agent.StepOutput(action=action, probs=probs)

    # Use the best response agent and add the transition in the reservoir
    # buffer.
    br_agent_output = self._br_rl_agent.step(time_step, is_evaluation=True)
    self._add_transition(time_step, br_agent_output)
    return br_agent_output

  def _add_transition(self, time_step, agent_output):
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations['legal_actions'][self._player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(time_step.observations['info_state'][self._player_id][:]),
        action_probs=agent_output.probs,
        legal_actions_mask=legal_actions_mask)
    self._reservoir_buffer.add(transition)

  def _loss_avg(self, param_avg, info_states, action_probs):
    avg_logit = self.avg_network.apply(param_avg, info_states)
    loss_value = -jnp.sum(
        action_probs * jax.nn.log_softmax(avg_logit)) / avg_logit.shape[0]
    return loss_value

  def learn(self) -> Optional[float]:
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (len(self._reservoir_buffer) < self._batch_size or
        len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)
    info_states = np.asarray([t.info_state for t in transitions])
    action_probs = np.asarray([t.action_probs for t in transitions])

    self._params_avg_network, self._opt_state, loss_val_avg = self._jit_update(
        self._params_avg_network, self._opt_state, info_states, action_probs)
    self._last_loss_value = float(loss_val_avg)
    return loss_val_avg


class AverageNetworkFictitiousPlay(object):
  """Deep Average-network Fictitious Play.

  See the file description for more information.
  """

  def __init__(self,
               game: pyspiel.Game,
               envs: Sequence[rl_environment.Environment],
               br_rl_agents: Sequence[rl_agent.AbstractAgent],
               num_episodes_per_iteration: int,
               num_training_steps_per_iteration: int,
               eval_every: int = 200,
               logging_fn: Optional[Callable[[int, int, Dict[str, Any]],
                                             None]] = None,
               **kwargs):
    """Initializes the greedy policy.

    Args:
      game: The game to analyze.
      envs: RL environment for each player.
      br_rl_agents: Best response, e.g. DQN, agents for each player.
      num_episodes_per_iteration: Number of episodes to collect samples that are
        added to the reservoir buffer.
      num_training_steps_per_iteration: Number of steps to train the average
        policy in each iteration.
      eval_every: Number of training steps between two evaluations.
      logging_fn: Callable for logging the metrics. The arguments will be the
        current iteration, episode and a dictionary of metrics to log.
      **kwargs: kwargs passed to the AveragePolicy() constructor.
    """
    self._game = game
    self._envs = envs
    self._num_episodes_per_iteration = num_episodes_per_iteration
    self._num_training_steps_per_iteration = num_training_steps_per_iteration
    self._eval_every = eval_every
    self._logging_fn = logging_fn

    self._num_players = game.num_players()
    self._fp_iteration = 0

    env = self._envs[0]
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']

    self._avg_rl_agents = [
        AveragePolicy(p, br_rl_agents[p], info_state_size, num_actions,
                      **kwargs) for p in range(self._num_players)
    ]
    self._policy = rl_agent_policy.JointRLAgentPolicy(
        self._game,
        {idx: agent for idx, agent in enumerate(self._avg_rl_agents)},
        use_observation=env.use_observation)
    self._update_distribution()

  def _update_distribution(self):
    """Calculates the current distribution and updates the environments."""
    self._distribution = distribution.DistributionPolicy(
        self._game, self._policy)
    for env in self._envs:
      env.update_mfg_distribution(self._distribution)

  @property
  def policy(self) -> rl_agent_policy.JointRLAgentPolicy:
    return self._policy

  def iteration(self):
    """An average-network fictitious play step."""
    # Generate samples using latest best-response and add them to the reservoir
    # buffer. Note that the algorithm is agnostic to the best-response policies
    # as we only use them to collect new samples. They can be approximate (e.g.
    # backed by a deep algorithm) or exact.
    training.run_episodes(
        self._envs,
        self._avg_rl_agents,
        num_episodes=self._num_episodes_per_iteration,
        is_evaluation=False)

    # Train the average policy.
    for step in range(self._num_training_steps_per_iteration):
      for avg_rl_agent in self._avg_rl_agents:
        avg_rl_agent.learn()

      if self._logging_fn and (step + 1) % self._eval_every == 0:
        self._logging_fn(
            self._fp_iteration, step, {
                f'avg_agent{i}/loss': float(agent.loss)
                for i, agent in enumerate(self._avg_rl_agents)
            })

    # Update the distribution.
    self._update_distribution()
    self._fp_iteration += 1
