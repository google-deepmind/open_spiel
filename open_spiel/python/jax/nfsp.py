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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import enum
import os

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python import rl_agent
from open_spiel.python.jax import dqn
from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer

Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

MODE = enum.Enum("mode", "best_response average_policy")


class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in JAX.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

  def __init__(self,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes,
               reservoir_buffer_capacity,
               anticipatory_param,
               batch_size=128,
               rl_learning_rate=0.01,
               sl_learning_rate=0.01,
               min_buffer_size_to_learn=1000,
               learn_every=64,
               optimizer_str="sgd",
               **kwargs):
    """Initialize the `NFSP` agent."""
    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._step_counter = 0

    # Inner RL agent
    kwargs.update({
        "batch_size": batch_size,
        "learning_rate": rl_learning_rate,
        "learn_every": learn_every,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "optimizer_str": optimizer_str,
    })
    self._rl_agent = dqn.DQN(player_id, state_representation_size,
                             num_actions, hidden_layers_sizes, **kwargs)

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._last_sl_loss_value = None

    # Average policy network.
    def network(x):
      mlp = hk.nets.MLP(self._layer_sizes + [num_actions])
      return mlp(x)

    self.hk_avg_network = hk.without_apply_rng(hk.transform(network))

    def avg_network_policy(param, info_state):
      action_values = self.hk_avg_network.apply(param, info_state)
      action_probs = jax.nn.softmax(action_values, axis=1)
      return action_values, action_probs

    self._avg_network_policy = jax.jit(avg_network_policy)

    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1, state_representation_size])
    self.params_avg_network = self.hk_avg_network.init(rng, x)
    self.params_avg_network = jax.device_put(self.params_avg_network)

    self._savers = [
        ("q_network", self._rl_agent.params_q_network),
        ("avg_network", self.params_avg_network)
    ]

    if optimizer_str == "adam":
      opt_init, opt_update = optax.chain(
          optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
          optax.scale(sl_learning_rate))
    elif optimizer_str == "sgd":
      opt_init, opt_update = optax.sgd(sl_learning_rate)
    else:
      raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")
    self._opt_update_fn = self._get_update_func(opt_update)
    self._opt_state = opt_init(self.params_avg_network)
    self._loss_and_grad = jax.value_and_grad(self._loss_avg, has_aux=False)

    self._sample_episode_policy()
    self._jit_update = jax.jit(self.get_update())

  def _get_update_func(self, opt_update):

    def update(params, opt_state, gradient):
      """Learning rule (stochastic gradient descent)."""
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    return update

  def get_step_counter(self):
    return self._step_counter

  @contextlib.contextmanager
  def temp_mode_as(self, mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._mode
    self._mode = mode
    yield
    self._mode = previous_mode

  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def _act(self, info_state, legal_actions):
    info_state = np.reshape(info_state, [1, -1])
    action_values, action_probs = self._avg_network_policy(
        self.params_avg_network, info_state
        )

    self._last_action_values = action_values[0]
    # Remove illegal actions, normalize probs
    probs = np.zeros(self._num_actions)
    action_probs = np.asarray(action_probs)
    probs[legal_actions] = action_probs[0][legal_actions]
    probs /= sum(probs)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

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
        action, probs = self._act(info_state, legal_actions)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent.add_transition(self._prev_timestep, self._prev_action,
                                      time_step)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if self._mode == MODE.average_policy:
          self._rl_agent.learn()

      # Prepare for the next episode.
      if time_step.last():
        self._sample_episode_policy()
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action
    return agent_output

  def _add_transition(self, time_step, agent_output):
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
        info_state=(time_step.observations["info_state"][self.player_id][:]),
        action_probs=agent_output.probs,
        legal_actions_mask=legal_actions_mask)
    self._reservoir_buffer.add(transition)

  def _loss_avg(self, param_avg, info_states, action_probs):
    avg_logit = self.hk_avg_network.apply(param_avg, info_states)
    loss_value = -jnp.sum(
        action_probs * jax.nn.log_softmax(avg_logit)) / avg_logit.shape[0]
    return loss_value

  def get_update(self):
    def update(param_avg, opt_state_avg, info_states, action_probs):
      loss_val, grad_val = self._loss_and_grad(param_avg, info_states,
                                               action_probs)
      new_param_avg, new_opt_state_avg = self._opt_update_fn(
          param_avg, opt_state_avg, grad_val)
      return new_param_avg, new_opt_state_avg, loss_val
    return update

  def _learn(self):
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

    self.params_avg_network, self._opt_state, loss_val_avg = self._jit_update(
        self.params_avg_network, self._opt_state, info_states, action_probs)
    return loss_val_avg

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    """Saves the average policy network and the inner RL agent's q-network.

    Note that this does not save the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory where checkpoints will be saved.
    """
    raise NotImplementedError

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      if os.path.exists(path):
        return True
    return False

  def restore(self, checkpoint_dir):
    """Restores the average policy network and the inner RL agent's q-network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory from which checkpoints will be restored.
    """
    raise NotImplementedError
