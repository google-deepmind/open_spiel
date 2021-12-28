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

"""DQN agent implemented in PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x


class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in PyTorch.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

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
               loss_str="mse"):
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
    self._q_network = MLP(state_representation_size, self._layer_sizes,
                          num_actions)

    self._target_q_network = MLP(state_representation_size, self._layer_sizes,
                                 num_actions)

    if loss_str == "mse":
      self.loss_class = F.mse_loss
    elif loss_str == "huber":
      self.loss_class = F.smooth_l1_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    if optimizer_str == "adam":
      self._optimizer = torch.optim.Adam(
          self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = torch.optim.SGD(
          self._q_network.parameters(), lr=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

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
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
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
        # state_dict method returns a dictionary containing a whole state of the
        # module.
        self._target_q_network.load_state_dict(self._q_network.state_dict())

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
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      info_state = torch.Tensor(np.reshape(info_state, [1, -1]))
      q_values = self._q_network(info_state).detach()[0]
      legal_q_values = q_values[legal_actions]
      action = legal_actions[torch.argmax(legal_q_values)]
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
    info_states = torch.Tensor([t.info_state for t in transitions])
    actions = torch.LongTensor([t.action for t in transitions])
    rewards = torch.Tensor([t.reward for t in transitions])
    next_info_states = torch.Tensor([t.next_info_state for t in transitions])
    are_final_steps = torch.Tensor([t.is_final_step for t in transitions])
    legal_actions_mask = torch.Tensor(
        [t.legal_actions_mask for t in transitions])

    self._q_values = self._q_network(info_states)
    self._target_q_values = self._target_q_network(next_info_states).detach()

    illegal_actions = 1 - legal_actions_mask
    illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY
    max_next_q = torch.max(self._target_q_values + illegal_logits, dim=1)[0]
    target = (
        rewards + (1 - are_final_steps) * self._discount_factor * max_next_q)
    action_indices = torch.stack([
        torch.arange(self._q_values.shape[0], dtype=torch.long), actions
    ],
                                 dim=0)
    predictions = self._q_values[list(action_indices)]

    loss = self.loss_class(predictions, target)

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

    return loss

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

  def get_weights(self):
    variables = [m.weight for m in self._q_network.model]
    variables.append([m.weight for m in self._target_q_network.model])
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates it with noise.

    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.

    Returns:
      Perturbated copy of the model.
    """
    _ = self._kwargs.pop("self", None)
    copied_object = DQN(**self._kwargs)

    q_network = getattr(copied_object, "_q_network")
    target_q_network = getattr(copied_object, "_target_q_network")

    if copy_weights:
      with torch.no_grad():
        for q_model in q_network.model:
          q_model.weight *= (1 + sigma * torch.randn(q_model.weight.shape))
        for tq_model in target_q_network.model:
          tq_model.weight *= (1 + sigma * torch.randn(tq_model.weight.shape))
    return copied_object
