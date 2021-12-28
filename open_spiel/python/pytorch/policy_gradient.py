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

# Lint as python3.
r"""Policy Gradient based agents implemented in PyTorch.

This class is composed of three policy gradient (PG) algorithms:

- Q-based Policy Gradient (QPG): an "all-actions" advantage actor-critic
algorithm differing from A2C in that all action values are used to estimate the
policy gradient (as opposed to only using the action taken into account):

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * (Q_a - baseline)

where (Q_a - baseline) is the usual advantage. QPG is also known as Mean
Actor-Critic (https://arxiv.org/abs/1709.00503).


- Regret policy gradient (RPG): a PG algorithm inspired by counterfactual regret
minimization (CFR). Unlike standard actor-critic methods (e.g. A2C), the loss is
defined purely in terms of thresholded regrets as follows:

    baseline = \sum_a pi_a * Q_a
    loss = regret = \sum_a relu(Q_a - baseline)

where gradients only flow through the action value (Q_a) part and are blocked on
the baseline part (which is trained separately by usual MSE loss).
The lack of negative sign in the front of the loss represents a switch from
gradient ascent on the score to descent on the loss.


- Regret Matching Policy Gradient (RMPG): inspired by regret-matching, the
policy gradient is by weighted by the thresholded regret:

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * relu(Q_a - baseline)


These algorithms were published in NeurIPS 2018. Paper title: "Actor-Critic
Policy Optimization in Partially Observable Multiagent Environment", the paper
is available at: https://arxiv.org/abs/1810.09026.

- Advantage Actor Critic (A2C): The popular advantage actor critic (A2C)
algorithm. The algorithm uses the baseline (Value function) as a control variate
to reduce variance of the policy gradient. The loss is only computed for the
actions actually taken in the episode as opposed to a loss computed for all
actions in the variants above.

  advantages = returns - baseline
  loss = -log(pi_a) * advantages

The algorithm can be found in the textbook:
https://incompleteideas.net/book/RLbook2018.pdf under the chapter on
`Policy Gradients`.

See  open_spiel/python/pytorch/losses/rl_losses_test.py for an example of the
loss computation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
from absl import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from open_spiel.python import rl_agent
from open_spiel.python.pytorch.dqn import SonnetLinear
from open_spiel.python.pytorch.losses import rl_losses

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")


class MLPTorso(nn.Module):
  """A specialized half-MLP module when constructing multiple heads.

  Note that every layer includes a ReLU non-linearity activation.
  """

  def __init__(self, input_size, hidden_sizes):
    """Create the MLPTorso.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
    """

    super(MLPTorso, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x


class PolicyGradient(rl_agent.AbstractAgent):
  """RPG Agent implementation in PyTorch.

  See open_spiel/python/examples/single_agent_catch.py for an usage example.
  """

  def __init__(self,
               player_id,
               info_state_size,
               num_actions,
               loss_str="a2c",
               loss_class=None,
               hidden_layers_sizes=(128,),
               batch_size=16,
               critic_learning_rate=0.01,
               pi_learning_rate=0.001,
               entropy_cost=0.01,
               num_critic_before_pi=8,
               additional_discount_factor=1.0,
               max_global_gradient_norm=None,
               optimizer_str="sgd"):
    """Initialize the PolicyGradient agent.

    Args:
      player_id: int, player identifier. Usually its position in the game.
      info_state_size: int, info_state vector size.
      num_actions: int, number of actions per info state.
      loss_str: string or None. If string, must be one of ["rpg", "qpg", "rm",
        "a2c"] and defined in `_get_loss_class`. If None, a loss class must be
        passed through `loss_class`. Defaults to "a2c".
      loss_class: Class or None. If Class, it must define the policy gradient
        loss. If None a loss class in a string format must be passed through
        `loss_str`. Defaults to None.
      hidden_layers_sizes: iterable, defines the neural network layers. Defaults
          to (128,), which produces a NN: [INPUT] -> [128] -> ReLU -> [OUTPUT].
      batch_size: int, batch size to use for Q and Pi learning. Defaults to 128.
      critic_learning_rate: float, learning rate used for Critic (Q or V).
        Defaults to 0.001.
      pi_learning_rate: float, learning rate used for Pi. Defaults to 0.001.
      entropy_cost: float, entropy cost used to multiply the entropy loss. Can
        be set to None to skip entropy computation. Defaults to 0.001.
      num_critic_before_pi: int, number of Critic (Q or V) updates before each
        Pi update. Defaults to 8 (every 8th critic learning step, Pi also
        learns).
      additional_discount_factor: float, additional discount to compute returns.
        Defaults to 1.0, in which case, no extra discount is applied.  None that
        users must provide *only one of* `loss_str` or `loss_class`.
      max_global_gradient_norm: float or None, maximum global norm of a gradient
        to which the gradient is shrunk if its value is larger.
      optimizer_str: String defining which optimizer to use. Supported values
        are {sgd, adam}
    """
    assert bool(loss_str) ^ bool(loss_class), "Please provide only one option."
    self._kwargs = locals()
    loss_class = loss_class if loss_class else self._get_loss_class(loss_str)
    self._loss_class = loss_class

    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._extra_discount = additional_discount_factor
    self._num_critic_before_pi = num_critic_before_pi
    self._max_global_gradient_norm = max_global_gradient_norm

    self._episode_data = []
    self._dataset = collections.defaultdict(list)
    self._prev_time_step = None
    self._prev_action = None

    # Step counters
    self._step_counter = 0
    self._episode_counter = 0
    self._num_learn_steps = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Network
    # activate final as we plug logit and qvalue heads afterwards.
    self._net_torso = MLPTorso(info_state_size, self._layer_sizes)
    torso_out_size = self._layer_sizes[-1]
    self._policy_logits_layer = SonnetLinear(
        torso_out_size, self._num_actions, activate_relu=False)
    # Do not remove policy_logits_network. Even if it's not used directly here,
    # other code outside this file refers to it.
    self.policy_logits_network = nn.Sequential(self._net_torso,
                                               self._policy_logits_layer)

    self._savers = []

    # Add baseline (V) head for A2C (or Q-head for QPG / RPG / RMPG)
    if optimizer_str == "adam":
      self._critic_optimizer = optim.Adam
    elif optimizer_str == "sgd":
      self._critic_optimizer = optim.SGD
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    if loss_class.__name__ == "BatchA2CLoss":
      self._baseline_layer = SonnetLinear(
          torso_out_size, 1, activate_relu=False)
      self._critic_network = nn.Sequential(self._net_torso,
                                           self._baseline_layer)
    else:
      self._q_values_layer = SonnetLinear(
          torso_out_size, self._num_actions, activate_relu=False)
      self._critic_network = nn.Sequential(self._net_torso,
                                           self._q_values_layer)

    self._critic_optimizer = self._critic_optimizer(
        self._critic_network.parameters(), lr=critic_learning_rate)

    # Pi loss
    self.pg_class = loss_class(entropy_cost=entropy_cost)
    self._pi_network = nn.Sequential(self._net_torso, self._policy_logits_layer)
    if optimizer_str == "adam":
      self._pi_optimizer = optim.Adam(
          self._pi_network.parameters(), lr=pi_learning_rate)
    elif optimizer_str == "sgd":
      self._pi_optimizer = optim.SGD(
          self._pi_network.parameters(), lr=pi_learning_rate)

    self._loss_str = loss_str

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rl_losses.BatchRPGLoss
    elif loss_str == "qpg":
      return rl_losses.BatchQPGLoss
    elif loss_str == "rm":
      return rl_losses.BatchRMLoss
    elif loss_str == "a2c":
      return rl_losses.BatchA2CLoss

  def minimize_with_clipping(self, model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    if self._max_global_gradient_norm is not None:
      nn.utils.clip_grad_norm_(model.parameters(),
                               self._max_global_gradient_norm)
    optimizer.step()

  def _act(self, info_state, legal_actions):
    # Make a singleton batch for NN compatibility: [1, info_state_size]
    info_state = torch.Tensor(np.reshape(info_state, [1, -1]))
    torso_out = self._net_torso(info_state)
    self._policy_logits = self._policy_logits_layer(torso_out)
    policy_probs = F.softmax(self._policy_logits, dim=1).detach()

    # Remove illegal actions, re-normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = policy_probs[0][legal_actions]
    if sum(probs) != 0:
      probs /= sum(probs)
    else:
      probs[legal_actions] = 1 / len(legal_actions)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs = self._act(info_state, legal_actions)
    else:
      action = None
      probs = []

    if not is_evaluation:
      self._step_counter += 1

      # Add data points to current episode buffer.
      if self._prev_time_step:
        self._add_transition(time_step)

      # Episode done, add to dataset and maybe learn.
      if time_step.last():
        self._add_episode_data_to_dataset()
        self._episode_counter += 1

        if len(self._dataset["returns"]) >= self._batch_size:
          self._critic_update()
          self._num_learn_steps += 1
          if self._num_learn_steps % self._num_critic_before_pi == 0:
            self._pi_update()
          self._dataset = collections.defaultdict(list)

        self._prev_time_step = None
        self._prev_action = None
        return
      else:
        self._prev_time_step = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    for name, model in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      torch.save(model.state_dict(), path)
      logging.info("Saved to path: %s", path)

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      if os.path.exists(path):
        return True
    return False

  def restore(self, checkpoint_dir):
    for name, model in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      model.load_state_dict(torch.load(full_checkpoint_dir))

  @property
  def loss(self):
    return (self._last_critic_loss_value, self._last_pi_loss_value)

  def _add_episode_data_to_dataset(self):
    """Add episode data to the buffer."""
    info_states = [data.info_state for data in self._episode_data]
    rewards = [data.reward for data in self._episode_data]
    discount = [data.discount for data in self._episode_data]
    actions = [data.action for data in self._episode_data]

    # Calculate returns
    returns = np.array(rewards)
    for idx in reversed(range(len(rewards[:-1]))):
      returns[idx] = (
          rewards[idx] +
          discount[idx] * returns[idx + 1] * self._extra_discount)

    # Add flattened data points to dataset
    self._dataset["actions"].extend(actions)
    self._dataset["returns"].extend(returns)
    self._dataset["info_states"].extend(info_states)
    self._episode_data = []

  def _add_transition(self, time_step):
    """Adds intra-episode transition to the `_episode_data` buffer.

    Adds the transition from `self._prev_time_step` to `time_step`.

    Args:
      time_step: an instance of rl_environment.TimeStep.
    """
    assert self._prev_time_step is not None
    legal_actions = (
        self._prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            self._prev_time_step.observations["info_state"][self.player_id][:]),
        action=self._prev_action,
        reward=time_step.rewards[self.player_id],
        discount=time_step.discounts[self.player_id],
        legal_actions_mask=legal_actions_mask)

    self._episode_data.append(transition)

  def _critic_update(self):
    """Compute the Critic loss on sampled transitions & perform a critic update.

    Returns:
      The average Critic loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_state = torch.Tensor(self._dataset["info_states"])
    action = torch.LongTensor(self._dataset["actions"])
    return_ = torch.Tensor(self._dataset["returns"])
    torso_out = self._net_torso(info_state)

    # Critic loss
    # Baseline loss in case of A2C
    if self._loss_class.__name__ == "BatchA2CLoss":
      baseline = torch.squeeze(self._baseline_layer(torso_out), dim=1)
      critic_loss = torch.mean(F.mse_loss(baseline, return_))
      self.minimize_with_clipping(self._baseline_layer, self._critic_optimizer,
                                  critic_loss)
    else:
      # Q-loss otherwise.
      q_values = self._q_values_layer(torso_out)
      action_indices = torch.stack([torch.range(q_values.shape[0]), action],
                                   dim=-1)
      value_predictions = torch.gather_nd(q_values, action_indices)
      critic_loss = torch.mean(F.mse_loss(value_predictions, return_))
      self.minimize_with_clipping(self._q_values_layer, self._critic_optimizer,
                                  critic_loss)
    self._last_critic_loss_value = critic_loss
    return critic_loss

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.

    Returns:
      The average Pi loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_state = torch.Tensor(self._dataset["info_states"])
    action = torch.LongTensor(self._dataset["actions"])
    return_ = torch.Tensor(self._dataset["returns"])
    torso_out = self._net_torso(info_state)
    self._policy_logits = self._policy_logits_layer(torso_out)

    if self._loss_class.__name__ == "BatchA2CLoss":
      baseline = torch.squeeze(self._baseline_layer(torso_out), dim=1)
      pi_loss = self.pg_class.loss(
          policy_logits=self._policy_logits,
          baseline=baseline,
          actions=action,
          returns=return_)
      self.minimize_with_clipping(self._baseline_layer, self._pi_optimizer,
                                  pi_loss)
    else:
      q_values = self._q_values_layer(torso_out)
      pi_loss = self.pg_class.loss(
          policy_logits=self._policy_logits, action_values=q_values)
      self.minimize_with_clipping(self._q_values_layer, self._pi_optimizer,
                                  pi_loss)
    self._last_pi_loss_value = pi_loss
    return pi_loss

  def get_weights(self):
    variables = [m.weight for m in self._net_torso.model]
    variables.append(self._policy_logits_layer.weight)
    if self._loss_class.__name__ == "BatchA2CLoss":
      variables.append(self._baseline_layer.weight)
    else:
      variables.append(self._q_values_layer.weight)
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates its network's weights with noise.

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
    copied_object = PolicyGradient(**self._kwargs)

    net_torso = getattr(copied_object, "_net_torso")
    policy_logits_layer = getattr(copied_object, "_policy_logits_layer")
    if hasattr(copied_object, "_q_values_layer"):
      q_values_layer = getattr(copied_object, "_q_values_layer")
    if hasattr(copied_object, "_baseline_layer"):
      baseline_layer = getattr(copied_object, "_baseline_layer")

    if copy_weights:
      with torch.no_grad():
        for layer in net_torso.model:
          layer.weight *= (1 + sigma * torch.randn(layer.weight.shape))

        policy_logits_layer.weight *= (
            1 + sigma * torch.randn(policy_logits_layer.weight.shape))

        if hasattr(copied_object, "_q_values_layer"):
          q_values_layer.weight *= (
              1 + sigma * torch.randn(q_values_layer.weight.shape))

        if hasattr(copied_object, "_baseline_layer"):
          baseline_layer.weight *= (
              1 + sigma * torch.randn(baseline_layer.weight.shape))

    return copied_object
