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
"""Boltzmann DQN agent implemented in JAX.

This algorithm is a variation of DQN that uses a softmax policy directly with
the unregularized action-value function. See https://arxiv.org/abs/2102.01585.
"""

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from open_spiel.python.pytorch import dqn


class BoltzmannDQN(dqn.DQN):
  """Boltzmann DQN implementation in JAX."""

  def __init__(self, *args, eta: float = 1.0, **kwargs):
    """Initializes the Boltzmann DQN agent.

    Args:
      *args: args passed to the underlying DQN agent.
      eta: Temperature parameter used in the softmax function.
      **kwargs: kwargs passed to the underlying DQN agent.
    """
    super().__init__(*args, **kwargs)
    self._prev_q_network = copy.deepcopy(self._q_network)
    self._temperature = eta

  def _boltzmann_action_probs(
      self,
      network: nn.Module,
      info_state: list[Any],
      legal_actions: list[Any],
      coeff: np.ndarray = None,
  ):
    """Returns a valid softmax action and action probabilities.

    Args:
      network: (nn.Module) Parameters of the Q-network.
      info_state: (list) Observations from the environment.
      legal_actions (list): List of legal actions.
      coeff (np.ndarray): If not None, then the terms in softmax function will
        be element-wise multiplied with these coefficients.

    Returns:
      A valid softmax action and action probabilities.
    """
    info_state = torch.FloatTensor(
        np.asarray(info_state).reshape(1, -1), device=self._device
    )
    q_values = network(info_state).detach().squeeze(0)

    illegal_actions_mask = torch.logical_not(
        torch.BoolTensor(legal_actions, device=self._device)
    )
    legal_q_values = q_values.masked_fill(
        illegal_actions_mask, dqn.ILLEGAL_ACTION_LOGITS_PENALTY
    )
    # Apply temperature and subtract the maximum value for numerical stability.
    temp = legal_q_values / self._temperature
    probs = F.softmax(coeff * temp, -1).detach().cpu().numpy()
    action = np.random.choice(np.arange(self._num_actions), p=probs)
    return action, probs

  def _act_epsilon_greedy(
      self,
      info_state, legal_actions, epsilon: float
  ):
    """Returns a selected action and the probabilities of legal actions."""
    if epsilon == 0:  # greedy evaluation mode
      # Soft-max normalized by the action probabilities from the previous
      # Q-network.
      _, prev_probs = self._boltzmann_action_probs(
          self._prev_q_network,
          info_state,
          legal_actions,
          torch.ones(self._num_actions, device=self._device),
      )
      return self._boltzmann_action_probs(
          self._q_network,
          info_state,
          legal_actions,
          torch.as_tensor(prev_probs, device=self._device),
      )

    # During training, we use the DQN action selection, which will be
    # epsilon-greedy.
    return super()._act_epsilon_greedy(info_state, legal_actions, epsilon)

  def update_prev_q_network(self) -> None:
    """Updates the parameters of the previous Q-network."""
    self._prev_q_network.load_state_dict(
        copy.deepcopy(self._q_network.state_dict()))
