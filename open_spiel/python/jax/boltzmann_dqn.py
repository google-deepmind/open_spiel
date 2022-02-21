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

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax import dqn


class BoltzmannDQN(dqn.DQN):
  """Boltzmann DQN implementation in JAX."""

  def __init__(self, *args, eta: float = 1.0, seed: int = 42, **kwargs):
    """Initializes the Boltzmann DQN agent.

    Args:
      *args: args passed to the underlying DQN agent.
      eta: Temperature parameter used in the softmax function.
      seed: Random seed used for action selection.
      **kwargs: kwargs passed to the underlying DQN agent.
    """
    self._eta = eta
    self._rs = np.random.RandomState(seed)  # Used to select actions.
    super().__init__(*args, seed=seed, **kwargs)

  def _create_networks(self, rng, state_representation_size):
    """Called to create the networks."""
    # We use the DQN networks and an additional network for the fixed policy.
    super()._create_networks(rng, state_representation_size)
    self.params_prev_q_network = self.hk_network.init(
        rng, jnp.ones([1, state_representation_size]))

  def _softmax_action_probs(self,
                            params,
                            info_state,
                            legal_actions,
                            coeff=None):
    """Returns a valid soft-max action and action probabilities.

    Args:
      params: Parameters of the Q-network.
      info_state: Observations from the environment.
      legal_actions: List of legal actions.
      coeff: If not None, then the terms in softmax function will be
        element-wise multiplied with these coefficients.

    Returns:
      a valid soft-max action and action probabilities.
    """
    info_state = np.reshape(info_state, [1, -1])
    q_values = self.hk_network_apply(params, info_state)[0]
    legal_one_hot = self._to_one_hot(legal_actions)
    legal_q_values = (
        q_values + (1 - legal_one_hot) * dqn.ILLEGAL_ACTION_LOGITS_PENALTY)
    # Apply temperature and subtract the maximum value for numerical stability.
    temp = legal_q_values / self._eta
    unnormalized = np.exp(temp - np.amax(temp))
    if coeff is not None:
      unnormalized = np.multiply(coeff, unnormalized)
    probs = unnormalized / unnormalized.sum()
    action = self._rs.choice(legal_actions, p=probs[legal_actions])
    return action, probs

  def _get_action_probs(self, info_state, legal_actions, is_evaluation=False):
    """Returns a selected action and the probabilities of legal actions."""
    if is_evaluation:
      # Soft-max normalized by the action probabilities from the previous
      # Q-network.
      _, prev_probs = self._softmax_action_probs(self.params_prev_q_network,
                                                 info_state, legal_actions)
      return self._softmax_action_probs(self.params_q_network, info_state,
                                        legal_actions, prev_probs)

    # During training, we use the DQN action selection, which will be
    # epsilon-greedy.
    return super()._get_action_probs(
        info_state, legal_actions, is_evaluation=False)

  def update_prev_q_network(self):
    """Updates the parameters of the previous Q-network."""
    self.params_prev_q_network = jax.tree_multimap(lambda x: x.copy(),
                                                   self.params_q_network)
