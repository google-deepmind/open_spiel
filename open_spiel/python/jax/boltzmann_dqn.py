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

from functools import partial

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp

from open_spiel.python.jax import dqn


class BoltzmannDQN(dqn.DQN):
  """Boltzmann DQN implementation in JAX."""

  def __init__(self, *args, eta: float = 1.0, **kwargs):
    """Initializes the Boltzmann DQN agent.

    Args:
      *args: args passed to the underlying DQN agent.
      eta: Temperature parameter used in the softmax function.
      seed: Random seed used for action selection.
      **kwargs: kwargs passed to the underlying DQN agent.
    """
    super().__init__(*args, **kwargs)
    self._prev_q_network_state = nn.clone(
      nn.state(self._q_network), variables=True
    )
    self._temperature = eta

  @partial(jax.jit, static_argnums=(0,))
  def _boltzmann_action_probs(
    self,
    network_state: nn.State,
    info_state: chex.Array,
    legal_actions: chex.Array,
    rng: chex.PRNGKey,
    coeff: chex.Array,
  ):
    """Returns a valid soft-max action and action probabilities.

    Args:
      network_state: nn.State of the Q-network.
      info_state: Observations from the environment.
      legal_actions: List of legal actions.
      rng: chex.PRNGKey for randomness
      coeff: If not None, then the terms in softmax function will be
        element-wise multiplied with these coefficients.

    Returns:
      a valid soft-max action and action probabilities.
    """
    q_values = self._jittable_inference(network_state, info_state)
    legal_q_values = jnp.where(
      legal_actions,
      q_values,
      jnp.full_like(q_values, dqn.ILLEGAL_ACTION_LOGITS_PENALTY),
    )
    # Apply temperature and subtract the maximum value for numerical stability.
    temp = legal_q_values / self._temperature
    probs = nn.softmax(coeff * temp)
    action = jax.random.choice(rng, jnp.arange(self._num_actions), p=probs)
    return action, probs

  def _act_epsilon_greedy(
    self,
    network_state: nn.State,
    info_state: chex.Array,
    legal_actions: chex.Array,
    rng: chex.PRNGKey,
    epsilon: float,
  ):
    """Returns a selected action and the probabilities of legal actions."""
    if epsilon == 0.0:  # greeddy evaluation
      # Soft-max normalized by the action probabilities from the previous
      # Q-network.
      prev_rng, rng = jax.random.split(rng)
      _, prev_probs = self._boltzmann_action_probs(
        self._prev_q_network_state,
        info_state,
        legal_actions,
        prev_rng,
        jnp.ones(self._num_actions),
      )
      return self._boltzmann_action_probs(
        network_state, info_state, legal_actions, rng, prev_probs
      )

    # During training, we use the DQN action selection, which will be
    # epsilon-greedy.
    return super()._act_epsilon_greedy(
      network_state, info_state, legal_actions, rng, epsilon
    )

  def update_prev_q_network(self):
    """Updates the parameters of the previous Q-network."""
    self._prev_q_network_state = nn.clone(
      nn.state(self._q_network), variables=True
    )
