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

# Lint as: python3
"""Perturbates any policy with tabular-saved, fixed noise.

The policy's probabilities P' on each state s are computed as

P'(s) = alpha * epsilon + (1-alpha) * P(s),

with P the former policy's probabilities, and epsilon ~ Softmax(beta *
Uniform)
"""

import numpy as np
from open_spiel.python import policy as openspiel_policy


class NoisyPolicy(openspiel_policy.Policy):
  """Pyspiel Best Response with added noise.

    This policy's probabilities P' on each `player_id` state s is computed as
    P'(s) = alpha * epsilon + (1-alpha) * P(s),

    with P the former policy's probabilities, and epsilon ~ Softmax(beta *
    Uniform)
  """

  def __init__(self, policy, player_id=None, alpha=0.1, beta=1.0):
    """Initializes the noisy policy.

    Note that this noise only affects `player_id`.

    Args:
      policy: Any OpenSpiel `policy.Policy` object.
      player_id: The player id, the policy of whom will be made noisy. If `None`
        noise will be added to the policies for all players.
      alpha: Mixing noise factor.
      beta: Softmax inverse temperature factor.
    """
    self._policy = policy
    self.game = policy.game
    self.game_type = self.game.get_type()

    self.player_id = player_id

    self._noise_dict = {}
    self._alpha = alpha
    self._beta = beta

  def _state_key(self, state, player):
    """Returns the key to use to look up this (state, player) pair."""
    if self.game_type.provides_information_state_string:
      if player is None:
        return state.information_state_string()
      else:
        return state.information_state_string(player)
    elif self.game_type.provides_observation_string:
      if player is None:
        return state.observation_string()
      else:
        return state.observation_string(player)
    else:
      return str(state)

  def get_or_create_noise(self, state, player_id=None):
    """Get noisy policy or create it and return it.

    Args:
      state: the state to which the policy will be applied.
      player_id: the player id that will apply the noisy policy. Default to
        current_player. Should be defined in the case of simultaneous games.

    Returns:
      noise_action_probs: The noisy probability distribution on the set of legal
        actions.
    """
    if player_id is None:
      player_id = state.current_player()
    info_state = self._state_key(state, player_id)
    if info_state not in self._noise_dict:
      action_ids = state.legal_actions(player_id)
      noise = self._beta * np.random.normal(size=len(action_ids))
      noise = np.exp(noise - noise.max())
      noise /= np.sum(noise)

      self._noise_dict[info_state] = {
          action_ids[i]: noise[i] for i in range(len(noise))
      }
    return self._noise_dict[info_state]

  def mix_probs(self, probs, noise_probs):
    return {
        i: (1 - self._alpha) * probs[i] + self._alpha * noise_probs[i]
        for i in probs
    }

  @property
  def policy(self):
    return self._policy

  def action_probabilities(self, state, player_id=None):
    """Returns the policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultabeous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """

    # If self._player_id is None, or if self.player_id == current_player, add
    # noise.
    if ((self.player_id is None) or
        (state.current_player() == self.player_id) or
        (player_id == self.player_id)):
      noise_probs = self.get_or_create_noise(state, player_id)
      probs = self._policy.action_probabilities(state, player_id)
      probs = self.mix_probs(probs, noise_probs)
      return probs

    # Send the default probabilities for all other players
    return self._policy.action_probabilities(state, player_id)
