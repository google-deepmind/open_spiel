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

"""Computes a softmax policy from a value function."""
from typing import Optional

import numpy as np

from open_spiel.python import policy
from open_spiel.python.mfg import value


class SoftmaxPolicy(policy.Policy):
  """Computes the softmax policy of a value function."""

  def __init__(self,
               game,
               player_ids,
               temperature: float,
               state_action_value: value.ValueFunction,
               prior_policy: Optional[policy.Policy] = None):
    """Initializes the softmax policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each
        should be in the range 0..game.num_players()-1.
      temperature: float to scale the values (multiplied by 1/temperature).
      state_action_value: A state-action value function.
      prior_policy: Optional argument. Prior policy to scale the softmax
        policy.
    """
    super(SoftmaxPolicy, self).__init__(game, player_ids)
    self._state_action_value = state_action_value
    self._prior_policy = prior_policy
    self._temperature = temperature

  def action_probabilities(self, state, player_id=None):
    legal_actions = state.legal_actions()
    max_q = np.max(
        [self._state_action_value(state, action) for action in legal_actions])
    exp_q = [
        np.exp((self._state_action_value(state, action) - max_q) /
               self._temperature) for action in legal_actions
    ]
    if self._prior_policy is not None:
      prior_probs = self._prior_policy.action_probabilities(state)
      exp_q = [
          prior_probs.get(action, 0) * exp_q[i]
          for i, action in enumerate(legal_actions)
      ]
    denom = sum(exp_q)
    smax_q = exp_q if denom == 0 else exp_q / denom
    return dict(zip(legal_actions, smax_q))
