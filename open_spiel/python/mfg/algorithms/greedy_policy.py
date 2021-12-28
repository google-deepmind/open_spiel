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

"""Computes a greedy policy from a value."""
import numpy as np

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value


class GreedyPolicy(policy_std.Policy):
  """Computes the greedy policy of a value."""

  def __init__(self, game, player_ids, state_action_value: value.ValueFunction):
    """Initializes the greedy policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
      state_action_value: A state-action value function.
    """
    super(GreedyPolicy, self).__init__(game, player_ids)
    self._state_action_value = state_action_value

  def action_probabilities(self, state, player_id=None):
    q = [
        self._state_action_value(state, action)
        for action in state.legal_actions()
    ]
    amax_q = [0.0 for _ in state.legal_actions()]
    amax_q[np.argmax(q)] = 1.0
    return dict(zip(state.legal_actions(), amax_q))
