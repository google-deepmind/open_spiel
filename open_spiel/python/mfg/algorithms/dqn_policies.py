# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes a set of DQN policies for a list of agents and environments."""
import numpy as np

from open_spiel.python import policy as policy_std
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn

class DQNPolicies(policy_std.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, envs, info_state_size, num_actions, **kwargs):
    game = envs[0].game
    player_ids = list(range(game.num_players()))
    super(DQNPolicies, self).__init__(game, player_ids)
    self._policies = [
        dqn.DQN(idx, info_state_size, num_actions, **kwargs)
        for idx in range(game.num_players())
    ]
    self._obs = {
        "info_state": [None] * game.num_players(),
        "legal_actions": [None] * game.num_players()
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (state.observation_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict
