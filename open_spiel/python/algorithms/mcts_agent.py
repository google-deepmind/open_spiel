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
"""An RL agent wrapper for the MCTS bot."""

import numpy as np
from open_spiel.python import rl_agent
import pyspiel


class MCTSAgent(rl_agent.AbstractAgent):
  """MCTS agent class.

  Important note: this agent requires the environment to provide the full state
  in its TimeStep objects. Hence, the environment must be created with the
  use_full_state flag set to True, and the state must be serializable.
  """

  def __init__(self, player_id, num_actions, mcts_bot, name="mcts_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._mcts_bot = mcts_bot
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    assert "serialized_state" in time_step.observations
    _, state = pyspiel.deserialize_game_and_state(
        time_step.observations["serialized_state"])

    # Call the MCTS bot's step to get the action.
    probs = np.zeros(self._num_actions)
    action = self._mcts_bot.step(state)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)
