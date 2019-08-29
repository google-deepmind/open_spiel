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

"""A bot that asks the user which action to play."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel


class HumanBot(pyspiel.Bot):
  """Asks the user which action to play."""

  def __init__(self, game, player_id):
    """Initializes a human bot. """
    super(HumanBot, self).__init__(game, player_id)

  def step(self, state):
    """Returns the stochastic policy and selected action in the given state. """
    legal_actions = state.legal_actions(self.player_id())
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]

    print("Legal actions(s):")
    for action in legal_actions:
        print(" ", action, "=", state.action_to_string(state.current_player(), action))
    action = -1
    while action not in legal_actions:
      action_str = input("Choose an action: ")
      try:
        action = int(action_str)
      except ValueError:
        continue
      return policy, action
