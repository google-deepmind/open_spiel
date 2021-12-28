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

"""A bot that asks the user which action to play."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import pyspiel

_MAX_WIDTH = int(os.getenv("COLUMNS", "80"))  # Get your TTY width.


def _print_columns(strings):
  """Prints a list of strings in columns."""
  padding = 2
  longest = max(len(s) for s in strings)
  max_columns = math.floor((_MAX_WIDTH - 1) / (longest + 2 * padding))
  rows = math.ceil(len(strings) / max_columns)
  columns = math.ceil(len(strings) / rows)  # Might not fill all max_columns.
  for r in range(rows):
    for c in range(columns):
      i = r + c * rows
      if i < len(strings):
        print(" " * padding + strings[i].ljust(longest + padding), end="")
    print()


class HumanBot(pyspiel.Bot):
  """Asks the user which action to play."""

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state."""
    legal_actions = state.legal_actions(state.current_player())
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]

    action_map = {
        state.action_to_string(state.current_player(), action): action
        for action in legal_actions
    }

    while True:
      action_str = input("Choose an action (empty to print legal actions): ")

      if not action_str:
        print("Legal actions(s):")
        longest_num = max(len(str(action)) for action in legal_actions)
        _print_columns([
            "{}: {}".format(str(action).rjust(longest_num), action_str)
            for action_str, action in sorted(action_map.items())
        ])
        continue

      if action_str in action_map:
        return policy, action_map[action_str]

      try:
        action = int(action_str)
      except ValueError:
        print("Could not parse the action:", action_str)
        continue

      if action in legal_actions:
        return policy, action

      print("Illegal action selected:", action_str)

  def step(self, state):
    return self.step_with_policy(state)[1]

  def restart_at(self, state):
    pass

