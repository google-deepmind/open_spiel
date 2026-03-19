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

"""A solver for small two-player zero-sum perfect information game."""

import collections
import numpy as np
import pyspiel

TOLERANCE = 1e-6

TableEntry = collections.namedtuple("Entry", ["action_values", "value"])


class MinimaxSolver:
  """A minimax oracle object."""

  def __init__(self, game_string, epsilon: float = 0.0):
    """A minimax oracle object.

    Args:
      game_string: The game string to solve.
      epsilon: The epsilon value to use for the minimax solver (probability
          of exploring a random action). Defaults to minimax-optimal policy if
          epsilon is 0, but can be used to solve under the assumption of a
          "trembling hand".
    """

    self._game_string = game_string
    self._epsilon = epsilon
    self._game = pyspiel.load_game(game_string)
    assert self._game.num_players() == 2, "Game must have two players."
    game_type = self._game.get_type()
    assert (
        game_type.information
        == pyspiel.GameType.Information.PERFECT_INFORMATION
    )
    assert game_type.utility == pyspiel.GameType.Utility.ZERO_SUM
    assert game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    self._default_utility = self._game.min_utility() - 1
    self._num_distinct_actions = self._game.num_distinct_actions()
    # Save the best action values and the best value for each state.
    self._table: dict[str, TableEntry] = {}

  def solve(self):
    """Solve the minimax problem."""
    state = self._game.new_initial_state()
    self._create_table(state)
    return self._table

  def _create_table(self, state: pyspiel.State) -> np.ndarray:
    """Recursively compute the minimax values and store them in the table."""
    assert not state.is_terminal()
    assert not state.is_chance_node()
    state_key = str(state)
    player = state.current_player()
    if state_key in self._table:
      values = [0.0, 0.0]
      values[player] = self._table[state_key].value
      values[1 - player] = -values[player]
      return np.asarray(values)
    action_values = np.zeros(self._num_distinct_actions)
    action_values.fill(self._default_utility)
    legal_actions = state.legal_actions()
    best_action_indices = []
    best_value = self._default_utility
    for action in legal_actions:
      child_state = state.child(action)
      if child_state.is_terminal():
        action_values[action] = child_state.player_return(player)
      else:
        child_values = self._create_table(child_state)
        action_values[action] = child_values[player]
      if action_values[action] > best_value + TOLERANCE:
        best_value = action_values[action]
        best_action_indices = [action]
      elif abs(action_values[action] - best_value) < TOLERANCE:
        best_action_indices.append(action)
    # Values for this state.
    values = [0.0, 0.0]
    if self._epsilon == 0:
      # Simple max case.
      values[player] = action_values.max()
    else:
      # Epsilon trembling-hand case. Backup the expected value under the
      # epsilon-greedy policy.
      assert len(best_action_indices) > 0
      uniform_policy = np.zeros(self._num_distinct_actions)
      uniform_policy[legal_actions] = 1.0 / len(legal_actions)
      greedy_policy = np.zeros(self._num_distinct_actions)
      greedy_policy[best_action_indices] = 1.0 / len(best_action_indices)
      eps_greedy_policy = ((1 - self._epsilon) * greedy_policy +
                           self._epsilon * uniform_policy)
      values[player] = np.dot(eps_greedy_policy, action_values)
    self._table[state_key] = TableEntry(action_values, values[player])
    values[1 - player] = -values[player]
    return np.asarray(values)

  def action_values_from_string(self, state_string_key: str) -> np.ndarray:
    return self._table[state_string_key].action_values

  def action_values_from_state(self, state: pyspiel.State) -> np.ndarray:
    return self.action_values_from_string(str(state))

  def values_from_string(self, state_string_key: str) -> np.ndarray:
    return self._table[state_string_key].value

  def values_from_state(self, state: pyspiel.State) -> np.ndarray:
    return self.action_values_from_string(str(state))

