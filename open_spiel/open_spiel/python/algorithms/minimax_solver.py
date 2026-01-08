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

import numpy as np
import pyspiel


class MinimaxSolver:
  """A minimax oracle object."""

  def __init__(self, game_string):
    self._game_string = game_string
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
    self._table: dict[str, np.ndarray] = {}

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
      values[player] = self._table[state_key].max()
      values[1 - player] = -values[player]
      return np.asarray(values)
    action_values = np.zeros(self._num_distinct_actions)
    action_values.fill(self._default_utility)
    for action in state.legal_actions():
      child_state = state.child(action)
      if child_state.is_terminal():
        action_values[action] = child_state.player_return(player)
      else:
        child_values = self._create_table(child_state)
        action_values[action] = child_values[player]
    self._table[state_key] = action_values
    values = [0.0, 0.0]
    values[player] = action_values.max()
    values[1 - player] = -values[player]
    return np.asarray(values)

  def action_values_from_string(self, state_string_key: str) -> np.ndarray:
    return self._table[state_string_key]

  def action_values_from_state(self, state: pyspiel.State) -> np.ndarray:
    return self.action_values_from_string(str(state))
