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

# Lint as python3
"""An example game written in Python.

Games are strongly encouraged to be written in C++ for efficiency and so that
they can be available from all APIs. Writing Python-only games means that they
they will be slow and only be accessible from Python algorithms.

Nonetheless, there are many cases where this is desirable (or required), hence
we provide this example.
"""

import pickle
import pyspiel

# pylint: disable=function-redefined
# pylint: disable=unused-argument

_GAME_NOT_DONE = -1
_GAME_TIED = 2


class TicTacToeState(object):
  """A python-only version of the Tic-Tac-Toe state.

  This class implements all the pyspiel.State API functions. Please see spiel.h
  for more thorough documentation of each function.

  Note that this class does not inherit from pyspiel.State since pickle
  serialization is not possible due to what is required on the C++ side
  (backpointers to the C++ game object, which we can't get from here).
  """

  def __init__(self, game):
    self._rows = 3
    self._cols = 3
    self._game = game
    self.set_state(
        cur_player=0, winner=_GAME_NOT_DONE, history=[], board=["."] * 9)

  # Helper functions (not part of the OpenSpiel API).

  def set_state(self, cur_player, winner, history, board):
    self._cur_player = cur_player
    self._winner = winner
    self._history = history
    self._board = board

  def board(self, row, col):
    return self._board[row * self._cols + col]

  def coord(self, move):
    return (move // self._cols, move % self._cols)

  def has_row(self, row):
    if (self.board(row, 0) != "." and
        self.board(row, 0) == self.board(row, 1) and
        self.board(row, 1) == self.board(row, 2)):
      return self.board(row, 0)
    else:
      return "."

  def has_col(self, col):
    if (self.board(0, col) != "." and
        self.board(0, col) == self.board(1, col) and
        self.board(1, col) == self.board(2, col)):
      return self.board(0, col)
    else:
      return "."

  def has_slash(self):
    if (self.board(2, 0) != "." and self.board(2, 0) == self.board(1, 1) and
        self.board(1, 1) == self.board(0, 1)):
      return self.board(2, 0)
    else:
      return "."

  def has_backslash(self):
    if (self.board(0, 0) != "." and self.board(0, 0) == self.board(1, 1) and
        self.board(1, 1) == self.board(2, 2)):
      return self.board(0, 0)
    else:
      return "."

  def line_exists(self):
    """Checks if a line exists, returns "x" or "o" if so, and "." otherwise."""
    for row in range(3):
      winner = self.has_row(row)
      if winner != ".":
        return winner
    for col in range(3):
      winner = self.has_col(col)
      if winner != ".":
        return winner
    winner = self.has_backslash()
    if winner != ".":
      return winner
    winner = self.has_slash()
    if winner != ".":
      return winner
    return "."

  # OpenSpiel (PySpiel) API functions are below. These need to be provided by
  # every game.

  def current_player(self):
    return self._cur_player

  def legal_actions(self, player=None):
    """Get a list of legal actions.

    Args:
      player: the player whose legal moves

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actios - 1],
      at non-terminal states, and empty list at terminal states.
    """

    if player is not None and player != self._cur_player:
      return []
    elif self.is_terminal():
      return []
    else:
      actions = []
      for i in range(9):
        if self._board[i] == ".":
          actions.append(i)
      return actions

  def legal_actions_mask(self, player=None):
    """Get a list of legal actions.

    Args:
      player: the player whose legal moves

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actios - 1],
      at non-terminal states, and empty list at terminal states.
    """
    if player is not None and player != self._cur_player:
      return []
    elif self.is_terminal():
      return []
    else:
      action_mask = [0] * 9
      legal_actions = self.legal_actions()
      for action in legal_actions:
        action_mask[action] = 1
      return action_mask

  def apply_action(self, action):
    """Apply the specific action to change the state."""
    if self._cur_player == 0:
      self._board[action] = "x"
    elif self._cur_player == 1:
      self._board[action] = "o"
    self._cur_player = 1 - self._cur_player
    winner = self.line_exists()
    if winner == "x":
      self._winner = 0
    elif winner == "o":
      self._winner = 1
    self._history.append(action)
    if self._winner < 0 and len(self._history) == 9:
      self._winner = _GAME_TIED

  def undo_action(self, action):
    # Optional function. Not used in many places.
    self._board[action] = "."
    self._cur_player = 1 - self._cur_player
    self._history.pop()
    self._winner = _GAME_NOT_DONE

  def action_to_string(self, player, action):
    return self.action_to_string(action)

  def action_to_string(self, action):
    coord = self.coord(action)
    return "{}({},{})".format("x" if self._cur_player == 0 else "y",
                              coord[0], coord[1])

  def string_to_action(self, player, string):
    # This method is not used in many places, likely ok to leave out.
    return self.string_to_action(string)

  def string_to_action(self, string):
    # This method is not used in many places, likely ok to leave out.
    parts = string.replace("x", "").replace("y", "").replace("(", "").replace(
        ")", "").split(",")
    return self._rows * int(parts[0]) + int(parts[1])

  def is_terminal(self):
    return self._winner != _GAME_NOT_DONE

  def returns(self):
    if self.is_terminal():
      if self._winner == 0:
        return [1, -1]
      elif self._winner == 1:
        return [-1, 1]
    return [0, 0]

  def rewards(self):
    return self.returns()

  def player_reward(self, player):
    return self.rewards()[player]

  def player_returns(self, player):
    return self.returns()[player]

  def is_chance_node(self):
    return False

  def is_simultaneous_node(self):
    return False

  def history(self):
    return self._history

  def history_str(self):
    return str(self._history)

  def information_state_string(self, player=None):
    return self.history_str()

  def information_state_tensor(self, player=None):
    # TODO(author5): implement this method.
    assert False, "Unimplemented"
    return []

  def observation_string(self, player=None):
    return str(self)

  def observation_tensor(self):
    # TODO(author5): implement this method.
    assert False, "Unimplemented"

  def child(self, action):
    cloned_state = self.clone()
    cloned_state.apply_action(action)
    return cloned_state

  def apply_actions(self, actions):
    # Only applies to simultaneous move games
    pass

  def num_distinct_actions(self):
    return 9

  def num_players(self):
    return 2

  def chance_outcomes(self):
    return []

  def get_game(self):
    return self._game

  def get_type(self):
    return self._game.get_type()

  def serialize(self):
    return pickle.dumps(self)

  def resample_from_infostate(self):
    return [self.clone()]

  def __str__(self):
    board_str = ""
    for row in range(self._rows):
      for col in range(self._cols):
        board_str += self.board(row, col)
      board_str += "\n"
    return board_str

  def clone(self):
    cloned_state = TicTacToeState(self._game)
    cloned_state.set_state(self._cur_player, self._winner, self._history[:],
                           self._board[:])
    return cloned_state


class TicTacToeGame(object):
  """A python-only version of the Tic-Tac-Toe game.

  This class implements all the pyspiel.Gae API functions. Please see spiel.h
  for more thorough documentation of each function.

  Note that this class does not inherit from pyspiel.Game since pickle
  serialization is not possible due to what is required on the C++ side
  (backpointers to the C++ game object, which we can't get from here).
  """

  def __init__(self):
    pass

  def new_initial_state(self):
    return TicTacToeState(self)

  def num_distinct_actions(self):
    return 9

  def clone(self):
    return TicTacToeGame()

  def max_chance_outcomes(self):
    return 0

  def get_parameters(self):
    # No parameters in TicTacToe
    return pyspiel.Parameters()

  def num_players(self):
    return 2

  def min_utility(self):
    return -1

  def max_utility(self):
    return 1

  def get_type(self):
    return pyspiel.GameType(
        "python_tic_tac_toe",
        "Python Tic-Tac-Toe",
        pyspiel.GameType.Dynamics.SEQUENTIAL,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.PERFECT_INFORMATION,
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        False,  # provides_information_state_tensor
        True,  # provides_observation
        False,  # provides_observation_tensor
        {}  # parameter_specification
    )

  def utility_sum(self):
    return 0

  def observation_tensor_shape(self):
    # Only define observation tensors for Tic-Tac-Toe
    # TODO(author5): implement me
    pass

  def observation_tensor_size(self):
    # Only define observation tensors for Tic-Tac-Toe
    # TODO(author5): implement me
    pass

  def deserialize_state(self, string):
    return pickle.loads(string)

  def max_game_length(self):
    return 9

  def __str__(self):
    return "python_tic_tac_toe"
