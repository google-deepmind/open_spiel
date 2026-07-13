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

# Lint as python3
"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

from typing import Any

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

JsonDict = dict[str, Any]

_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="python_tic_tac_toe",
    long_name="Python Tic-Tac-Toe",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS)


class TicTacToeGame(pyspiel.Game):
  """A Python version of the Tic-Tac-Toe game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self, state=None):
    """Returns a state corresponding to the start of a game.

    Args:
      state: None for a fresh initial state, a dict (from to_dict), or a
        StateStruct (from to_struct) to reconstruct a state.
    """
    return TicTacToeState(self, state=state)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class TicTacToeState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""

  def __init__(
      self,
      game: TicTacToeGame,
      state: pyspiel.StateStruct | JsonDict | None = None,
  ) -> None:
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

    if state is None:
      return

    if isinstance(state, pyspiel.StateStruct):
      state = state.to_dict()
    board = state["board"]
    if len(board) != _NUM_CELLS:
      raise ValueError(
          f"Invalid board size: expected {_NUM_CELLS}, got {len(board)}"
      )
    char_map = {".": ".", "x": "x", "o": "o", "empty": "."}
    for i, cell in enumerate(board):
      self.board[_coord(i)] = char_map.get(cell, cell)
    num_x = sum(1 for c in self.board.ravel() if c == "x")
    num_o = sum(1 for c in self.board.ravel() if c == "o")
    if num_x < num_o or num_x > num_o + 1:
      raise ValueError(f"Invalid board: x={num_x}, o={num_o}")
    self._cur_player = 0 if num_x == num_o else 1

    x_wins = _has_line(self.board, "x")
    o_wins = _has_line(self.board, "o")

    if x_wins and o_wins:
      raise ValueError("Invalid board state: both players have a line.")

    if x_wins:
      if num_x != num_o + 1:
        raise ValueError(
            "Invalid board state: x has a line, but number of pieces is "
            f"inconsistent, got x={num_x}, o={num_o}"
        )
      self._is_terminal = True
      self._player0_score = 1.0
    elif o_wins:
      if num_x != num_o:
        raise ValueError(
            "Invalid board state: o has a line, but number of pieces is "
            f"inconsistent, got x={num_x}, o={num_o}"
        )
      self._is_terminal = True
      self._player0_score = -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True

    if "current_player" in state:
      if self._is_terminal:
        expected_player_str = "Terminal"
      else:
        expected_player_str = "x" if self._cur_player == 0 else "o"
      if state["current_player"] != expected_player_str:
        raise ValueError(
            f"Invalid current player: expected {expected_player_str}, "
            f"got {state['current_player']}"
        )

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"
    if _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    row, col = _coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)

  # SpielStruct API methods (optional). These return plain Python dicts that
  # the C++ trampoline wraps as DictStateStruct / DictActionStruct /
  # DictObservationStruct objects, enabling JSON serialization and the
  # full struct-based action API (validate_action_struct, apply_action_struct).

  def _to_struct(self):
    """Returns a dict representation of the state."""
    cell_names = {".": ".", "x": "x", "o": "o"}
    if self.is_terminal():
      current_player_str = "Terminal"
    else:
      current_player_str = "x" if self._cur_player == 0 else "o"
    return {
        "current_player": current_player_str,
        "board": [cell_names[self.board[_coord(i)]] for i in range(_NUM_CELLS)],
    }

  def _action_to_struct(self, player, action):
    """Converts an action integer to a dict with row/col fields."""
    del player  # Unused.
    row, col = _coord(action)
    return {"row": row, "col": col}

  def _struct_to_actions(self, action_dict):
    """Converts an action dict back to a list of integer actions."""
    return [action_dict["row"] * _NUM_COLS + action_dict["col"]]

  def _to_observation_struct(self, player):
    """Returns a dict observation from the perspective of `player`."""
    del player  # Unused.
    return self._to_struct()


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _has_line(board, player):
  """Checks if the player has a line."""
  return (
      all(board[0] == player)
      or all(board[1] == player)
      or all(board[2] == player)
      or all(board[:, 0] == player)
      or all(board[:, 1] == player)
      or all(board[:, 2] == player)
      or all(board.diagonal() == player)
      or all(np.fliplr(board).diagonal() == player)
  )


def _line_exists(board):
  """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
  if _has_line(board, "x"):
    return "x"
  if _has_line(board, "o"):
    return "o"
  return None


def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TicTacToeGame)
