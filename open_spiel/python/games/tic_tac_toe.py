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

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS


def _line_value(line):
  """Checks a possible line, returning the winning symbol if any."""
  if all(line == "x") or all(line == "o"):
    return line[0]


class TicTacToeState(object):
  """A python-only version of the Tic-Tac-Toe state.

  This class implements all the pyspiel.State API functions. Please see spiel.h
  for more thorough documentation of each function.

  Note that this class does not inherit from pyspiel.State since pickle
  serialization is not possible due to what is required on the C++ side
  (backpointers to the C++ game object, which we can't get from here).
  """

  def __init__(self, game):
    self._game = game
    self.set_state(
        cur_player=0,
        winner=None,
        is_terminal=False,
        history=[],
        board=np.full((_NUM_ROWS, _NUM_COLS), "."))

  # Helper functions (not part of the OpenSpiel API).

  def set_state(self, cur_player, winner, is_terminal, history, board):
    self._cur_player = cur_player
    self._winner = winner
    self._is_terminal = is_terminal
    self._history = history
    self._board = board

  def coord(self, move):
    return (move // _NUM_COLS, move % _NUM_COLS)

  def line_exists(self):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    return (_line_value(self._board[0]) or _line_value(self._board[1]) or
            _line_value(self._board[2]) or _line_value(self._board[:, 0]) or
            _line_value(self._board[:, 1]) or _line_value(self._board[:, 2]) or
            _line_value(self._board.diagonal()) or
            _line_value(np.fliplr(self._board).diagonal()))

  # OpenSpiel (PySpiel) API functions are below. These need to be provided by
  # every game. Some not-often-used methods have been omitted.

  def current_player(self):
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def legal_actions(self, player=None):
    """Returns a list of legal actions, sorted in ascending order.

    Args:
      player: the player whose legal moves

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actions - 1]
      at non-terminal states, and empty list at terminal states.
    """

    if player is not None and player != self._cur_player:
      return []
    elif self.is_terminal():
      return []
    else:
      actions = []
      for action in range(_NUM_CELLS):
        if self._board[self.coord(action)] == ".":
          actions.append(action)
      return actions

  def legal_actions_mask(self, player=None):
    """Get a list of legal actions.

    Args:
      player: the player whose moves we want; defaults to the current player.

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actios - 1].
      Returns an empty list at terminal states, or if it is not the specified
      player's turn.
    """
    if player is not None and player != self._cur_player:
      return []
    elif self.is_terminal():
      return []
    else:
      action_mask = [0] * _NUM_CELLS
      for action in self.legal_actions():
        action_mask[action] = 1
      return action_mask

  def apply_action(self, action):
    """Applies the specified action to the state."""
    self._board[self.coord(action)] = "x" if self._cur_player == 0 else "o"
    self._history.append(action)
    if self.line_exists():
      self._is_terminal = True
      self._winner = self._cur_player
    elif len(self._history) == _NUM_CELLS:
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def undo_action(self, action):
    # Optional function. Not used in many places.
    self._board[self.coord(action)] = "."
    self._cur_player = 1 - self._cur_player
    self._history.pop()
    self._winner = None
    self._is_terminal = False

  def action_to_string(self, arg0, arg1=None):
    """Action -> string. Args either (player, action) or (action)."""
    player = self.current_player() if arg1 is None else arg0
    action = arg0 if arg1 is None else arg1
    row, col = self.coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    if self.is_terminal():
      if self._winner == 0:
        return [1.0, -1.0]
      elif self._winner == 1:
        return [-1.0, 1.0]
    return [0.0, 0.0]

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
    del player  # Same information state for both players.
    return self.history_str()

  def information_state_tensor(self, player=None):
    raise NotImplementedError

  def observation_string(self, player=None):
    del player  # Same observation for both players.
    return str(self)

  def observation_tensor(self, player=None):
    del player  # Same observation for both players.
    observation = np.zeros((1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS))
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        index = ".ox".index(self._board[row, col])
        observation[index, row, col] = 1.0
    return list(observation.flatten())

  def child(self, action):
    cloned_state = self.clone()
    cloned_state.apply_action(action)
    return cloned_state

  def apply_actions(self, actions):
    raise NotImplementedError  # Only applies to simultaneous move games

  def num_distinct_actions(self):
    return _NUM_CELLS

  def num_players(self):
    return _NUM_PLAYERS

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
    return "\n".join("".join(row) for row in self._board)

  def clone(self):
    cloned_state = TicTacToeState(self._game)
    cloned_state.set_state(self._cur_player, self._winner, self._is_terminal,
                           self._history[:], np.array(self._board))
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
    return _NUM_CELLS

  def policy_tensor_shape(self):
    return (_NUM_ROWS, _NUM_COLS, 1)

  def clone(self):
    return TicTacToeGame()

  def max_chance_outcomes(self):
    return 0

  def get_parameters(self):
    return {}

  def num_players(self):
    return _NUM_PLAYERS

  def min_utility(self):
    return -1.0

  def max_utility(self):
    return 1.0

  def get_type(self):
    return pyspiel.GameType(
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
        parameter_specification={},
    )

  def utility_sum(self):
    return 0.0

  def observation_tensor_shape(self):
    return [1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS]

  def observation_tensor_layout(self):
    return pyspiel.TensorLayout.CHW

  def observation_tensor_size(self):
    return np.product(self.observation_tensor_shape())

  def deserialize_state(self, string):
    return pickle.loads(string)

  def max_game_length(self):
    return _NUM_CELLS

  def __str__(self):
    return "python_tic_tac_toe"
