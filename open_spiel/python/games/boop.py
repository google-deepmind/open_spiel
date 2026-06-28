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

"""Boop board game, implemented in Python.

Boop is a 2-player abstract strategy game on a 6x6 grid. Players alternate
placing pieces (kittens and cats) on the board. Placing a piece boops
(pushes) all orthogonally and diagonally adjacent pieces one step away.
Getting three of your cats in a row wins. Three kittens in a row promotes
them: those kittens return to hand and the player earns cats.

Rules:
  - 6x6 board, 2 players
  - Each player starts with 8 kittens in hand, 0 cats in hand
  - Turn: place 1 piece (kitten or cat) from hand onto any empty cell
  - Boop: push all adjacent pieces 1 step away from the placed piece
    - Kittens push only kittens; cats push both kittens and cats
    - If pushed off the board edge, the piece returns to its owner's hand
    - If push destination is occupied, the piece stays (no chain reaction)
  - Win: 3 of your cats in a row (any of 8 directions)
  - Promote: 3 of your kittens in a row -> remove from board (back to hand),
    earn cats (capped at 6 total cats per player across hand + board)
  - Draw: game reaches 500 moves without a winner

Action encoding: piece_type * 36 + row * 6 + col
  piece_type 0 = kitten, piece_type 1 = cat

Observation tensor (184 floats):
  5 planes of 6x6 = 180: [empty, my_kitten, my_cat, opp_kitten, opp_cat]
  4 scalars: [my_kittens/8, my_cats/6, opp_kittens/8, opp_cats/6]
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_ROWS = 6
_COLS = 6
_NUM_CELLS = _ROWS * _COLS           # 36
_NUM_PIECE_TYPES = 2                 # 0=kitten, 1=cat
_NUM_ACTIONS = _NUM_PIECE_TYPES * _NUM_CELLS  # 72
_MAX_KITTENS = 8
_MAX_CATS = 6
_MAX_GAME_LENGTH = 500

# Board cell encoding
_EMPTY = 0
_P0_KITTEN = 1
_P0_CAT = 2
_P1_KITTEN = 3
_P1_CAT = 4

# Lookup tables keyed by player index
_KITTEN_VAL = [_P0_KITTEN, _P1_KITTEN]
_CAT_VAL = [_P0_CAT, _P1_CAT]
_PIECE_VALS = [[_P0_KITTEN, _P0_CAT], [_P1_KITTEN, _P1_CAT]]

_GAME_TYPE = pyspiel.GameType(
    short_name='python_boop',
    long_name='Python Boop',
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
    num_distinct_actions=_NUM_ACTIONS,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH)


class BoopGame(pyspiel.Game):
  """A Python version of the Boop board game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    return BoopState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoopObserver(params)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)


class BoopState(pyspiel.State):
  """State for the Boop game."""

  def __init__(self, game):
    super().__init__(game)
    self._cur_player = 0
    self._is_terminal = False
    self._winner = None
    self._move_count = 0
    # 6x6 board; values: 0=empty, 1=P0 kitten, 2=P0 cat, 3=P1 kitten, 4=P1 cat
    self.board = np.zeros((_ROWS, _COLS), dtype=np.int8)
    # _hand[player][piece_type]: piece_type 0=kitten, 1=cat
    self._hand = [[_MAX_KITTENS, 0], [_MAX_KITTENS, 0]]

  # ---- OpenSpiel API ----

  def current_player(self):
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    actions = []
    for r in range(_ROWS):
      for c in range(_COLS):
        if self.board[r, c] == _EMPTY:
          cell = r * _COLS + c
          if self._hand[player][0] > 0:
            actions.append(cell)               # place kitten
          if self._hand[player][1] > 0:
            actions.append(_NUM_CELLS + cell)  # place cat
    return sorted(actions)

  def _apply_action(self, action):
    piece_type = action // _NUM_CELLS   # 0=kitten, 1=cat
    cell = action % _NUM_CELLS
    r, c = cell // _COLS, cell % _COLS
    p = self._cur_player

    # Place piece on board
    self._hand[p][piece_type] -= 1
    self.board[r, c] = _PIECE_VALS[p][piece_type]

    # Boop adjacent pieces
    self._boop(r, c, is_cat=(piece_type == 1))

    self._move_count += 1

    # Draw condition
    if self._move_count >= _MAX_GAME_LENGTH:
      self._is_terminal = True
      return

    # Check win immediately after boop (cats in a row)
    for player in (p, 1 - p):
      if self._check_win(player):
        self._is_terminal = True
        self._winner = player
        return

    # Promote 3-in-a-row kittens for both players
    self._promote_kittens(p)
    self._promote_kittens(1 - p)

    # Check win again after promotions (safe guard)
    for player in (p, 1 - p):
      if self._check_win(player):
        self._is_terminal = True
        self._winner = player
        return

    self._cur_player = 1 - p

  def _action_to_string(self, player, action):
    pt = action // _NUM_CELLS
    cell = action % _NUM_CELLS
    r, c = cell // _COLS, cell % _COLS
    piece = 'cat' if pt else 'kitten'
    return f'p{player}:{piece}@({r},{c})'

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    if self._winner == 0:
      return [1.0, -1.0]
    if self._winner == 1:
      return [-1.0, 1.0]
    return [0.0, 0.0]

  def __str__(self):
    syms = {
        _EMPTY: '.', _P0_KITTEN: 'k', _P0_CAT: 'K',
        _P1_KITTEN: 'o', _P1_CAT: 'O',
    }
    rows = [
        ''.join(syms[self.board[r, c]] for c in range(_COLS))
        for r in range(_ROWS)
    ]
    rows.append(
        f'P0: {self._hand[0][0]}k {self._hand[0][1]}K  '
        f'P1: {self._hand[1][0]}k {self._hand[1][1]}K  '
        f'move={self._move_count}')
    return '\n'.join(rows)

  # ---- Game logic ----

  def _boop(self, r, c, is_cat):
    """Push all pieces adjacent to (r,c) one step away from it."""
    for dr in (-1, 0, 1):
      for dc in (-1, 0, 1):
        if dr == 0 and dc == 0:
          continue
        nr, nc = r + dr, c + dc
        if not (0 <= nr < _ROWS and 0 <= nc < _COLS):
          continue
        neighbor = self.board[nr, nc]
        if neighbor == _EMPTY:
          continue
        neighbor_is_cat = neighbor in (_P0_CAT, _P1_CAT)
        # Kittens cannot push cats
        if not is_cat and neighbor_is_cat:
          continue
        dest_r, dest_c = nr + dr, nc + dc
        owner = 0 if neighbor in (_P0_KITTEN, _P0_CAT) else 1
        n_type = 1 if neighbor_is_cat else 0
        if not (0 <= dest_r < _ROWS and 0 <= dest_c < _COLS):
          # Pushed off the board: return piece to owner's hand
          self.board[nr, nc] = _EMPTY
          self._hand[owner][n_type] += 1
        elif self.board[dest_r, dest_c] == _EMPTY:
          # Slide piece to empty destination
          self.board[dest_r, dest_c] = neighbor
          self.board[nr, nc] = _EMPTY
        # If destination occupied, piece is blocked (stays in place)

  def _promote_kittens(self, player):
    """Find 3-in-a-row kittens, remove them, and earn cats (up to cap)."""
    kitten_val = _KITTEN_VAL[player]
    cat_val = _CAT_VAL[player]
    to_promote = set()
    for r in range(_ROWS):
      for c in range(_COLS):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
          cells = []
          for k in range(3):
            nr, nc = r + dr * k, c + dc * k
            if (0 <= nr < _ROWS and 0 <= nc < _COLS
                and self.board[nr, nc] == kitten_val):
              cells.append((nr, nc))
            else:
              break
          if len(cells) == 3:
            to_promote.update(cells)
    if not to_promote:
      return
    n = len(to_promote)
    cats_on_board = int(np.sum(self.board == cat_val))
    for pr, pc in to_promote:
      self.board[pr, pc] = _EMPTY
      self._hand[player][0] += 1  # kitten returns to hand
    cats_to_add = min(n, max(0, _MAX_CATS - cats_on_board - self._hand[player][1]))
    self._hand[player][1] += cats_to_add

  def _check_win(self, player):
    """Return True if player has 3 cats in a row (any of 8 directions)."""
    cat_val = _CAT_VAL[player]
    for r in range(_ROWS):
      for c in range(_COLS):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
          if all(
              0 <= r + dr * k < _ROWS
              and 0 <= c + dc * k < _COLS
              and self.board[r + dr * k, c + dc * k] == cat_val
              for k in range(3)):
            return True
    return False


class BoopObserver:
  """Observer for Boop, conforming to the PyObserver interface."""

  def __init__(self, params):
    if params:
      raise ValueError(f'Observation parameters not supported; passed {params}')
    # 5 planes of 6x6 board (180) + 4 hand scalars = 184 total
    board_size = 5 * _ROWS * _COLS
    self.tensor = np.zeros(board_size + 4, np.float32)
    self.dict = {
        'observation': np.reshape(self.tensor[:board_size], (5, _ROWS, _COLS)),
        'hand': self.tensor[board_size:],
    }

  def set_from(self, state, player):
    """Update tensor and dict to reflect state from player's POV."""
    self.tensor.fill(0)
    obs = self.dict['observation']
    hand = self.dict['hand']
    opp = 1 - player
    mk, mc = _KITTEN_VAL[player], _CAT_VAL[player]
    ok, oc = _KITTEN_VAL[opp], _CAT_VAL[opp]
    for r in range(_ROWS):
      for c in range(_COLS):
        v = state.board[r, c]
        if v == _EMPTY:   obs[0, r, c] = 1.0
        elif v == mk:     obs[1, r, c] = 1.0
        elif v == mc:     obs[2, r, c] = 1.0
        elif v == ok:     obs[3, r, c] = 1.0
        elif v == oc:     obs[4, r, c] = 1.0
    hand[0] = state._hand[player][0] / _MAX_KITTENS
    hand[1] = state._hand[player][1] / _MAX_CATS
    hand[2] = state._hand[opp][0] / _MAX_KITTENS
    hand[3] = state._hand[opp][1] / _MAX_CATS

  def string_from(self, state, player):
    del player
    return str(state)


pyspiel.register_game(_GAME_TYPE, BoopGame)
