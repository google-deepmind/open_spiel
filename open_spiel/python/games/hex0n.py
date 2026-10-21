# Copyright 2026 Georgios Tritsaris
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

"""HEX0N, implemented in Python.

A two-player placement game on an N x N board (5 x 5 by default). "Touching"
always means the 8 surrounding cells, diagonals included. On a turn a player
places one piece:

  square    any empty cell; each player has round(N*N / 3) of them
  circle    an empty cell touching a square of either colour
  triangle  an empty cell touching a circle and a square, at least one of the
            two belonging to the mover (both, with opponent_pieces=False)
  pass      only when the player has no other legal placement

Placing a triangle removes it together with the circle and the square, and the
same player then places a hex on one of the three cleared cells. Hexes are
permanent and score one point each. If the triangle touches several eligible
circle+square pairs, one is removed at random: this is an explicit chance node
whose outcomes are the pairs, each equally likely.

The game ends when neither player can place a piece. Most hexes wins; on equal
hexes the larger connected group of hexes wins (tie_break=True), otherwise the
game is a draw.

Actions: kind * N*N + cell for kind in (square, circle, triangle, hex), with
cell = row * N + col; N*N * 4 is pass. At a chance node, outcome k removes the
k-th eligible pair, in the order chance_outcomes() lists them.
"""

import functools

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

SQUARE, CIRCLE, TRIANGLE, HEX = range(4)
_KIND_NAMES = ("square", "circle", "triangle", "hex")
_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"size": 5, "tie_break": True, "opponent_pieces": True}
# A triangle touches at most 8 cells, so at most 4 circles x 4 squares.
_MAX_CHANCE_OUTCOMES = 16

_GAME_TYPE = pyspiel.GameType(
    short_name="python_hex0n",
    long_name="Python HEX0N",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS)


@functools.lru_cache(maxsize=None)
def _neighbors(size):
  """For each cell, the cells touching it (diagonals included), ascending."""
  result = []
  for cell in range(size * size):
    r, c = divmod(cell, size)
    result.append(tuple(
        rr * size + cc
        for rr in (r - 1, r, r + 1)
        for cc in (c - 1, c, c + 1)
        if (rr, cc) != (r, c) and 0 <= rr < size and 0 <= cc < size))
  return tuple(result)


def _squares_per_player(size):
  return round(size * size / 3)  # 8 on 5x5, 16 on 7x7


def _max_game_length(size):
  """Upper bound on the number of actions in a game, chance actions included.

  Every triad leaves a permanent hex, so there are at most N*N triads, each a
  triangle, a chance outcome and a hex, and each removing one circle. Squares
  run out, and a pass is always followed by a placement.
  """
  cells = size * size
  moves = 2 * _squares_per_player(size) + 2 * cells + 3 * cells
  return 2 * moves + 1


class Hex0nGame(pyspiel.Game):
  """A Python version of HEX0N."""

  def __init__(self, params=None):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    size = params["size"]
    if size < 2:
      raise ValueError(f"size must be at least 2, got {size}")
    game_info = pyspiel.GameInfo(
        num_distinct_actions=4 * size * size + 1,
        max_chance_outcomes=_MAX_CHANCE_OUTCOMES,
        num_players=_NUM_PLAYERS,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=_max_game_length(size))
    super().__init__(_GAME_TYPE, game_info, params)
    self.size = size
    self.tie_break = params["tie_break"]
    self.opponent_pieces = params["opponent_pieces"]

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return Hex0nState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return Hex0nObserver(self.size, params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class Hex0nState(pyspiel.State):
  """A Python version of the HEX0N state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.size = game.size
    self.num_cells = game.size * game.size
    self.tie_break = game.tie_break
    self.opponent_pieces = game.opponent_pieces
    self.squares_per_player = _squares_per_player(game.size)
    self.pass_action = 4 * self.num_cells
    self.board = [None] * self.num_cells  # each cell: None or (kind, owner)
    self.player = 0  # whose turn it is; player 0 starts
    self.squares_left = [self.squares_per_player] * _NUM_PLAYERS
    self.hexes = [0] * _NUM_PLAYERS
    self.pairs = []  # (circle, square) pairs waiting for the chance outcome
    self.triangle = None  # cell of the triangle waiting for that outcome
    self.hex_options = None  # the cleared cells while a hex must be placed
    self.over = False

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.over:
      return pyspiel.PlayerId.TERMINAL
    if self.pairs:
      return pyspiel.PlayerId.CHANCE
    return self.player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    if self.hex_options:
      return [HEX * self.num_cells + c for c in self.hex_options]
    return self._placements(player) or [self.pass_action]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.pairs
    p = 1.0 / len(self.pairs)
    return [(k, p) for k in range(len(self.pairs))]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.pairs:  # chance picked which pair goes
      self._remove_triad(self.pairs[action])
      return
    if action == self.pass_action:
      self._end_turn()
      return
    kind, cell = divmod(action, self.num_cells)
    p = self.player
    self.board[cell] = (kind, p)
    if kind == SQUARE:
      self.squares_left[p] -= 1
    elif kind == HEX:
      self.hexes[p] += 1
      self.hex_options = None
    elif kind == TRIANGLE:
      pairs = self.triad_pairs(cell, p)
      self.triangle = cell
      if len(pairs) == 1:
        self._remove_triad(pairs[0])
      else:
        self.pairs = pairs  # wait for the chance node
      return  # the same player continues with the hex
    self._end_turn()

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      circle, square = self.pairs[action]
      return f"remove circle {self._rc(circle)} + square {self._rc(square)}"
    if action == self.pass_action:
      return "pass"
    kind, cell = divmod(action, self.num_cells)
    return f"{_KIND_NAMES[kind]} {self._rc(cell)}"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self.over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    w = self.winner()
    return [0.0, 0.0] if w is None else [1.0 - 2 * w, 2 * w - 1.0]

  def __str__(self):
    """Player 0 in capitals (S C T H), player 1 in lower case, '.' empty."""
    rows = []
    for r in range(self.size):
      row = self.board[r * self.size:(r + 1) * self.size]
      rows.append(" ".join(
          "." if p is None else ("SCTH" if p[1] == 0 else "scth")[p[0]]
          for p in row))
    if self.over:
      who = "game over"
    elif self.pairs:
      who = "chance"
    else:
      who = f"player {self.player} to move"
    rows.append(f"hexes {self.hexes}  squares left {self.squares_left}  {who}")
    return "\n".join(rows)

  # Game rules.

  def triad_pairs(self, cell, player):
    """(circle, square) pairs touching `cell` usable by `player`'s triangle."""
    near = [(n, self.board[n]) for n in _neighbors(self.size)[cell]
            if self.board[n] is not None]
    circles = [(n, p[1]) for n, p in near if p[0] == CIRCLE]
    squares = [(n, p[1]) for n, p in near if p[0] == SQUARE]
    # How many of the two must be the mover's: one, or both.
    mine = any if self.opponent_pieces else all
    return [(c, s) for c, c_owner in circles for s, s_owner in squares
            if mine((c_owner == player, s_owner == player))]

  def _touches_square(self, cell):
    return any(self.board[n] is not None and self.board[n][0] == SQUARE
               for n in _neighbors(self.size)[cell])

  def _placements(self, player):
    """Every square, circle and triangle `player` could place now."""
    n, actions = self.num_cells, []
    for cell in range(n):
      if self.board[cell] is not None:
        continue
      if self.squares_left[player] > 0:
        actions.append(SQUARE * n + cell)
      if self._touches_square(cell):
        actions.append(CIRCLE * n + cell)
      if self.triad_pairs(cell, player):
        actions.append(TRIANGLE * n + cell)
    return sorted(actions)

  def _can_place(self, player):
    """True if `player` has at least one square, circle or triangle to place."""
    for cell in range(self.num_cells):
      if self.board[cell] is not None:
        continue
      if (self.squares_left[player] > 0 or self._touches_square(cell) or
          self.triad_pairs(cell, player)):
        return True
    return False

  def _remove_triad(self, pair):
    cleared = sorted([self.triangle, *pair])
    for cell in cleared:
      self.board[cell] = None
    self.hex_options = cleared
    self.pairs, self.triangle = [], None

  def _end_turn(self):
    self.player = 1 - self.player
    if not self._can_place(0) and not self._can_place(1):
      self.over = True

  def largest_group(self, player):
    """Size of `player`'s largest group of touching hexes."""
    seen, best = set(), 0
    neighbors = _neighbors(self.size)
    for start in range(self.num_cells):
      if self.board[start] != (HEX, player) or start in seen:
        continue
      stack, size = [start], 0
      seen.add(start)
      while stack:
        cell = stack.pop()
        size += 1
        for n in neighbors[cell]:
          if n not in seen and self.board[n] == (HEX, player):
            seen.add(n)
            stack.append(n)
      best = max(best, size)
    return best

  def winner(self):
    """0 or 1 once the game is over; None for a draw or a game in progress."""
    if not self.over:
      return None
    a, b = self.hexes
    if a == b and self.tie_break:
      a, b = self.largest_group(0), self.largest_group(1)
    return None if a == b else (0 if a > b else 1)

  def _rc(self, cell):
    return "(%d,%d)" % divmod(cell, self.size)


class Hex0nObserver:
  """Observer, conforming to the PyObserver interface (see observation.py).

  The tensor has shape (11, N, N), from the observing player's point of view:
    0-3   the player's squares, circles, triangles, hexes
    4-7   the opponent's squares, circles, triangles, hexes
    8     empty cells
    9     cells where a hex may be placed now
    10    the player's squares left minus the opponent's, scaled to [-1, 1]
  """

  def __init__(self, size, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    shape = (11, size, size)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    obs = self.dict["observation"]
    obs.fill(0)
    planes = obs.reshape(11, -1)  # one row per plane, one column per cell
    for cell, piece in enumerate(state.board):
      if piece is None:
        planes[8, cell] = 1
      else:
        kind, owner = piece
        planes[kind + (0 if owner == player else 4), cell] = 1
    for cell in state.hex_options or []:
      planes[9, cell] = 1
    diff = state.squares_left[player] - state.squares_left[1 - player]
    planes[10] = diff / state.squares_per_player

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return str(state)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, Hex0nGame)
