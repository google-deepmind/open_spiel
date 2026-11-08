# Copyright 2026 DeepMind Technologies Limited
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

"""Breakthrough, OpenSpiel's `breakthrough`, on a configurable board.

State: a `GameState` with a row-major board of `rows * columns` cells (-1
empty, 0 black, 1 white) whose row 0 is the top row, the piece counts and a
cached legal-action mask. Black (player 0) starts on rows 0-1, moves down
and moves first; white starts on the last two rows and moves up.

Actions, as in OpenSpiel: ((row * columns + col) * 6 + direction) * 2 +
capture, `rows * columns * 12` ids. Directions 0-2 move down (row + 1) to
col - 1, col, col + 1; directions 3-5 move up (row - 1) to col - 1, col,
col + 1. A piece moves one cell forward, straight onto an empty cell or
diagonally onto an empty cell or an opponent's piece; `capture` is 1 when
the move takes a piece.

Observation: f32[rows, columns, 3], planes (the observer's pieces, the
opponent's pieces, a constant plane equal to the observer's id), in board
orientation for both players. Returns: +1 / -1 for reaching the far row or
taking every opposing piece. The game cannot be drawn.

`egocentric_view(game)` gives the same game with the board turned 180
degrees for white and a compact action space of `rows * columns * 3`
forward moves, so both players see themselves moving down the board.
"""

import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel

NUM_DIRECTIONS = 6  # Three forward directions for each player.


class GameState(NamedTuple):
  """State of a Breakthrough game.

  Attributes:
    board: i32[rows * columns]: -1 empty, 0 black, 1 white.
    current_player: i32: 0 or 1.
    is_terminal: bool.
    legal_actions_mask: bool[rows * columns * 12].
    pieces: i32[2]: pieces left per player.
    winner: i32: -1 (none), 0 or 1.
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  legal_actions_mask: jax.Array
  pieces: jax.Array


def action_id(
    row: int, col: int, direction: int, capture: int, columns: int
) -> int:
  """Returns the OpenSpiel action id of moving the piece on (row, col).

  Args:
    row: The piece's row.
    col: The piece's column.
    direction: The direction index.
    capture: 1 for a capture, else 0.
    columns: The board's columns.

  Returns:
    The action id.
  """
  return ((row * columns + col) * NUM_DIRECTIONS + direction) * 2 + capture


class Breakthrough(game_lib.Game, interop.PyspielInterop):
  """Breakthrough on a `rows` x `columns` board (default 8 x 8).

  Attributes:
    num_cells: The number of cells, rows * columns.
    num_cols: The number of columns.
    num_rows: The number of rows.
  """

  def __init__(self, rows: int = 8, columns: int = 8):
    self.num_rows = rows
    self.num_cols = columns
    self.num_cells = rows * columns

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.num_cells * NUM_DIRECTIONS * 2

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.num_rows, self.num_cols, 3)

  def _decode(self, action):
    """Decodes an action id into its move.

    Args:
      action: The action id.

    Returns:
      (row, col, destination row, destination col, direction, capture).
    """
    capture = action % 2
    rem = action // 2
    direction = rem % NUM_DIRECTIONS
    cell = rem // NUM_DIRECTIONS
    row, col = cell // self.num_cols, cell % self.num_cols
    dr = jnp.where(direction < 3, 1, -1)
    dc = direction % 3 - 1
    return row, col, row + dr, col + dc, direction, capture

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    board = jnp.full((self.num_cells,), -1, dtype=jnp.int32)
    board = board.at[: 2 * self.num_cols].set(0)
    board = board.at[(self.num_rows - 2) * self.num_cols :].set(1)
    pieces = jnp.array([2 * self.num_cols, 2 * self.num_cols], dtype=jnp.int32)
    current_player = jnp.array(0, dtype=jnp.int32)
    return GameState(
        board=board,
        current_player=current_player,
        is_terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        legal_actions_mask=self.legal_actions_for(board, current_player),
        pieces=pieces,
    )

  def legal_actions_for(
      self, board: jax.Array, current_player: jax.Array
  ) -> jax.Array:
    """Returns the legal moves of `current_player`.

    Args:
      board: The board.
      current_player: The player.

    Returns:
      bool[num_actions].
    """
    board_2d = board.reshape(self.num_rows, self.num_cols)
    actions = jnp.arange(self.num_actions, dtype=jnp.int32)
    row, col, tr, tc, direction, capture = self._decode(actions)
    own_direction = jnp.where(
        current_player == 0, direction < 3, direction >= 3
    )
    in_bounds = (
        (tr >= 0) & (tr < self.num_rows) & (tc >= 0) & (tc < self.num_cols)
    )
    target = board_2d[
        jnp.clip(tr, 0, self.num_rows - 1), jnp.clip(tc, 0, self.num_cols - 1)
    ]
    own_piece = board_2d[row, col] == current_player
    diagonal = direction % 3 != 1
    move = (target == -1) & (capture == 0)
    take = diagonal & (target == 1 - current_player) & (capture == 1)
    return own_direction & own_piece & in_bounds & (move | take)

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    row, col, tr, tc, _, _ = self._decode(action)
    player = state.current_player
    opponent = 1 - player
    src_idx = row * self.num_cols + col
    dst_idx = tr * self.num_cols + tc

    board = state.board
    is_capture = board[dst_idx] == opponent
    board = board.at[src_idx].set(-1)
    board = board.at[dst_idx].set(player.astype(jnp.int32))

    pieces = state.pieces
    pieces = jax.lax.select(is_capture, pieces.at[opponent].add(-1), pieces)

    reached_home = jax.lax.select(player == 0, tr == self.num_rows - 1, tr == 0)
    won = reached_home | (pieces[opponent] == 0)
    winner = jax.lax.select(won, player, jnp.int32(-1))
    is_terminal = won | (pieces[player] == 0)

    legal = jax.lax.cond(
        is_terminal,
        lambda _: jnp.zeros((self.num_actions,), dtype=jnp.bool_),
        lambda _: self.legal_actions_for(board, opponent),
        None,
    )
    return GameState(
        board=board,
        current_player=opponent,
        is_terminal=is_terminal,
        winner=winner,
        legal_actions_mask=legal,
        pieces=pieces,
    )

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move; at a terminal state, the player who lost.

    Args:
      state: The state.

    Returns:
      i32[].
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    del state  # Unused.
    return jnp.bool_(False)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    return state.legal_actions_mask

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    board_2d = state.board.reshape(self.num_rows, self.num_cols)
    mine = board_2d == observing_player
    theirs = board_2d == 1 - observing_player
    player_plane = jnp.full(
        (self.num_rows, self.num_cols),
        jnp.asarray(observing_player).astype(jnp.float32),
    )
    return jnp.stack([mine, theirs, player_plane], axis=-1).astype(jnp.float32)

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the observation: the board determines the information state.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      The observation tensor.
    """
    return self.observation_tensor(state, observing_player)

  def returns(self, state: GameState) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    is_win0 = (state.winner == 0) & state.is_terminal
    is_win1 = (state.winner == 1) & state.is_terminal
    p0 = jnp.where(is_win0, 1.0, jnp.where(is_win1, -1.0, 0.0))
    p1 = jnp.where(is_win1, 1.0, jnp.where(is_win0, -1.0, 0.0))
    return jnp.stack([p0, p1])

  def observation_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    # OpenSpiel's planes are (black, white, empty).
    os_obs = np.asarray(os_state.observation_tensor(player)).reshape(
        3, self.num_rows, self.num_cols
    )
    mine, theirs = (
        (os_obs[0], os_obs[1]) if player == 0 else (os_obs[1], os_obs[0])
    )
    player_plane = np.full((self.num_rows, self.num_cols), float(player))
    return jnp.asarray(
        np.stack([mine, theirs, player_plane], axis=-1), dtype=jnp.float32
    )


class EgocentricBreakthrough(game_lib.Game):
  """Breakthrough seen by each player as moving down the board.

  For white the board is turned 180 degrees. Action e ∈ [0, rows * columns *
  3) moves the piece on the player's own (row, col) = divmod(e // 3,
  columns), in that view, one cell forward: down-left, down or down-right
  for e % 3 = 0, 1, 2. Whether the move captures follows from the board.
  The observation is the base game's, turned 180 degrees for white.

  Attributes:
    game: The wrapped Breakthrough game.
  """

  def __init__(self, game: Breakthrough):
    self.game = game
    rows, cols = game.num_rows, game.num_cols
    n = rows * cols
    # [player, compact action] -> OpenSpiel id without the capture bit, and
    # the target cell, clamped onto the board (an off-board move is never
    # legal).
    base = np.zeros((2, 3 * n), np.int32)
    dst = np.zeros((2, 3 * n), np.int32)
    for p in (0, 1):
      for e in range(3 * n):
        ego_dir, rem = e % 3, e // 3
        ego_col, ego_row = rem % cols, rem // cols
        if p == 0:
          r, c, dr, dc = ego_row, ego_col, 1, ego_dir - 1
        else:
          r, c, dr, dc = rows - 1 - ego_row, cols - 1 - ego_col, -1, 1 - ego_dir
        direction = dc + 1 if dr == 1 else 3 + dc + 1
        base[p, e] = action_id(r, c, direction, 0, cols)
        tr = min(max(r + dr, 0), rows - 1)
        tc = min(max(c + dc, 0), cols - 1)
        dst[p, e] = tr * cols + tc
    self._base, self._dst = base, dst

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.game.num_cells * 3

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return self.game.obs_shape

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return self.game.new_initial_state()

  def _os_action(self, state: GameState, action):
    """The OpenSpiel action id of an egocentric action.

    Args:
      state: The state.
      action: The egocentric action.

    Returns:
      i32[], the OpenSpiel action id.
    """
    p = jnp.asarray(state.current_player, jnp.int32)
    capture = state.board[jnp.asarray(self._dst)[p, action]] == 1 - p
    return jnp.asarray(self._base)[p, action] + capture.astype(jnp.int32)

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    return self.game.apply_action(state, self._os_action(state, action))

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return self.game.current_player(state)

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return self.game.is_terminal(state)

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return self.game.is_chance_node(state)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    p = jnp.asarray(state.current_player, jnp.int32)
    by_move = state.legal_actions_mask.reshape(-1, 2).any(axis=-1)
    return by_move[jnp.asarray(self._base)[p] // 2]

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    obs = self.game.observation_tensor(state, observing_player)
    return jax.lax.select(
        observing_player == 1, jnp.flip(obs, axis=(0, 1)), obs
    )

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the observation: the board determines the information state.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      The observation tensor.
    """
    return self.observation_tensor(state, observing_player)

  def returns(self, state: GameState) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    return self.game.returns(state)


def egocentric_view(game: Breakthrough) -> EgocentricBreakthrough:
  """Returns `game` with each player's own view and compact actions.

  Args:
    game: The game.

  Returns:
    The EgocentricBreakthrough.
  """
  return EgocentricBreakthrough(game)
