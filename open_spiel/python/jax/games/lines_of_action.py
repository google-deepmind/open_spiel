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

"""Lines of Action, OpenSpiel's `lines_of_action`.

State: a `GameState` with a row-major board (-1 empty, 0 black, 1 white), a
cached legal-action mask and every board of the game so far. Black (player 0)
starts on the top and bottom rows and moves first; white starts on the side
columns. A piece moves in a straight line exactly as many cells as there are
pieces on that line, may jump its own pieces but not the opponent's, and
captures by landing on an opponent piece. Actions: ((from_row * columns +
from_col) * rows + to_row) * columns + to_col) * 2 + capture, as in OpenSpiel.
Observation: f32[rows, columns, 3], planes (the observer's pieces, the
opponent's pieces, a constant plane equal to the observer's id).

The game ends after a move, in OpenSpiel's order:
  1. a player whose pieces form one 8-connected group wins (the mover, if
     both groups connect at once);
  2. the `max_game_length`-th move is a draw, even if it connects;
  3. a move that recreates a board seen before in the game (the stones only,
     whoever is to move; the initial board included) is a draw, even if it
     connects;
  4. otherwise a player left with no legal move loses.
Returns: +1 / -1 for a win, 0 for a draw.
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


class GameState(NamedTuple):
  """State of a Lines of Action game.

  Attributes:
    board: i32[rows * columns]: -1 empty, 0 black, 1 white.
    board_history: The bitboards of earlier positions, for repetition.
    current_player: 0 or 1.
    is_terminal: bool.
    legal_actions_mask: bool[num_actions].
    move_count: int32, number of moves played.
    winner: -1 (none), 0, 1, 2 (draw).
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  legal_actions_mask: jax.Array
  move_count: jax.Array
  # u32[max_game_length + 1, 2 * words]: row k is the board after k moves as
  # (black, white) bitboards of `words` 32-bit words each; rows after
  # move_count are zero.
  board_history: jax.Array


class LinesOfAction(game_lib.Game, interop.PyspielInterop):
  """Lines of Action on a `rows` x `columns` board.

  Attributes:
    adj_matrix: bool[num_cells, num_cells] 8-connected adjacency.
    bit_weights: The bit of each cell in each bitboard word (0 outside it).
    bitboard_words: The number of 32-bit words of a bitboard.
    col_offsets: The column step of each of the 8 directions.
    line_masks: bool[num_cells, 4, num_cells] the cells on each line type
      through each cell.
    max_game_length: Moves after which the game is a draw.
    num_cells: The number of cells, rows * columns.
    num_cols: The number of columns.
    num_rows: The number of rows.
    row_offsets: The row step of each of the 8 directions.
  """

  def __init__(
      self, rows: int = 8, columns: int = 8, max_game_length: int = 1000
  ):
    if isinstance(max_game_length, bool) or not isinstance(
        max_game_length, int
    ):
      raise TypeError(
          f"max_game_length must be an int, got {max_game_length!r}"
      )
    if max_game_length < 1:
      raise ValueError(f"max_game_length must be >= 1, got {max_game_length}")
    self.num_rows = rows
    self.num_cols = columns
    self.num_cells = rows * columns
    self.max_game_length = max_game_length
    self.bitboard_words = -(-self.num_cells // 32)
    # bit_weights[w, cell]: the bit of `cell` in word w (0 outside it).
    cells = np.arange(self.num_cells)
    weights = np.zeros((self.bitboard_words, self.num_cells), dtype=np.uint32)
    weights[cells // 32, cells] = np.left_shift(
        np.uint32(1), (cells % 32).astype(np.uint32)
    )
    self.bit_weights = weights

    self.row_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    self.col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

    self.line_masks = self._precompute_line_masks()
    self.adj_matrix = self._precompute_adj_matrix()

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.num_rows * self.num_cols * self.num_rows * self.num_cols * 2

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.num_rows, self.num_cols, 3)

  def _precompute_line_masks(self):
    """For each cell and each of the 4 line types, the cells on that line.

    Returns:
      bool[num_cells, 4, num_cells], the line masks.
    """
    masks = np.zeros((self.num_cells, 4, self.num_cells), dtype=bool)

    for r in range(self.num_rows):
      for c in range(self.num_cols):
        cell = r * self.num_cols + c

        # Line 0: Vertical (same column)
        for rr in range(self.num_rows):
          masks[cell, 0, rr * self.num_cols + c] = True

        # Line 1: Diagonal / (r - c = const)
        for rr in range(self.num_rows):
          cc = rr - r + c
          if 0 <= cc < self.num_cols:
            masks[cell, 1, rr * self.num_cols + cc] = True

        # Line 2: Horizontal (same row)
        for cc in range(self.num_cols):
          masks[cell, 2, r * self.num_cols + cc] = True

        # Line 3: Diagonal \ (r + c = const)
        for rr in range(self.num_rows):
          cc = r + c - rr
          if 0 <= cc < self.num_cols:
            masks[cell, 3, rr * self.num_cols + cc] = True

    return jnp.array(masks)

  def _precompute_adj_matrix(self):
    """Adjacency matrix for flood-fill connectivity check (8-connected).

    Returns:
      bool[num_cells, num_cells], the adjacency matrix.
    """
    adj = np.zeros((self.num_cells, self.num_cells), dtype=bool)
    for r in range(self.num_rows):
      for c in range(self.num_cols):
        idx = r * self.num_cols + c
        for d in range(8):
          nr = r + self.row_offsets[d]
          nc = c + self.col_offsets[d]
          if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
            adj[idx, nr * self.num_cols + nc] = True
    return jnp.array(adj)

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    board = jnp.full((self.num_cells,), -1, dtype=jnp.int32)

    # Black (player 0) on rows 0 and 7, columns 1-6.
    for c in range(1, self.num_cols - 1):
      board = board.at[0 * self.num_cols + c].set(0)
      board = board.at[(self.num_rows - 1) * self.num_cols + c].set(0)

    # White (player 1) on columns 0 and 7, rows 1-6.
    for r in range(1, self.num_rows - 1):
      board = board.at[r * self.num_cols + 0].set(1)
      board = board.at[r * self.num_cols + (self.num_cols - 1)].set(1)

    current_player = jnp.array(0, dtype=jnp.int32)
    mask = self._compute_legal_actions(board, current_player)
    move_count = jnp.array(0, dtype=jnp.int32)

    return GameState(
        board=board,
        current_player=current_player,
        is_terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        legal_actions_mask=mask,
        move_count=move_count,
        board_history=self.board_history_from(board, move_count),
    )

  def bitboards(self, board: jax.Array) -> jax.Array:
    """Returns the (black, white) bitboards of `board`, exact.

    Args:
      board: i32[num_cells], the board.

    Returns:
      u32[2 * words].
    """
    weights = jnp.asarray(self.bit_weights)
    black = jnp.sum(
        jnp.where(board[None, :] == 0, weights, jnp.uint32(0)),
        axis=1,
        dtype=jnp.uint32,
    )
    white = jnp.sum(
        jnp.where(board[None, :] == 1, weights, jnp.uint32(0)),
        axis=1,
        dtype=jnp.uint32,
    )
    return jnp.concatenate([black, white])

  def board_history_from(
      self, board: jax.Array, move_count: jax.Array
  ) -> jax.Array:
    """Returns a `board_history` that holds `board` alone.

    `board` is the board after `move_count` moves: the history of a position
    whose earlier boards are unknown.

    Args:
      board: i32[num_cells], the board.
      move_count: The number of moves played.

    Returns:
      u32[max_game_length + 1, 2 * words].
    """
    history = jnp.zeros(
        (self.max_game_length + 1, 2 * self.bitboard_words), dtype=jnp.uint32
    )
    return history.at[jnp.asarray(move_count, jnp.int32)].set(
        self.bitboards(board), mode="drop"
    )

  def _line_type_from_dr_dc(self, dr: jax.Array, dc: jax.Array) -> jax.Array:
    """Maps direction (dr, dc) to line type 0-3.

    Args:
      dr: The row step.
      dc: The column step.

    Returns:
      i32[], the line type.
    """
    return jnp.where(
        dc == 0,
        0,  # Vertical
        jnp.where(
            dr == 0,
            2,  # Horizontal
            jnp.where(dr == dc, 1, 3),  # Diagonal / or \ .
        ),
    )

  def _compute_legal_actions(
      self, board: jax.Array, current_player: jax.Array
  ) -> jax.Array:
    """Computes legal actions mask for the current player.

    Args:
      board: The board.
      current_player: The player to move.

    Returns:
      bool[num_actions], the legal-action mask.
    """
    occupied = board != -1  # (num_cells,) bool
    opponent = 1 - current_player

    def is_legal_action(action_idx):
      """Returns whether one encoded move is legal.

      Args:
        action_idx: The action.

      Returns:
        bool[].
      """
      # Decode mixed base {rows, cols, rows, cols, 2}
      capture = action_idx % 2
      rem = action_idx // 2
      to_c = rem % self.num_cols
      rem = rem // self.num_cols
      to_r = rem % self.num_rows
      rem = rem // self.num_rows
      from_c = rem % self.num_cols
      from_r = rem // self.num_cols

      # 1. Source cell must contain current player's piece.
      from_cell = from_r * self.num_cols + from_c
      valid_piece = board[from_cell] == current_player

      # 2. Direction must be a valid straight line.
      row_diff = to_r - from_r
      col_diff = to_c - from_c
      abs_row = jnp.abs(row_diff)
      abs_col = jnp.abs(col_diff)
      dist = jnp.maximum(abs_row, abs_col)
      valid_direction = (dist > 0) & (
          (abs_row == 0) | (abs_col == 0) | (abs_row == abs_col)
      )

      # 3. Distance must equal piece count on this line.
      dr = jnp.sign(row_diff)
      dc = jnp.sign(col_diff)
      line = self._line_type_from_dr_dc(dr, dc)
      line_mask = self.line_masks[from_cell, line]
      pieces_on_line = jnp.sum(occupied & line_mask)
      valid_dist = dist == pieces_on_line

      # 4. Path must be clear of opponent pieces (intermediate cells only).
      def check_step(clear, k):
        """Checks one intermediate cell of the move's path.

        Args:
          clear: Whether the path is clear so far.
          k: The step index.

        Returns:
          (clear, None) for `jax.lax.scan`.
        """
        step = k + 1
        nr = from_r + step * dr
        nc = from_c + step * dc
        is_intermediate = step < dist
        in_bounds = (
            (nr >= 0) & (nr < self.num_rows) & (nc >= 0) & (nc < self.num_cols)
        )
        clamped_idx = jnp.clip(
            nr, 0, self.num_rows - 1
        ) * self.num_cols + jnp.clip(nc, 0, self.num_cols - 1)
        cell_val = board[clamped_idx]
        blocked = is_intermediate & in_bounds & (cell_val == opponent)
        return clear & ~blocked, None

      path_clear, _ = jax.lax.scan(check_step, jnp.bool_(True), jnp.arange(7))

      # 5. Destination cell must not contain own piece.
      to_cell = to_r * self.num_cols + to_c
      dest_val = board[to_cell]
      not_own = dest_val != current_player

      # 6. Capture flag must match destination contents.
      capture_match = jnp.where(
          capture == 1, dest_val == opponent, dest_val == -1
      )

      return (
          valid_piece
          & valid_direction
          & valid_dist
          & path_clear
          & not_own
          & capture_match
      )

    action_indices = jnp.arange(self.num_actions)
    return jax.vmap(is_legal_action)(action_indices)

  def legal_actions_for(self, board: jax.Array, player: jax.Array) -> jax.Array:
    """Returns the legal moves of `player` on `board`.

    Args:
      board: i32[num_cells], the board.
      player: The player.

    Returns:
      bool[num_actions].
    """
    return self._compute_legal_actions(board, player)

  def is_connected(self, board: jax.Array, player: jax.Array) -> jax.Array:
    """Returns whether all of `player`'s pieces form one 8-connected group.

    True with no pieces.

    Args:
      board: i32[num_cells], the board.
      player: The player.

    Returns:
      bool[].
    """
    return self._is_connected(board, player)

  def _is_connected(self, board: jax.Array, player: jax.Array) -> jax.Array:
    """Checks if all pieces of `player` form a single connected group.

    Args:
      board: The board.
      player: The player.

    Returns:
      bool[], whether the pieces are connected.
    """
    pieces = board == player  # (num_cells,)
    num_pieces = jnp.sum(pieces)

    seed_idx = jnp.argmax(pieces)
    reached = jnp.zeros(self.num_cells, dtype=jnp.bool_).at[seed_idx].set(True)
    reached = reached & pieces
    adj = self.adj_matrix.astype(jnp.float32)

    # Grow the group from the seed until it stops growing.
    def grow(carry):
      """Grows the reached set by one neighbour step.

      Args:
        carry: (reached, changed).

      Returns:
        (grown, whether it grew).
      """
      reached, _ = carry
      neighbors = jnp.dot(reached.astype(jnp.float32), adj) > 0
      grown = (reached | neighbors) & pieces
      return grown, jnp.any(grown != reached)

    reached, _ = jax.lax.while_loop(
        lambda carry: carry[1], grow, (reached, jnp.bool_(True))
    )

    num_reached = jnp.sum(reached)
    return (num_reached == num_pieces) | (num_pieces == 0)

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
    action = jnp.asarray(action, jnp.int32)
    rem = action // 2
    to_c = rem % self.num_cols
    rem = rem // self.num_cols
    to_r = rem % self.num_rows
    rem = rem // self.num_rows
    from_c = rem % self.num_cols
    from_r = rem // self.num_cols

    player = state.current_player
    opponent = 1 - player

    from_idx = from_r * self.num_cols + from_c
    to_idx = to_r * self.num_cols + to_c

    board = state.board.at[from_idx].set(-1)
    board = board.at[to_idx].set(player.astype(jnp.int32))

    move_count = state.move_count + 1

    # 1. A connected group wins; the mover first.
    current_connected = self._is_connected(board, player)
    opponent_connected = self._is_connected(board, opponent)
    winner = jnp.where(
        current_connected,
        player,
        jnp.where(opponent_connected, opponent, -1),
    ).astype(jnp.int32)

    # 2. The move limit, then 3. a repeated board, draw over a win.
    bits = self.bitboards(board)
    earlier = jnp.arange(self.max_game_length + 1, dtype=jnp.int32) < move_count
    repeated = jnp.any(
        earlier & jnp.all(state.board_history == bits[None, :], axis=1)
    )
    is_draw = (move_count >= self.max_game_length) | repeated
    winner = jnp.where(is_draw, jnp.int32(2), winner)
    is_terminal = winner != -1

    next_player = opponent

    def compute_next_legal(_):
      """Returns the next player's legal moves.

      Args:
        _: Unused.

      Returns:
        bool[num_actions].
      """
      return self._compute_legal_actions(board, next_player)

    def zero_legal(_):
      """Returns no legal moves (the game is over).

      Args:
        _: Unused.

      Returns:
        bool[num_actions], all False.
      """
      return jnp.zeros((self.num_actions,), dtype=jnp.bool_)

    new_legal = jax.lax.cond(is_terminal, zero_legal, compute_next_legal, None)

    # 4. A player left without a legal move loses, so the mover wins.
    no_moves = (~is_terminal) & (~jnp.any(new_legal))
    winner = jnp.where(no_moves, player, winner).astype(jnp.int32)
    is_terminal = is_terminal | no_moves

    return GameState(
        board=board,
        current_player=next_player,
        is_terminal=is_terminal,
        winner=winner,
        legal_actions_mask=new_legal,
        move_count=move_count,
        board_history=state.board_history.at[move_count].set(bits, mode="drop"),
    )

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
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

    my_pieces = board_2d == observing_player
    opp_pieces = board_2d == (1 - observing_player)
    player_plane = jnp.full(
        (self.num_rows, self.num_cols),
        jnp.asarray(observing_player).astype(jnp.float32),
    )

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
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
    os_obs = np.asarray(os_state.observation_tensor(player)).reshape(
        3, self.num_rows, self.num_cols
    )

    black_pieces = os_obs[0]
    white_pieces = os_obs[1]

    is_player_0 = player == 0

    my_pieces = jnp.where(is_player_0, black_pieces, white_pieces)
    opp_pieces = jnp.where(is_player_0, white_pieces, black_pieces)

    player_plane = jnp.full((self.num_rows, self.num_cols), float(player))

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )
