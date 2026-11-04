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

"""Othello (Reversi), OpenSpiel's `othello`, on an 8x8 board.

State: a `GameState` with an i32[8, 8] board (0 empty, 1 black, 2 white),
the player to move and a consecutive-pass counter. Black (player 0) moves
first from the standard centre position.

Actions: 0..63 place a disc at (action // 8, action % 8); 64 passes, which is
legal only when the player has no placement. A placement must flip at least
one opponent disc; every line of opponent discs between the new disc and an
existing disc of the mover's colour, in all 8 directions, flips.

Observation: f32[8, 8, 3], planes (the observer's discs, the opponent's
discs, a constant plane equal to the observer's id).

Returns: the game ends when neither player can place (two consecutive
passes); the player with more discs gets +1 and the other -1, 0 for a tie.
"""

import typing
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel

ROWS = 8
COLS = 8
NUM_CELLS = ROWS * COLS
ACTION_PASS = NUM_CELLS  # 64
NUM_ACTIONS = NUM_CELLS + 1  # 65

# 8 directions: N, NE, E, SE, S, SW, W, NW.
_DIRECTIONS = np.array(
    [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
    ],
    dtype=np.int32,
)


class GameState(NamedTuple):
  """State of an Othello game.

  Attributes:
    board: i32[8, 8]: 0 empty, 1 black, 2 white.
    current_player: 0 (Black) or 1 (White).
    is_terminal: bool.
    pass_count: Consecutive-pass counter; 2 ends the game.
    winner: -1 (draw/none), 0 (Black wins), 1 (White wins).
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  pass_count: jax.Array


STATE_DTYPES = {
    "board": jnp.int32,
    "current_player": jnp.int32,
    "is_terminal": jnp.bool_,
    "winner": jnp.int32,
    "pass_count": jnp.int32,
}


def assert_state_dtypes(state: GameState) -> None:
  """Raises if a field of `state` has a dtype other than STATE_DTYPES.

  A check for tests.

  Args:
    state: The state.
  """
  chex.assert_type(state.board, STATE_DTYPES["board"])
  chex.assert_type(state.current_player, STATE_DTYPES["current_player"])
  chex.assert_type(state.is_terminal, STATE_DTYPES["is_terminal"])
  chex.assert_type(state.winner, STATE_DTYPES["winner"])
  chex.assert_type(state.pass_count, STATE_DTYPES["pass_count"])


def _in_bounds(row: jax.Array, col: jax.Array) -> jax.Array:
  """Whether (row, col) is on the board.

  Args:
    row: The row.
    col: The column.

  Returns:
    bool[], True on the board.
  """
  return (row >= 0) & (row < ROWS) & (col >= 0) & (col < COLS)


def _scan_direction(board, row, col, direction, player_mark, opp_mark):
  """Walk one direction from (row, col).

  Args:
    board: The board.
    row: The start row.
    col: The start column.
    direction: (row step, column step).
    player_mark: The mover's mark.
    opp_mark: The opponent's mark.

  Returns:
    (flip_mask, valid): flip_mask marks opponent cells that would flip IF
    this line is valid; the scalar `valid` is True iff the line terminates at
    a same-color disc with at least one opponent disc between.
  """
  dr, dc = direction[0], direction[1]
  init_mask = jnp.zeros((ROWS, COLS), dtype=jnp.bool_)

  def scan_fn(carry, xs):
    """Advances one cell along the direction.

    Args:
      carry: (row, col, mask, found opponent, done, valid).
      xs: The scan's xs, None (unused).

    Returns:
      (carry, None) for `jax.lax.scan`.
    """
    del xs
    r, c, mask, found_opp, done, valid = carry
    new_r = r + dr
    new_c = c + dc
    in_bounds = _in_bounds(new_r, new_c)
    cell = jnp.where(in_bounds, board[new_r % ROWS, new_c % COLS], jnp.int32(0))
    is_opp = cell == opp_mark
    is_own = cell == player_mark
    is_empty = cell == 0

    new_mask = jnp.where(
        ~done & is_opp & in_bounds,
        mask.at[new_r % ROWS, new_c % COLS].set(True),
        mask,
    )
    new_found_opp = jnp.where(~done, found_opp | is_opp, found_opp)
    new_valid = valid | (is_own & found_opp & ~done)
    new_done = done | ~in_bounds | is_empty | is_own
    return (new_r, new_c, new_mask, new_found_opp, new_done, new_valid), None

  init_carry = (
      row,
      col,
      init_mask,
      jnp.bool_(False),
      jnp.bool_(False),
      jnp.bool_(False),
  )
  (_, _, final_mask, _, _, valid), _ = jax.lax.scan(
      scan_fn, init_carry, None, length=ROWS - 1
  )
  return final_mask, valid


def _all_flips(board, row, col, player_mark, opp_mark, directions):
  """Mask of opponent discs flipped by placing at (row, col).

  Args:
    board: The board.
    row: The row placed on.
    col: The column placed on.
    player_mark: The mover's mark.
    opp_mark: The opponent's mark.
    directions: The directions.

  Returns:
    bool[ROWS, COLS], the flipped discs.
  """

  def per_dir(direction):
    """Returns the discs flipped in one direction.

    Args:
      direction: The direction index.

    Returns:
      bool[ROWS, COLS].
    """
    mask, valid = _scan_direction(
        board, row, col, direction, player_mark, opp_mark
    )
    return jnp.where(valid, mask, jnp.zeros((ROWS, COLS), dtype=jnp.bool_))

  all_masks = jax.vmap(per_dir)(directions)
  return jnp.any(all_masks, axis=0)


def _is_valid_placement(board, row, col, player_mark, opp_mark, directions):
  """Whether placing at (row, col) is legal.

  The cell must be empty and the placement must flip a disc in some direction.

  Args:
    board: The board.
    row: The row.
    col: The column.
    player_mark: The mover's mark.
    opp_mark: The opponent's mark.
    directions: The direction indices.

  Returns:
    bool[].
  """
  is_empty = board[row, col] == 0

  def per_dir(direction):
    """Returns whether the placement flips in one direction.

    Args:
      direction: The direction index.

    Returns:
      bool[].
    """
    _, valid = _scan_direction(
        board, row, col, direction, player_mark, opp_mark
    )
    return valid

  any_valid = jnp.any(jax.vmap(per_dir)(directions))
  return is_empty & any_valid


def _board_placement_mask(board, player_mark, opp_mark, directions):
  """Boolean mask over 64 board cells: True where placement is legal.

  Args:
    board: The board.
    player_mark: The mover's mark.
    opp_mark: The opponent's mark.
    directions: The directions.

  Returns:
    bool[64], the legal placements.
  """

  def check_pos(idx):
    """Returns whether placing on cell `idx` is valid.

    Args:
      idx: The cell.

    Returns:
      bool[].
    """
    row = idx // COLS
    col = idx % COLS
    return _is_valid_placement(
        board, row, col, player_mark, opp_mark, directions
    )

  return jax.vmap(check_pos)(jnp.arange(NUM_CELLS))


class Othello(game_lib.Game, interop.PyspielInterop):
  """Othello / Reversi.

  Attributes:
    num_cells: The number of cells, rows * columns.
    num_cols: The number of columns.
    num_rows: The number of rows.
  """

  def __init__(self):
    self.num_rows = ROWS
    self.num_cols = COLS
    self.num_cells = NUM_CELLS

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return NUM_ACTIONS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (ROWS, COLS, 3)

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  def _directions(self) -> jax.Array:
    """The eight directions as an array.

    Returns:
      i32[8, 2], the directions.
    """
    return jnp.asarray(_DIRECTIONS, dtype=jnp.int32)

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    board = jnp.zeros((ROWS, COLS), dtype=jnp.int32)
    # Standard starting position (center 2x2): white on the a1-h8 diagonal,
    # black on the other diagonal.
    board = board.at[3, 3].set(2)
    board = board.at[3, 4].set(1)
    board = board.at[4, 3].set(1)
    board = board.at[4, 4].set(2)
    return GameState(
        board=board,
        current_player=jnp.array(0, dtype=jnp.int32),
        is_terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        pass_count=jnp.array(0, dtype=jnp.int32),
    )

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
    board = state.board
    player = state.current_player
    directions = self._directions()

    player_mark = player + 1  # 1 for Black, 2 for White
    opp_mark = 2 - player  # opponent's mark (1 or 2)

    is_pass = action == ACTION_PASS
    row = action // COLS
    col = action % COLS

    # Clamp indices so out-of-bounds gathers on the pass branch don't fault.
    safe_row = jnp.clip(row, 0, ROWS - 1)
    safe_col = jnp.clip(col, 0, COLS - 1)
    flip_mask = _all_flips(
        board, safe_row, safe_col, player_mark, opp_mark, directions
    )

    placed = board.at[safe_row, safe_col].set(player_mark)
    placed = jnp.where(flip_mask, jnp.full_like(placed, player_mark), placed)
    new_board = jnp.where(is_pass | state.is_terminal, board, placed)

    new_pass_count = jnp.where(
        is_pass & ~state.is_terminal,
        state.pass_count + 1,
        jnp.int32(0),
    )

    next_player = jnp.int32(1) - player
    next_mark = next_player + 1
    next_opp_mark = 2 - next_player

    next_has_moves = jnp.any(
        _board_placement_mask(new_board, next_mark, next_opp_mark, directions)
    )
    curr_has_moves = jnp.any(
        _board_placement_mask(new_board, player_mark, opp_mark, directions)
    )

    neither_can_move = ~next_has_moves & ~curr_has_moves
    double_passed = new_pass_count >= 2
    new_is_terminal = state.is_terminal | neither_can_move | double_passed

    black_count = jnp.sum(new_board == 1)
    white_count = jnp.sum(new_board == 2)
    new_winner = jnp.where(
        new_is_terminal & (black_count > white_count),
        jnp.int32(0),
        jnp.where(
            new_is_terminal & (white_count > black_count),
            jnp.int32(1),
            jnp.int32(-1),
        ),
    )

    new_current_player = jnp.where(new_is_terminal, player, next_player)

    return GameState(
        board=new_board,
        current_player=new_current_player,
        is_terminal=new_is_terminal,
        winner=new_winner,
        pass_count=new_pass_count,
    )

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    directions = self._directions()
    player_mark = state.current_player + 1
    opp_mark = 2 - state.current_player
    board_mask = _board_placement_mask(
        state.board, player_mark, opp_mark, directions
    )
    pass_legal = ~jnp.any(board_mask)
    full_mask = jnp.concatenate([board_mask, pass_legal[jnp.newaxis]], axis=0)
    return jnp.where(
        state.is_terminal, jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), full_mask
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
    my_mark = observing_player + 1
    opp_mark = 2 - observing_player
    my_pieces = state.board == my_mark
    opp_pieces = state.board == opp_mark
    player_plane = jnp.full(
        (ROWS, COLS), jnp.asarray(observing_player, dtype=jnp.float32)
    )
    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )

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
    # OpenSpiel's planes are (empty, the observer's discs, the opponent's).
    os_obs = np.asarray(os_state.observation_tensor(player), dtype=np.float32)
    os_obs = os_obs.reshape(3, ROWS, COLS)
    my_pieces = os_obs[1]
    opp_pieces = os_obs[2]

    player_plane = jnp.full((ROWS, COLS), float(player))
    return jnp.stack(
        [jnp.asarray(my_pieces), jnp.asarray(opp_pieces), player_plane],
        axis=-1,
    ).astype(jnp.float32)
