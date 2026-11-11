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

"""Breakthrough Env over the JAX engine `breakthrough`.

In OpenSpiel's action ids.

Conventions:
  * actions are OpenSpiel's: ((row * cols + col) * 6 + dir) * 2 + capture,
    rows * cols * 12 of them (768 on 8x8). dir 0-2 move down (row + 1;
    col - 1, col, col + 1): seat 0 (black); dir 3-5 move up (row - 1;
    col - 1, col, col + 1): seat 1 (white). The capture bit is set where the
    target holds an opponent piece.
  * board: row-major, row 0 at the top, -1 empty, 0 / 1 the seat's piece;
    seat 0 starts on rows 0-1 and wins on the last row, seat 1 starts on the
    last two rows and wins on row 0.
  * terminal: actor -1 and an empty legal mask.

A side always has a legal move while it has a piece (its most advanced
piece's off-file diagonal is empty or an opponent), and every move
advances, so the game always ends: each pawn advances at most to the row
before its goal, plus one winning move a side.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import breakthrough
from open_spiel.python.scale.envs import jax_game_env

NUM_DIRS = breakthrough.NUM_DIRECTIONS  # 3 per seat


def action_id(
    row: int, col: int, direction: int, capture: int = 0, columns: int = 8
) -> int:
  """The OpenSpiel action id of moving the piece on (row, col).

  In `direction`.

  Args:
    row: The piece's row.
    col: The piece's column.
    direction: The move direction (0-5).
    capture: 1 for a capture, else 0.
    columns: The board's column count.

  Returns:
    The action id.
  """
  assert 0 <= direction < NUM_DIRS and capture in (0, 1), (direction, capture)
  return breakthrough.action_id(row, col, direction, capture, columns)


class BreakthroughInfo(NamedTuple):
  """What one seat sees: the whole position.

  Attributes:
    seat: i32 the seat.
    board: i32[rows * cols]: -1 empty, 0 / 1 the seat's piece.
    to_move: i32 the seat to move, -1 at terminal.
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[rows * cols]: -1 empty, 0 / 1 the seat's piece
  to_move: jax.Array  # i32: the seat to move, -1 at terminal


class BreakthroughEnv(jax_game_env.JaxGameEnv):
  """Breakthrough: 2 seats, rows * cols * 12 pyspiel actions.

  Attributes:
    rows: The board rows.
    columns: The board columns.
    max_len: The longest game in decisions.
  """

  name = "breakthrough"
  zero_sum = True

  def __init__(self, rows: int = 8, columns: int = 8):
    if not isinstance(rows, int) or rows < 6:
      # pyspiel gives each side one home row below 6 rows, the engine always
      # two: only rows >= 6 are the same game.
      raise ValueError(
          "breakthrough rows must be an int >= 6 (two home rows a side, as "
          f"pyspiel sets up from 6 rows), got {rows!r}"
      )
    if not isinstance(columns, int) or columns < 2:
      raise ValueError(
          f"breakthrough columns must be an int >= 2, got {columns!r}"
      )
    self.rows, self.columns = rows, columns
    self.max_len = 2 * (columns * (rows - 2) + columns * (rows - 3) + 1)
    super().__init__(breakthrough.Breakthrough(rows=rows, columns=columns))
    assert self.max_chance_outcomes == 0
    assert self.num_actions == rows * columns * NUM_DIRS * 2

  def legal_mask(self, s: breakthrough.GameState) -> jax.Array:
    """The actor's legal actions.

    Args:
      s: The state.

    Returns:
      bool[A] mask; all False at terminal.
    """
    return jnp.asarray(
        self.game.legal_actions_mask(s), jnp.bool_
    ) & ~self.terminal(s)

  def info_state(self, s: breakthrough.GameState, seat) -> BreakthroughInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return BreakthroughInfo(
        seat=jnp.asarray(seat, jnp.int32),
        board=jnp.asarray(s.board, jnp.int32),
        to_move=self.actor(s),
    )

  def world_from_info(self, info: BreakthroughInfo) -> breakthrough.GameState:
    """The engine state of the position.

    The engine state (exact before the end; at terminal, the engine's
    current_player is the loser, the seat after the winner's move).

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    board = jnp.asarray(info.board, jnp.int32)
    b2 = board.reshape(self.rows, self.columns)
    pieces = jnp.stack([
        jnp.sum(board == 0, dtype=jnp.int32),
        jnp.sum(board == 1, dtype=jnp.int32),
    ])
    won0 = jnp.any(b2[-1] == 0) | (pieces[1] == 0)
    won1 = jnp.any(b2[0] == 1) | (pieces[0] == 0)
    winner = jnp.where(
        won0, jnp.int32(0), jnp.where(won1, jnp.int32(1), jnp.int32(-1))
    )
    term = jnp.asarray(info.to_move, jnp.int32) == -1
    cp = jnp.where(
        term, jnp.int32(1) - winner, jnp.asarray(info.to_move, jnp.int32)
    )
    legal = jnp.where(
        term,
        jnp.zeros((self.game.num_actions,), jnp.bool_),
        self.game.legal_actions_for(board, cp),
    )
    return breakthrough.GameState(
        board=board,
        current_player=cp,
        is_terminal=jnp.asarray(term, jnp.bool_),
        winner=winner,
        legal_actions_mask=legal,
        pieces=pieces,
    )
