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

"""Gomoku Env over the JAX engine `gomoku` (perfect information).

Board: size * size cells, row-major (pyspiel's action ids: row * size +
col), -1 empty, 0 black (seat 0, moves first), 1 white. `connect` in a row
(or more: an overline wins, as in pyspiel) on a row, column or diagonal
wins; a full board is a draw. Terminal: actor -1, empty legal mask
(the engine flips current_player past the end).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gomoku
from open_spiel.python.scale.envs import jax_game_env


class GomokuInfo(NamedTuple):
  """What one seat sees: the whole board.

  Attributes:
    seat: i32 the seat.
    board: i32[size * size]: -1 empty, 0 / 1 the seat's stone.
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[size * size]: -1 empty, 0 / 1 the seat's stone


class GomokuEnv(jax_game_env.JaxGameEnv):
  """Gomoku: 2 seats, size * size actions (cells).

  Attributes:
    board_size: The board side.
    connect: The stones in a row that win.
    max_len: The longest game in decisions (board_size ** 2).
  """

  name = "gomoku"
  zero_sum = True

  def __init__(self, board_size: int = 15, connect: int = 5):
    if not isinstance(board_size, int) or board_size < 1:
      raise ValueError(
          f"gomoku board_size must be a positive int, got {board_size!r}"
      )
    if not isinstance(connect, int) or not 2 <= connect <= board_size:
      raise ValueError(
          f"gomoku connect must be an int in [2, board_size={board_size}], got "
          f"{connect!r}"
      )
    self.board_size, self.connect = board_size, connect
    self.max_len = board_size * board_size
    super().__init__(gomoku.Gomoku(board_size=board_size, connect=connect))
    assert self.max_chance_outcomes == 0 and self.num_actions == self.max_len
    self._win_lines = np.asarray(self.game.win_lines, np.int32)

  def info_state(self, s: gomoku.GameState, seat) -> GomokuInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return GomokuInfo(
        seat=jnp.asarray(seat, jnp.int32), board=jnp.asarray(s.board, jnp.int32)
    )

  def world_from_info(self, info: GomokuInfo) -> gomoku.GameState:
    """The exact engine state: the board determines every other field.

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    board = jnp.asarray(info.board, jnp.int32)
    n0 = jnp.sum(board == 0, dtype=jnp.int32)
    n1 = jnp.sum(board == 1, dtype=jnp.int32)
    won0 = jnp.any(jnp.all(board[self._win_lines] == 0, axis=1))
    won1 = jnp.any(jnp.all(board[self._win_lines] == 1, axis=1))
    term = won0 | won1 | jnp.all(board != -1)
    return gomoku.GameState(
        board=board,
        current_player=jnp.where(n0 > n1, jnp.int32(1), jnp.int32(0)),
        is_terminal=jnp.asarray(term, jnp.bool_),
        winner=jnp.where(
            won0, jnp.int32(0), jnp.where(won1, jnp.int32(1), jnp.int32(-1))
        ),
        legal_actions_mask=(board == -1) & ~term,
    )
