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

"""Connect Four Env over the JAX connect_four game (perfect information).

Board: 42 cells, row-major, row 0 at the TOP (the JAX game's layout); -1 empty,
0 seat 0, 1 seat 1. The JAX connect_four game has no current_player() method and
leaves current_player at the post-move value after terminal; the adapter
reports actor -1 and an empty legal mask at terminal.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import connect_four
from open_spiel.python.scale.envs import jax_game_env

ROWS, COLS = 6, 7
NUM_CELLS = ROWS * COLS


class ConnectFourInfo(NamedTuple):
  """What a seat observes: the whole board.

  Attributes:
    seat: i32.
    board: i32[42]: -1 empty, 0 / 1 the seat that played there.
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[42]: -1 empty, 0 / 1 the seat that played there


class ConnectFourEnv(jax_game_env.JaxGameEnv):
  """Connect Four: 2 seats, 7 actions (columns)."""

  name = "connect_four"
  max_len = NUM_CELLS
  zero_sum = True

  def __init__(self):
    super().__init__(connect_four.ConnectFour())
    assert self.max_chance_outcomes == 0
    assert (self.game.num_rows, self.game.num_cols) == (ROWS, COLS)
    self._win_lines = np.asarray(self.game.win_lines, np.int32)

  def info_state(self, s: connect_four.GameState, seat) -> ConnectFourInfo:
    """What `seat` observes (the whole board).

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      The seat's ConnectFourInfo.
    """
    return ConnectFourInfo(
        seat=jnp.asarray(seat, jnp.int32), board=jnp.asarray(s.board, jnp.int32)
    )

  def world_from_info(self, info: ConnectFourInfo) -> connect_four.GameState:
    """The exact game state: the board determines every other field.

    Args:
      info: A ConnectFourInfo.

    Returns:
      The game state.
    """
    board = jnp.asarray(info.board, jnp.int32)
    n0 = jnp.sum(board == 0, dtype=jnp.int32)
    n1 = jnp.sum(board == 1, dtype=jnp.int32)
    won0 = jnp.any(jnp.all(board[self._win_lines] == 0, axis=1))
    won1 = jnp.any(jnp.all(board[self._win_lines] == 1, axis=1))
    term = won0 | won1 | jnp.all(board != -1)
    return connect_four.GameState(
        board=board,
        current_player=jnp.where(n0 > n1, jnp.int32(1), jnp.int32(0)),
        is_terminal=jnp.asarray(term, jnp.bool_),
        winner=jnp.where(
            won0, jnp.int32(0), jnp.where(won1, jnp.int32(1), jnp.int32(-1))
        ),
        legal_actions_mask=(board[:COLS] == -1) & ~term,
    )
