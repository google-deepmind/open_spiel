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

"""Tic-tac-toe Env over the JAX engine `tic_tac_toe` (perfect information).

At a terminal state actor() is -1 and the legal mask is empty (the engine's
state keeps alternating current_player there). Board cells: -1 empty, 0 X
(seat 0), 1 O (seat 1).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import tic_tac_toe
from open_spiel.python.scale.envs import jax_game_env

NUM_CELLS = 9
WIN_LINES = np.array(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ],
    dtype=np.int32,
)


class TicTacToeInfo(NamedTuple):
  """What one seat sees: the whole board.

  Attributes:
    seat: i32 the seat.
    board: i32[9]: -1 empty, 0 X, 1 O.
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[9]: -1 empty, 0 X, 1 O


class TicTacToeEnv(jax_game_env.JaxGameEnv):
  """Tic-tac-toe: 2 seats, 9 actions (cells, row-major)."""

  name = "tic_tac_toe"
  max_len = NUM_CELLS
  zero_sum = True

  def __init__(self):
    super().__init__(tic_tac_toe.TicTacToe())
    assert self.max_chance_outcomes == 0
    np.testing.assert_array_equal(np.asarray(self.game.win_lines), WIN_LINES)

  def info_state(self, s: tic_tac_toe.GameState, seat) -> TicTacToeInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return TicTacToeInfo(
        seat=jnp.asarray(seat, jnp.int32), board=jnp.asarray(s.board, jnp.int32)
    )

  def world_from_info(self, info: TicTacToeInfo) -> tic_tac_toe.GameState:
    """The exact engine state: the board determines every other field.

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    board = jnp.asarray(info.board, jnp.int32)
    n0 = jnp.sum(board == 0, dtype=jnp.int32)
    n1 = jnp.sum(board == 1, dtype=jnp.int32)
    won0 = jnp.any(jnp.all(board[WIN_LINES] == 0, axis=1))
    won1 = jnp.any(jnp.all(board[WIN_LINES] == 1, axis=1))
    winner = jnp.where(
        won0, jnp.int32(0), jnp.where(won1, jnp.int32(1), jnp.int32(-1))
    )
    return tic_tac_toe.GameState(
        board=board,
        current_player=jnp.where(n0 > n1, jnp.int32(1), jnp.int32(0)),
        is_terminal=jnp.asarray(won0 | won1 | jnp.all(board != -1), jnp.bool_),
        winner=winner,
    )
