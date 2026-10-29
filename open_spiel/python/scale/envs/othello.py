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

"""Othello Env over the JAX engine `othello`, with a normalized board.

Actions are pyspiel's (and the engine's): row * 8 + col for a placement (row
0 = rank 1, col 0 = file a), 64 = pass, legal only when the seat to move has
no placement. Black (seat 0) moves first. The game ends when neither side
can place (no closing double pass, as in pyspiel); more discs wins, equal
is a draw.

Conventions normalized here: the InfoState board is i32[64], -1 empty, 0
black, 1 white (the engine's raw board is 0 empty, 1 black, 2 white, [8, 8]);
terminal reports actor -1 (the engine keeps the last mover). Because of
passes the seat to move is not a function of the board, so the InfoState
carries it (to_move, -1 at terminal) and the engine's consecutive-pass count.
max_len 120: at most 60 placements, and every pass is followed by a
placement.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import othello
from open_spiel.python.scale.envs import jax_game_env

ROWS = COLS = 8
NUM_CELLS = ROWS * COLS
PASS = othello.ACTION_PASS  # 64


class OthelloInfo(NamedTuple):
  """What one seat sees: the whole position.

  Attributes:
    seat: i32 the seat.
    board: i32[64]: -1 empty, 0 black, 1 white.
    to_move: i32 the seat to move, -1 at terminal.
    passes: i32 consecutive passes just made (0 or 1).
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[64]: -1 empty, 0 black, 1 white
  to_move: jax.Array  # i32: the seat to move, -1 at terminal
  passes: jax.Array  # i32: consecutive passes just made (0 or 1)


class OthelloEnv(jax_game_env.JaxGameEnv):
  """Othello: 2 seats, 65 actions (64 cells + pass)."""

  name = "othello"
  max_len = 2 * (NUM_CELLS - 4)
  zero_sum = True

  def __init__(self):
    super().__init__(othello.Othello())
    assert self.max_chance_outcomes == 0 and self.num_actions == NUM_CELLS + 1
    assert PASS == NUM_CELLS

  def info_state(self, s: othello.GameState, seat) -> OthelloInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return OthelloInfo(
        seat=jnp.asarray(seat, jnp.int32),
        board=jnp.asarray(s.board, jnp.int32).reshape(-1) - 1,
        to_move=self.actor(s),
        passes=jnp.asarray(s.pass_count, jnp.int32),
    )

  def world_from_info(self, info: OthelloInfo) -> othello.GameState:
    """The engine state of the position.

    The engine state (exact before the end; at terminal the engine's
    current_player is the last mover, not recorded, and is set to 0).

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    board = jnp.asarray(info.board, jnp.int32).reshape(ROWS, COLS) + 1
    to_move = jnp.asarray(info.to_move, jnp.int32)
    term = to_move == -1
    black, white = jnp.sum(board == 1), jnp.sum(board == 2)
    winner = jnp.where(
        term & (black > white),
        jnp.int32(0),
        jnp.where(term & (white > black), jnp.int32(1), jnp.int32(-1)),
    )
    return othello.GameState(
        board=board,
        current_player=jnp.where(term, jnp.int32(0), to_move),
        is_terminal=jnp.asarray(term, jnp.bool_),
        winner=winner,
        pass_count=jnp.asarray(info.passes, jnp.int32),
    )
