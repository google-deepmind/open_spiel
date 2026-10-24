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

"""Go Env over the JAX engine `go` (perfect information; 9x9 by default).

Rules are the engine's, which match OpenSpiel `go` (envs/tests/go_test.py
replays random pyspiel games node by node): Tromp-Taylor area scoring with
komi, simple ko, suicide illegal, the game ends after two consecutive passes
or max_game_length = 2 * board_size**2 moves (OpenSpiel's default for every
size), or as a draw when a move recreates an earlier position (OpenSpiel's
positional superko). No handicap: black (seat 0) moves first and the
seats alternate on every move, so the player to move is move_count % 2.

Board: board_size**2 cells, row-major in pyspiel's action order; action
board_size**2 is pass. The engine stores a chain id per stone (> 0 black,
< 0 white, 0 empty); the InfoState carries it, and whether the last move
ended the game by superko, so world_from_info rebuilds the position, its
legal moves and whether (and how) the game is over exactly. It does not
rebuild the earlier positions (world_from_info).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import go
from open_spiel.python.scale.envs import jax_game_env

I32 = jnp.int32


class GoInfo(NamedTuple):
  """The whole position (perfect information), seen by `seat`.

  Attributes:
    seat: i32.
    board: i32[N*N]: the engine's chain ids, > 0 black, < 0 white, 0 empty.
    ko_point: i32: the cell recapture is forbidden on, -1 none.
    consecutive_passes: i32.
    move_count: i32: moves played (passes included).
    last_action: i32: -1 at the root.
    superko: i32: 1 when the last move recreated an earlier position (game over,
      a draw).
  """

  seat: jax.Array
  board: jax.Array
  ko_point: jax.Array
  consecutive_passes: jax.Array
  move_count: jax.Array
  last_action: jax.Array
  superko: jax.Array


def check_board_size(board_size) -> int:
  """board_size as an int.

  Raises TypeError / ValueError on a non-int or < 2.

  Args:
    board_size: The board size option.

  Returns:
    The board size.

  Raises:
    TypeError: A non-int.
    ValueError: A size < 2.
  """
  if isinstance(board_size, bool) or not isinstance(board_size, int):
    raise TypeError(f"go board_size must be an int, got {board_size!r}")
  if board_size < 2:
    raise ValueError(f"go board_size must be >= 2, got {board_size}")
  return int(board_size)


def check_komi(komi) -> float:
  """Komi as a float.

  Raises TypeError / ValueError on a non-number or non-finite.

  Args:
    komi: The komi option.

  Returns:
    The komi.

  Raises:
    TypeError: A non-number.
    ValueError: A non-finite number.
  """
  if isinstance(komi, bool) or not isinstance(komi, (int, float)):
    raise TypeError(f"go komi must be a number, got {komi!r}")
  if not math.isfinite(komi):
    raise ValueError(f"go komi must be finite, got {komi!r}")
  return float(komi)


class GoEnv(jax_game_env.JaxGameEnv):
  """Go: 2 seats, board_size**2 + 1 actions (the last is pass).

  Attributes:
    board_size: The board side.
    komi: The komi added to white's score.
    num_cells: board_size ** 2.
    pass_action: The pass action id (num_cells).
    max_len: The longest game in decisions (2 * num_cells).
  """

  name = "go"
  zero_sum = True

  def __init__(self, board_size: int = 9, komi: float = 7.5):
    self.board_size = check_board_size(board_size)
    self.komi = check_komi(komi)
    self.num_cells = self.board_size * self.board_size
    self.pass_action = self.num_cells
    self.max_len = 2 * self.num_cells
    super().__init__(go.Go(board_size=self.board_size, komi=self.komi))
    assert self.max_chance_outcomes == 0
    assert self.game.max_game_length == self.max_len, (
        self.game.max_game_length,
        self.max_len,
    )
    assert self.num_actions == self.num_cells + 1

  def info_state(self, s: go.GameState, seat) -> GoInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return GoInfo(
        seat=jnp.asarray(seat, I32),
        board=jnp.asarray(s.board, I32),
        ko_point=jnp.asarray(s.ko_point, I32),
        consecutive_passes=jnp.asarray(s.consecutive_passes, I32),
        move_count=jnp.asarray(s.move_count, I32),
        last_action=jnp.asarray(s.last_action, I32),
        superko=jnp.asarray(s.superko, I32),
    )

  def world_from_info(self, info: GoInfo) -> go.GameState:
    """The engine state of the position.

    The rest follows from it, as in the engine's apply_action (terminal,
    Tromp-Taylor winner, legal moves).

    The InfoState carries the position and whether the last move ended the
    game by superko (so such a game rebuilds as over, a draw), not the
    game's earlier positions: the rebuilt state's superko history holds only
    this position (go.Go.position_history), so in a game continued from it
    a later repetition of an earlier position is not detected (a known
    limitation). Superko is rare (a cycle of captures), and every
    other field is exact.

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    g = self.game
    board = jnp.asarray(info.board, I32)
    move_count = jnp.asarray(info.move_count, I32)
    passes = jnp.asarray(info.consecutive_passes, I32)
    ko = jnp.asarray(info.ko_point, I32)
    superko = jnp.asarray(info.superko, I32) != 0
    player = (move_count % 2).astype(I32)
    term = (passes >= 2) | (move_count >= self.max_len) | superko
    score = g.tromp_taylor_score(board)
    scored = term & ~superko  # a superko end is a draw, not scored
    winner = jnp.where(
        scored & (score > 0),
        I32(0),
        jnp.where(scored & (score < 0), I32(1), I32(-1)),
    )
    legal = g.legal_actions_for(board, 1 - 2 * player, ko)
    legal = jnp.where(term, jnp.zeros_like(legal), legal)
    return go.GameState(
        board=board,
        current_player=player,
        is_terminal=term,
        winner=winner,
        legal_actions_mask=legal,
        move_count=move_count,
        last_action=jnp.asarray(info.last_action, I32),
        ko_point=ko,
        consecutive_passes=passes,
        position_hashes=g.position_history(board, move_count),
        superko=superko,
    )
