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

"""Hex Env over the JAX engine `hex`, without the swap rule.

Conventions:
  * actions: cell = row * size + col for both seats, OpenSpiel's ids (a1 =
    0, b1 = 1). The swap rule is off (OpenSpiel's default), so there are
    size * size actions.
  * board: row-major, -1 empty, 0 x (seat 0, moves first, joins the top and
    bottom rows), 1 o (seat 1, joins the left and right columns). Row r is
    shifted right of row r-1: (r, c) touches (r-1, c), (r-1, c+1), (r, c-1),
    (r, c+1), (r+1, c-1), (r+1, c).
  * terminal: actor -1, empty legal mask. Hex has no draws.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import hex as hex_game
from open_spiel.python.scale.envs import jax_game_env


class HexInfo(NamedTuple):
  """What one seat sees: the whole board.

  Attributes:
    seat: i32 the seat.
    board: i32[size * size]: -1 empty, 0 x, 1 o.
  """

  seat: jax.Array  # i32
  board: jax.Array  # i32[size * size]: -1 empty, 0 x, 1 o


def _neighbours(g: jax.Array) -> jax.Array:
  """bool[n, n]: the cells touching a True cell of g (hex adjacency above).

  Args:
    g: bool[n, n], the cells.

  Returns:
    bool[n, n], the touching cells.
  """
  f = jnp.zeros_like(g)
  out = f
  out = out | f.at[:-1, :].set(g[1:, :])  # from (r+1, c)
  out = out | f.at[1:, :].set(g[:-1, :])  # from (r-1, c)
  out = out | f.at[:, :-1].set(g[:, 1:])  # from (r, c+1)
  out = out | f.at[:, 1:].set(g[:, :-1])  # from (r, c-1)
  out = out | f.at[:-1, 1:].set(g[1:, :-1])  # from (r+1, c-1)
  out = out | f.at[1:, :-1].set(g[:-1, 1:])  # from (r-1, c+1)
  return out


def connected(stones: jax.Array, seat: int) -> jax.Array:
  """Whether the seat's stones join its two edges.

  bool: the seat's stones (bool[n, n]) join its two edges.

  Args:
    stones: bool[n, n] the seat's stones.
    seat: The seat.

  Returns:
    bool.
  """
  n = stones.shape[0]
  start = jnp.zeros_like(stones)
  start = (
      start.at[0, :].set(stones[0, :])
      if seat == 0
      else start.at[:, 0].set(stones[:, 0])
  )
  reach = jax.lax.fori_loop(
      0, n * n, lambda _, g: g | (_neighbours(g) & stones), start
  )
  return jnp.any(reach[-1, :]) if seat == 0 else jnp.any(reach[:, -1])


class HexEnv(jax_game_env.JaxGameEnv):
  """Hex: 2 seats, size * size actions (cells).

  Attributes:
    board_size: The board side.
    max_len: The longest game in decisions (board_size ** 2).
    num_actions: The number of actions, one per cell.
  """

  name = "hex"
  zero_sum = True

  def __init__(self, board_size: int = 11):
    if not isinstance(board_size, int) or board_size < 2:
      raise ValueError(
          f"hex board_size must be an int >= 2, got {board_size!r}"
      )
    self.board_size = board_size
    self.max_len = board_size * board_size
    super().__init__(hex_game.Hex(board_size=board_size, swap=False))
    assert self.max_chance_outcomes == 0
    # no swap id without the swap rule
    assert self.game.num_actions == self.max_len
    self.num_actions = self.max_len

  def legal_mask(self, s: hex_game.GameState) -> jax.Array:
    """The actor's legal actions.

    Args:
      s: The state.

    Returns:
      bool[A] mask; all False at terminal.
    """
    return (jnp.asarray(s.board, jnp.int32) == -1) & ~self.terminal(s)

  def info_state(self, s: hex_game.GameState, seat) -> HexInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return HexInfo(
        seat=jnp.asarray(seat, jnp.int32), board=jnp.asarray(s.board, jnp.int32)
    )

  def world_from_info(self, info: HexInfo) -> hex_game.GameState:
    """The exact engine state: the board determines every other field.

    Args:
      info: One seat's InfoState.

    Returns:
      The engine state.
    """
    n = self.board_size
    board = jnp.asarray(info.board, jnp.int32)
    b2 = board.reshape(n, n)
    n0 = jnp.sum(board == 0, dtype=jnp.int32)
    n1 = jnp.sum(board == 1, dtype=jnp.int32)
    won0, won1 = connected(b2 == 0, 0), connected(b2 == 1, 1)
    term = won0 | won1
    cp = jnp.where(n0 > n1, jnp.int32(1), jnp.int32(0))
    empty = ((b2 == -1) & ~term).reshape(-1)
    return hex_game.GameState(
        board=board,
        current_player=cp,
        is_terminal=jnp.asarray(term, jnp.bool_),
        winner=jnp.where(
            won0, jnp.int32(0), jnp.where(won1, jnp.int32(1), jnp.int32(-1))
        ),
        legal_actions_mask=empty,
        move_count=n0 + n1,
    )
