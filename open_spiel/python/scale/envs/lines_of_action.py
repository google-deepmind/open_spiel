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

"""Lines of Action Env over the JAX engine `lines_of_action`.

With OpenSpiel's rules.

OpenSpiel `lines_of_action` is the reference; serving replays pyspiel
histories through this env. The engine implements OpenSpiel's rules
(repetition draws over every board of the game, the move limit and repetition
taking precedence over a connection, a full flood fill for connectivity);
envs/tests/lines_of_action_test.py has a known answer for each and replays
random pyspiel games node by node.

Board: 64 cells row-major (row 0 = pyspiel's row 0), -1 empty, 0 black
(seat 0, moves first), 1 white. Action = mixed-radix (from_row, from_col,
to_row, to_col, capture) over (8, 8, 8, 8, 2), pyspiel's encoding
(`encode_action`). The seats alternate, so the player to move is
move_count % 2.

The InfoState is the position (board, move count), not the repetition
history: world_from_info gives the position with only itself seen, so a
search from it does not know which earlier boards would repeat.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import lines_of_action
from open_spiel.python.scale.envs import jax_game_env

I32 = jnp.int32
ROWS = COLS = 8
NUM_CELLS = ROWS * COLS
NUM_ACTIONS = NUM_CELLS * NUM_CELLS * 2
DEFAULT_MAX_GAME_LENGTH = 1000  # OpenSpiel kDefaultMaxGameLength
EMPTY, BLACK, WHITE = -1, 0, 1


def encode_action(
    from_row: int, from_col: int, to_row: int, to_col: int, capture: int
) -> int:
  """The pyspiel action id: RankActionMixedBase over (8, 8, 8, 8, 2).

  Args:
    from_row: The origin row.
    from_col: The origin column.
    to_row: The destination row.
    to_col: The destination column.
    capture: 1 for a capture, else 0.

  Returns:
    The action id.
  """
  for v, n in (
      (from_row, ROWS),
      (from_col, COLS),
      (to_row, ROWS),
      (to_col, COLS),
      (capture, 2),
  ):
    assert 0 <= int(v) < n, (from_row, from_col, to_row, to_col, capture)
  return (
      ((int(from_row) * COLS + int(from_col)) * ROWS + int(to_row)) * COLS
      + int(to_col)
  ) * 2 + int(capture)


class LinesOfActionState(NamedTuple):
  """The env state: the engine state.

  Attributes:
    game: The engine state; it carries every board of the game (repetition).
  """

  # carries every board of the game (repetition)
  game: lines_of_action.GameState


class LinesOfActionInfo(NamedTuple):
  """The position (perfect information) seen by `seat`.

  Attributes:
    seat: i32.
    board: i32[64]: -1 empty, 0 black, 1 white.
    move_count: i32: moves played.
  """

  seat: jax.Array
  board: jax.Array
  move_count: jax.Array


def check_max_game_length(n) -> int:
  """The max_game_length option `n` as an int.

  Raises TypeError / ValueError on a non-int or < 1.

  Args:
    n: The max_game_length option.

  Returns:
    The max game length.

  Raises:
    TypeError: A non-int.
    ValueError: A value < 1.
  """
  if isinstance(n, bool) or not isinstance(n, int):
    raise TypeError(
        f"lines_of_action max_game_length must be an int, got {n!r}"
    )
  if n < 1:
    raise ValueError(f"lines_of_action max_game_length must be >= 1, got {n}")
  return int(n)


class LinesOfActionEnv(jax_game_env.JaxGameEnv):
  """Lines of Action (8x8): 2 seats, 8192 actions.

  Attributes:
    max_game_length: The game's move limit.
    max_len: The longest game in decisions (max_game_length).
  """

  name = "lines_of_action"
  zero_sum = True

  def __init__(self, max_game_length: int = DEFAULT_MAX_GAME_LENGTH):
    self.max_game_length = check_max_game_length(max_game_length)
    self.max_len = self.max_game_length
    super().__init__(
        lines_of_action.LinesOfAction(
            rows=ROWS, columns=COLS, max_game_length=self.max_game_length
        )
    )
    assert self.max_chance_outcomes == 0 and self.num_actions == NUM_ACTIONS
    self._current_player = lambda s: s.game.current_player

  # -- raw layer
  # ------------------------------------------------------------------
  def root(self) -> LinesOfActionState:
    """The game's initial state, chance unresolved.

    Returns:
      The root state (it may be a chance node).
    """
    return LinesOfActionState(game=self.game.new_initial_state())

  def apply(self, s: LinesOfActionState, a) -> LinesOfActionState:
    """Applies player action `a`; chance stays unresolved.

    Args:
      s: The state.
      a: The player action id.

    Returns:
      The successor state.
    """
    return LinesOfActionState(
        game=self.game.apply_action(s.game, jnp.asarray(a, I32))
    )

  # -- queries
  # --------------------------------------------------------------------
  def terminal(self, s: LinesOfActionState) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s.game.is_terminal, jnp.bool_)

  def legal_mask(self, s: LinesOfActionState) -> jax.Array:
    """The actor's legal actions.

    Args:
      s: The state.

    Returns:
      bool[A] mask; all False at terminal.
    """
    return jnp.asarray(s.game.legal_actions_mask, jnp.bool_)

  def returns(self, s: LinesOfActionState) -> jax.Array:
    """Per-seat returns.

    Args:
      s: The state.

    Returns:
      f32[P] returns; meaningful at terminal.
    """
    return jnp.asarray(self.game.returns(s.game), jnp.float32)

  def info_state(self, s: LinesOfActionState, seat) -> LinesOfActionInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    return LinesOfActionInfo(
        seat=jnp.asarray(seat, I32),
        board=jnp.asarray(s.game.board, I32),
        move_count=jnp.asarray(s.game.move_count, I32),
    )

  def world_from_info(self, info: LinesOfActionInfo) -> LinesOfActionState:
    """The position with only itself seen.

    (the earlier boards are not in the InfoState): terminal by a connection, the
    move limit or a stalemate exactly as after a move.

    Args:
      info: One seat's InfoState.

    Returns:
      The env state.
    """
    g = self.game
    board = jnp.asarray(info.board, I32)
    move_count = jnp.asarray(info.move_count, I32)
    player = (move_count % 2).astype(I32)
    mover = 1 - player  # who made the last move
    mover_conn = g.is_connected(board, mover)
    other_conn = g.is_connected(board, player)
    conn_winner = jnp.where(
        mover_conn, mover, jnp.where(other_conn, player, I32(-1))
    )
    at_limit = move_count >= self.max_game_length
    legal = g.legal_actions_for(board, player)
    started = move_count > 0
    stalemate = started & ~jnp.any(legal)
    winner = jnp.where(
        started & at_limit,
        I32(2),
        jnp.where(
            started & (conn_winner >= 0),
            conn_winner,
            jnp.where(stalemate, mover, I32(-1)),
        ),
    )
    term = winner >= 0
    state = lines_of_action.GameState(
        board=board,
        current_player=player,
        is_terminal=term,
        winner=winner.astype(I32),
        legal_actions_mask=legal & ~term,
        move_count=move_count,
        board_history=g.board_history_from(board, move_count),
    )
    return LinesOfActionState(game=state)
