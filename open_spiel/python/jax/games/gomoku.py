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

"""Gomoku (k in a row), OpenSpiel's `gomoku`, on a square board.

State: a `GameState` with a row-major board of `board_size**2` cells (-1
empty, 0 / 1 a player's stone) and a cached legal-action mask; player 0
moves first. Actions: the cell index row * board_size + col, as in
OpenSpiel. Observation: f32[board_size, board_size, 3], planes (the
observer's stones, the opponent's stones, a constant plane equal to the
observer's id). Returns: +1 / -1 for `connect` in a row horizontally,
vertically or diagonally, 0 for a full board.
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
  """State of a Gomoku game.

  Attributes:
    board: i32[board_size**2]: -1 empty, 0 / 1 a player's stone.
    current_player: 0 or 1.
    is_terminal: bool.
    legal_actions_mask: bool[board_size**2].
    winner: -1 (none), 0 or 1.
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  legal_actions_mask: jax.Array


class Gomoku(game_lib.Game, interop.PyspielInterop):
  """Gomoku: `connect` in a row wins on a `board_size` square board.

  Attributes:
    board_size: The board is board_size x board_size.
    connect: The stones in a row that win.
    num_cells: The number of cells, board_size * board_size.
    num_cols: The number of columns.
    num_rows: The number of rows.
    win_lines: The cell indices of every winning line.
  """

  def __init__(self, board_size: int = 15, connect: int = 5):
    self.board_size = board_size
    self.connect = connect
    self.num_rows = board_size
    self.num_cols = board_size
    self.num_cells = board_size * board_size
    self.win_lines = self._make_win_lines()

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.num_cells

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.num_rows, self.num_cols, 3)

  def _make_win_lines(self):
    """Precomputes indices for all possible winning lines.

    Returns:
      int32[num_lines, connect], the cell indices of each line.
    """
    lines = []

    # Horizontal
    for r in range(self.num_rows):
      for c in range(self.num_cols - self.connect + 1):
        base = r * self.num_cols + c
        lines.append([base + i for i in range(self.connect)])

    # Vertical
    for r in range(self.num_rows - self.connect + 1):
      for c in range(self.num_cols):
        base = r * self.num_cols + c
        lines.append([base + i * self.num_cols for i in range(self.connect)])

    # Positive Diagonal (top-left to bottom-right)
    for r in range(self.num_rows - self.connect + 1):
      for c in range(self.num_cols - self.connect + 1):
        base = r * self.num_cols + c
        lines.append(
            [base + i * (self.num_cols + 1) for i in range(self.connect)]
        )

    # Negative Diagonal (top-right to bottom-left)
    for r in range(self.num_rows - self.connect + 1):
      for c in range(self.connect - 1, self.num_cols):
        base = r * self.num_cols + c
        lines.append(
            [base + i * (self.num_cols - 1) for i in range(self.connect)]
        )

    return np.array(lines, dtype=np.int32)

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return GameState(
        board=jnp.full((self.num_cells,), -1, dtype=jnp.int32),
        current_player=jnp.array(0, dtype=jnp.int32),
        is_terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        legal_actions_mask=jnp.ones((self.num_cells,), dtype=jnp.bool_),
    )

  def _check_win(self, board: jax.Array, player: jax.Array) -> jax.Array:
    """Returns bool[]: True if `player` has `connect` in a row.

    Args:
      board: The board.
      player: The player.

    Returns:
      bool[], whether `player` has a line.
    """
    player_mask = board == player
    line_values = player_mask[self.win_lines]
    won = jnp.any(jnp.all(line_values, axis=1))
    return won

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
    new_board = state.board.at[action].set(state.current_player)

    won = self._check_win(new_board, state.current_player)
    is_full = jnp.all(new_board != -1)

    # lax.select needs equal dtypes; a bare -1 is int64 when x64 is enabled.
    new_winner = jax.lax.select(won, state.current_player, jnp.int32(-1))
    is_terminal = won | is_full

    next_player = 1 - state.current_player

    new_legal_actions_mask = (new_board == -1) & (~is_terminal)

    return GameState(
        board=new_board,
        current_player=next_player,
        is_terminal=is_terminal,
        winner=new_winner,
        legal_actions_mask=new_legal_actions_mask,
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
        jnp.array(observing_player).astype(jnp.float32),
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
    os_obs = np.array(os_state.observation_tensor(player), dtype=np.float32)
    os_obs = os_obs.reshape((3, self.num_rows, self.num_cols))

    is_player_0 = player == 0
    plane0 = os_obs[0]
    plane1 = os_obs[1]

    my_pieces = jnp.where(is_player_0, plane0, plane1)
    opp_pieces = jnp.where(is_player_0, plane1, plane0)

    player_plane = jnp.full((self.num_rows, self.num_cols), float(player))

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )
