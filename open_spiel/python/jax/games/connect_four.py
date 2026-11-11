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

"""Connect Four, OpenSpiel's `connect_four`, on a configurable board.

State: a `GameState` with a row-major board of `rows * columns` cells (-1
empty, 0 / 1 the player's disc) whose row 0 is the top row, and a cached
legal-action mask. Actions: the column 0..columns-1 to drop a disc into, as
in OpenSpiel. Observation: f32[rows, columns, 3], planes (the observer's
discs, the opponent's discs, a constant plane equal to the observer's id).
Returns: +1 / -1 for four in a row / the opponent's four, 0 for a full
board. Applying an action to a terminal state leaves it unchanged.
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
  """State of a Connect Four game.

  Attributes:
    board: i32[rows * columns]: -1 empty, 0 / 1 a player's disc.
    current_player: 0 or 1.
    is_terminal: bool.
    legal_actions_mask: bool[columns].
    winner: -1 (none), 0 or 1.
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  legal_actions_mask: jax.Array


class ConnectFour(game_lib.Game, interop.PyspielInterop):
  """Connect Four with `rows` x `columns` cells (default 6 x 7).

  Attributes:
    num_cells: The number of cells, rows * columns.
    num_cols: The number of columns.
    num_rows: The number of rows.
    win_lines: The cell indices of every line of four.
  """

  def __init__(self, rows: int = 6, columns: int = 7):
    self.num_rows = rows
    self.num_cols = columns
    self.num_cells = rows * columns
    self.win_lines = self._make_win_lines()

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.num_cols

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.num_rows, self.num_cols, 3)

  def _make_win_lines(self):
    """Precomputes indices for all possible winning lines of 4-in-a-row.

    Returns:
      i32[num_lines, 4], the cell indices of each line.
    """
    lines = []

    # Horizontal
    for r in range(self.num_rows):
      for c in range(self.num_cols - 3):
        base = r * self.num_cols + c
        lines.append([base, base + 1, base + 2, base + 3])

    # Vertical
    for r in range(self.num_rows - 3):
      for c in range(self.num_cols):
        base = r * self.num_cols + c
        lines.append([
            base,
            base + self.num_cols,
            base + 2 * self.num_cols,
            base + 3 * self.num_cols,
        ])

    # Positive Diagonal
    for r in range(self.num_rows - 3):
      for c in range(self.num_cols - 3):
        base = r * self.num_cols + c
        lines.append([
            base,
            base + self.num_cols + 1,
            base + 2 * self.num_cols + 2,
            base + 3 * self.num_cols + 3,
        ])

    # Negative Diagonal
    for r in range(self.num_rows - 3):
      for c in range(3, self.num_cols):
        base = r * self.num_cols + c
        lines.append([
            base,
            base + self.num_cols - 1,
            base + 2 * self.num_cols - 2,
            base + 3 * self.num_cols - 3,
        ])

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
        legal_actions_mask=jnp.ones((self.num_cols,), dtype=jnp.bool_),
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
    col_mask = jnp.arange(self.num_cells) % self.num_cols == action
    is_empty = state.board == -1
    indices = jnp.arange(self.num_cells)
    valid_indices = jnp.where(col_mask & is_empty, indices, -1)
    drop_idx = jnp.max(valid_indices)

    new_board = state.board.at[drop_idx].set(state.current_player)

    player_mask = new_board == state.current_player
    line_values = player_mask[self.win_lines]
    won = jnp.any(jnp.all(line_values, axis=1))

    is_full = jnp.all(new_board != -1)

    # Selecting state.winner (int32) rather than a literal keeps lax.select's
    # operand dtypes equal when x64 is enabled.
    new_winner = jax.lax.select(won, state.current_player, state.winner)
    is_terminal = won | is_full

    top_cells = new_board[: self.num_cols]
    new_legal_actions_mask = (top_cells == -1) & (~is_terminal)

    next_player = 1 - state.current_player

    return GameState(
        board=jnp.where(state.is_terminal, state.board, new_board),
        current_player=jnp.where(
            state.is_terminal, state.current_player, next_player
        ),
        is_terminal=jnp.where(
            state.is_terminal, state.is_terminal, is_terminal
        ),
        winner=jnp.where(state.is_terminal, state.winner, new_winner),
        legal_actions_mask=jnp.where(
            state.is_terminal, state.legal_actions_mask, new_legal_actions_mask
        ),
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
    my_pieces = jnp.where(is_player_0, os_obs[0], os_obs[1])
    opp_pieces = jnp.where(is_player_0, os_obs[1], os_obs[0])

    # OpenSpiel's row 0 is the bottom row; this game's is the top row.
    my_pieces = my_pieces[::-1, :]
    opp_pieces = opp_pieces[::-1, :]

    player_plane = jnp.full((self.num_rows, self.num_cols), float(player))

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )
