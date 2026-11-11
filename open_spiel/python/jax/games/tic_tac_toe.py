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

"""Tic-tac-toe, OpenSpiel's `tic_tac_toe`.

State: a `GameState` with a 9-cell row-major board (-1 empty, 0 for x,
1 for o); player 0 (x) moves first. Actions: the cell index 0-8, as in
OpenSpiel. Observation: f32[3, 3, 3], planes (the observer's marks, the
opponent's marks, a constant plane equal to the observer's id). Returns:
+1 / -1 for a win / loss, 0 for a draw.
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

_WIN_LINES = np.array(
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


class GameState(NamedTuple):
  """State of a Tic-Tac-Toe game.

  Attributes:
    board: Shape (9,), values -1 (empty), 0 (P1/X), 1 (P2/O).
    current_player: 0 or 1.
    is_terminal: bool.
    winner: -1, 0, 1.
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array


class TicTacToe(game_lib.Game, interop.PyspielInterop):
  """Tic-tac-toe on a 3x3 board.

  Attributes:
    num_cells: The number of cells, rows * columns.
    num_cols: The number of columns.
    num_rows: The number of rows.
    win_lines: The cell indices of every line of three.
  """

  def __init__(self):
    self.num_rows = 3
    self.num_cols = 3
    self.num_cells = 9
    self.win_lines = _WIN_LINES

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return 9

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (3, 3, 3)

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
    new_board = state.board.at[action].set(state.current_player)

    player_mask = new_board == state.current_player
    line_values = player_mask[self.win_lines]
    won = jnp.any(jnp.all(line_values, axis=1))

    is_full = jnp.all(new_board != -1)

    new_winner = jax.lax.select(won, state.current_player, state.winner)
    is_terminal = won | is_full

    next_player = 1 - state.current_player

    return GameState(
        board=new_board,
        current_player=next_player,
        is_terminal=is_terminal,
        winner=new_winner,
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
    return (state.board == -1) & (~state.is_terminal)

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
    # OpenSpiel's planes are (empty, o, x).
    os_obs = np.asarray(os_state.observation_tensor(player), dtype=np.float32)
    os_obs = os_obs.reshape(3, self.num_rows, self.num_cols)

    is_player_0 = player == 0
    my_pieces = os_obs[2] if is_player_0 else os_obs[1]
    opp_pieces = os_obs[1] if is_player_0 else os_obs[2]

    player_plane = jnp.full((self.num_rows, self.num_cols), float(player))
    return jnp.stack(
        [jnp.asarray(my_pieces), jnp.asarray(opp_pieces), player_plane],
        axis=-1,
    ).astype(jnp.float32)
