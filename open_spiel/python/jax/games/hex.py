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

"""Hex, OpenSpiel's `hex`, on a square board, with an optional swap rule.

The swap rule is off by default, as in OpenSpiel.

State: a `GameState` with a row-major board of `board_size**2` cells (-1
empty, 0 black, 1 white) and a cached legal-action mask. Black (player 0)
moves first and joins the top and bottom rows; white joins the left and
right columns. Row r is shifted half a cell right of row r - 1, so (r, c)
touches (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1) and (r+1, c).

Actions, as in OpenSpiel: the cell row * board_size + col, and with
`swap=True` the swap action `board_size**2`, legal for white's first move: it
replaces black's stone by a white stone mirrored across the long diagonal,
and black moves next. The swap rule is off by default, as in OpenSpiel, and
then the game has `board_size**2` actions; with `swap=True` it has one more.

Observation: f32[board_size, board_size, 3], planes (the observer's stones,
the opponent's stones, a constant plane equal to the observer's id), in board
orientation for both players. Returns: +1 / -1 for connecting one's edges.
Hex has no draws.

`canonical_view(game)` gives the same game transposed for white, so both
players connect the top and bottom rows of their own view.
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
  """State of a Hex game.

  Attributes:
    board: i32[board_size**2]: -1 empty, 0 black, 1 white.
    current_player: 0 or 1.
    is_terminal: bool.
    legal_actions_mask: bool[num_cells] the legal actions.
    move_count: int32.
    winner: -1 (none), 0 or 1.
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  # bool[num_actions]: the cells, then swap with swap=True
  legal_actions_mask: jax.Array
  move_count: jax.Array


class Hex(game_lib.Game, interop.PyspielInterop):
  """Hex on a `board_size` square board (default 11).

  The swap rule is off by default (OpenSpiel's defaults).

  Attributes:
    board_size: The board is board_size x board_size.
    num_cells: The number of cells, board_size * board_size.
    swap: Whether the swap rule is on.
  """

  def __init__(self, board_size: int = 11, swap: bool = False):
    if not isinstance(swap, (bool, np.bool_)):
      raise TypeError(f"hex swap must be a bool, got {swap!r}")
    self.board_size = board_size
    self.swap = bool(swap)
    self.num_cells = board_size * board_size

  @property
  def num_actions(self) -> int:
    """The cells, plus the swap action with swap=True (OpenSpiel's count)."""
    return self.num_cells + int(self.swap)

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.board_size, self.board_size, 3)

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
        legal_actions_mask=jnp.arange(self.num_actions) < self.num_cells,
        move_count=jnp.array(0, dtype=jnp.int32),
    )

  def _check_win(
      self,
      board: jax.Array,
      player: jax.Array,
      last_action: jax.Array,
      max_iter: jax.Array,
  ) -> jax.Array:
    """Whether `player` connects their edges.

    The connection runs through the stone on `last_action`.

    Args:
      board: The board.
      player: The player.
      last_action: The cell of the stone just placed.
      max_iter: The bound on flood-fill iterations.

    Returns:
      bool[]: True if `player` connects their edges.
    """
    board_2d = board.reshape(self.board_size, self.board_size)
    player_mask = board_2d == player

    r = last_action // self.board_size
    c = last_action % self.board_size

    # Initial grid has only the last placed piece
    init_grid = (
        jnp.zeros((self.board_size, self.board_size), dtype=jnp.bool_)
        .at[r, c]
        .set(True)
    )

    def cond_fun(val):
      """Returns whether the flood fill is still growing.

      Args:
        val: (grid, old grid, count).

      Returns:
        bool[].
      """
      grid, old_grid, count = val
      changed = jnp.any(grid != old_grid)
      return changed & (count < max_iter)

    def body_fun(val):
      """Grows the flood fill by one neighbour step.

      Args:
        val: (grid, old grid, count).

      Returns:
        (new grid, grid, count + 1).
      """
      grid, _, count = val

      shifted_up = jnp.roll(grid, shift=-1, axis=0).at[-1].set(False)
      shifted_up_right = jnp.roll(
          jnp.roll(grid, shift=-1, axis=0), shift=1, axis=1
      )
      shifted_up_right = shifted_up_right.at[-1].set(False)
      shifted_up_right = shifted_up_right.at[:, 0].set(False)
      shifted_left = jnp.roll(grid, shift=-1, axis=1).at[:, -1].set(False)
      shifted_right = jnp.roll(grid, shift=1, axis=1).at[:, 0].set(False)
      shifted_down_left = jnp.roll(
          jnp.roll(grid, shift=1, axis=0), shift=-1, axis=1
      )
      shifted_down_left = shifted_down_left.at[0].set(False)
      shifted_down_left = shifted_down_left.at[:, -1].set(False)
      shifted_down = jnp.roll(grid, shift=1, axis=0).at[0].set(False)

      new_grid = grid | (
          (
              shifted_up
              | shifted_up_right
              | shifted_left
              | shifted_right
              | shifted_down_left
              | shifted_down
          )
          & player_mask
      )
      return new_grid, grid, count + 1

    final_grid, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (init_grid, jnp.zeros_like(init_grid), jnp.array(0)),
    )

    # P0 (Black) wins if connected to top and bottom
    win_p0 = jnp.any(final_grid[0]) & jnp.any(final_grid[-1])

    # P1 (White) wins if connected to left and right
    win_p1 = jnp.any(final_grid[:, 0]) & jnp.any(final_grid[:, -1])

    return jax.lax.select(player == 0, win_p0, win_p1)

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
    is_swap = (jnp.asarray(action) == self.num_cells) & self.swap
    abs_action = jnp.where(is_swap, 0, action).astype(jnp.int32)

    normal_board = state.board.at[abs_action].set(
        state.current_player.astype(jnp.int32)
    )

    # The swap mirrors black's only stone across the long diagonal.
    # jnp.argmax returns int64 under x64; lax.select below needs int32.
    first_move_idx = jnp.argmax(state.board != -1).astype(jnp.int32)
    r = first_move_idx // self.board_size
    c = first_move_idx % self.board_size
    mirrored_move = c * self.board_size + r

    swap_board = jnp.full((self.num_cells,), -1, dtype=jnp.int32)
    swap_board = swap_board.at[mirrored_move].set(jnp.array(1, dtype=jnp.int32))

    new_board = jax.lax.select(is_swap, swap_board, normal_board)

    new_move_count = state.move_count + 1
    actual_action = jax.lax.select(is_swap, mirrored_move, abs_action)
    max_iter = (new_move_count + 1) // 2
    # Only the mover can complete a connection.
    won = self._check_win(
        new_board, state.current_player, actual_action, max_iter
    )

    # lax.select needs equal dtypes; a bare literal is int64 under x64.
    new_winner = jax.lax.select(won, state.current_player, jnp.int32(-1))
    is_terminal = won

    next_player = jax.lax.select(
        is_swap, jnp.int32(0), 1 - state.current_player
    )
    swap_legal = (
        (new_move_count == 1) & (next_player == 1) & (~is_terminal) & self.swap
    )
    new_legal_actions_mask = (new_board == -1) & (~is_terminal)
    if self.swap:
      new_legal_actions_mask = jnp.append(new_legal_actions_mask, swap_legal)

    return GameState(
        board=new_board,
        current_player=next_player,
        is_terminal=is_terminal,
        winner=new_winner,
        legal_actions_mask=new_legal_actions_mask,
        move_count=new_move_count,
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
    board_2d = state.board.reshape(self.board_size, self.board_size)
    mine = board_2d == observing_player
    theirs = board_2d == (1 - observing_player)
    player_plane = jnp.full(
        (self.board_size, self.board_size),
        jnp.asarray(observing_player).astype(jnp.float32),
    )
    return jnp.stack([mine, theirs, player_plane], axis=-1).astype(jnp.float32)

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
    # OpenSpiel's planes 0-3 are white stones, 5-8 black stones.
    os_obs = np.asarray(os_state.observation_tensor(player)).reshape(
        9, self.board_size, self.board_size
    )
    black = os_obs[5] + os_obs[6] + os_obs[7] + os_obs[8]
    white = os_obs[0] + os_obs[1] + os_obs[2] + os_obs[3]
    mine, theirs = (black, white) if player == 0 else (white, black)
    player_plane = np.full((self.board_size, self.board_size), float(player))
    return jnp.asarray(
        np.stack([mine, theirs, player_plane], axis=-1), dtype=jnp.float32
    )


class CanonicalHex(game_lib.Game):
  """Hex transposed for white, so each player joins top and bottom.

  For white, cell (r, c) of the view is cell (c, r) of the board: actions,
  the legal mask and the observation are transposed. The swap action (with
  `swap=True`) keeps its id. Black's view is the board.

  Attributes:
    game: The wrapped Hex game.
  """

  def __init__(self, game: Hex):
    self.game = game
    n = game.board_size
    cells = np.arange(n * n)
    transpose = (cells % n) * n + cells // n
    if game.swap:  # the swap action keeps its id
      transpose = np.append(transpose, n * n)
    self._transpose = transpose.astype(np.int32)

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.game.num_actions

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return self.game.obs_shape

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return self.game.new_initial_state()

  def _to_board(self, state, action):
    """The board cell of `action` in the canonical (player 0) frame.

    Args:
      state: The state.
      action: The canonical action.

    Returns:
      i32[], the board cell.
    """
    return jnp.where(
        state.current_player == 1, jnp.asarray(self._transpose)[action], action
    ).astype(jnp.int32)

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
    return self.game.apply_action(state, self._to_board(state, action))

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return self.game.current_player(state)

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return self.game.is_terminal(state)

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return self.game.is_chance_node(state)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    mask = state.legal_actions_mask
    return jnp.where(
        state.current_player == 1, mask[jnp.asarray(self._transpose)], mask
    )

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
    obs = self.game.observation_tensor(state, observing_player)
    return jax.lax.select(
        observing_player == 1, jnp.transpose(obs, (1, 0, 2)), obs
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
    return self.game.returns(state)


def canonical_view(game: Hex) -> CanonicalHex:
  """Returns `game` transposed for white.

  Args:
    game: The game.

  Returns:
    The CanonicalHex.
  """
  return CanonicalHex(game)
