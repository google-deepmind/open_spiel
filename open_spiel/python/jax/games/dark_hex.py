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

"""Classical dark hex on a 3x3 board, OpenSpiel's `dark_hex(board_size=3)`.

Dark hex is imperfect-information hex. A referee board holds both players'
stones; each player sees only the cells they have played and the occupied
cells they tried to play. In the classical variant (`gameversion=cdh`) a
rejected move does not pass the turn: the player moves again until they
choose a cell that is empty on the referee board. Observation type is
OpenSpiel's default, `reveal-nothing`.

State: a dict with the referee board (0 empty, 1 black, 2 white), each
player's view of the cells, and the history of attempted moves. Black
(player 0) joins the top and bottom rows; white joins the left and right
columns. Actions: the cell index row * 3 + col, as in OpenSpiel.
Observation (f32[81]): per cell a one-hot over the nine view states of
OpenSpiel's encoding. Information state (f32[162]): the observation, then
the observer's own attempts in order as cell one-hots. Returns: +1 / -1;
hex has no draws.

Supports only board_size=3.
"""

import chex
import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

BOARD_SIZE = 3
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_ACTIONS = NUM_CELLS
NUM_PLAYERS = 2
MAX_GAME_LENGTH = NUM_CELLS * NUM_CELLS

EMPTY = 0
BLACK = 1
WHITE = 2
BLACK_VIEW = 1
WHITE_VIEW = -1
BLACK_WIN_VIEW = 4
WHITE_WIN_VIEW = -4
NUM_CELL_STATES = 9
MIN_CELL_STATE = -4


OBS_DIM = NUM_CELLS * NUM_CELL_STATES
INFO_STATE_DIM = OBS_DIM + NUM_CELLS * NUM_CELLS

STATE_DTYPES = {
    "board": jnp.int8,
    "views": jnp.int8,
    "current_player": jnp.int8,
    "done": jnp.bool_,
    "winner": jnp.int8,
    "move_count": jnp.int8,
    "action_sequence": jnp.int8,
    "action_players": jnp.int8,
    "history_length": jnp.int8,
    "returns": jnp.float32,
}


def assert_state_dtypes(state):
  """Raises if a field of `state` has a dtype other than STATE_DTYPES.

  A check for tests.

  Args:
    state: The state dict.
  """
  for field, expected_dtype in STATE_DTYPES.items():
    if field in state:
      chex.assert_type(state[field], expected_dtype)


def _get_edges():
  """Returns the bool[3, 3] masks of the top, bottom, left and right edges.

  Returns:
    (top, bottom, left, right), each bool[3, 3].
  """
  top_edge = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_).at[0, :].set(True)
  )
  bottom_edge = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
      .at[BOARD_SIZE - 1, :]
      .set(True)
  )
  left_edge = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_).at[:, 0].set(True)
  )
  right_edge = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
      .at[:, BOARD_SIZE - 1]
      .set(True)
  )
  return top_edge, bottom_edge, left_edge, right_edge


def _init_state():
  """Returns the initial state; black (player 0) moves first.

  Returns:
    The initial state dict.
  """
  return {
      "board": jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int8),
      "views": jnp.zeros((NUM_PLAYERS, NUM_CELLS), dtype=jnp.int8),
      "current_player": jnp.int8(0),
      "done": jnp.bool_(False),
      "winner": jnp.int8(-1),
      "move_count": jnp.int8(0),
      "action_sequence": jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int8),
      "action_players": jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int8),
      "history_length": jnp.int8(0),
      "returns": jnp.array([0.0, 0.0], dtype=jnp.float32),
  }


def _expand_neighbors(reached):
  """Expand a mask to adjacent hex cells.

  Args:
    reached: bool[3, 3], the mask.

  Returns:
    bool[3, 3], the cells adjacent to the mask.
  """
  p = jnp.pad(reached, 1, constant_values=False)
  size = BOARD_SIZE
  return (
      p[0:size, 1 : size + 1]
      | p[0:size, 2 : size + 2]
      | p[1 : size + 1, 0:size]
      | p[1 : size + 1, 2 : size + 2]
      | p[2 : size + 2, 0:size]
      | p[2 : size + 2, 1 : size + 1]
  )


def _check_connected(board, mark, start_edge, end_edge):
  """Whether mark's stones connect start_edge to end_edge.

  Args:
    board: The board, int8[BOARD_SIZE, BOARD_SIZE].
    mark: The stone value to trace.
    start_edge: The bool mask of the edge the chain starts on.
    end_edge: The bool mask of the edge the chain must reach.

  Returns:
    A bool scalar: whether a chain of mark joins the two edges.
  """
  cells = board == mark
  reached = cells & start_edge

  def body_fn(_, carry):
    """Grows the connected set by one neighbour step.

    Args:
      _: The iteration; unused.
      carry: The connected set.

    Returns:
      The grown set.
    """
    return carry | (_expand_neighbors(carry) & cells)

  reached = jax.lax.fori_loop(0, NUM_CELLS, body_fn, reached)
  return jnp.any(reached & end_edge)


def _check_winner(board):
  """The winner of `board`.

  Args:
    board: int8[3, 3], the board.

  Returns:
    int8[], 0 or 1 for the winner, -1 for none yet.
  """
  top_edge, bottom_edge, left_edge, right_edge = _get_edges()
  black_wins = _check_connected(board, jnp.int8(BLACK), top_edge, bottom_edge)
  white_wins = _check_connected(board, jnp.int8(WHITE), left_edge, right_edge)
  return jnp.where(
      black_wins,
      jnp.int8(0),
      jnp.where(white_wins, jnp.int8(1), jnp.int8(-1)),
  )


def _raise_history_overflow(history_length):
  raise RuntimeError(
      "Dark Hex action history overflow: "
      f"history_length={history_length}, capacity={MAX_GAME_LENGTH}"
  )


def _assert_history_capacity(history_length):
  """Raises if a history write would fall off the end of the buffer.

  `.at[i].set` silently drops out-of-range writes, so an eager call checks
  the bound here. Under tracing the check is skipped: a game ends long
  before MAX_GAME_LENGTH attempts, and `step` leaves a terminal state
  unchanged.

  Args:
    history_length: The number of entries written so far.
  """
  if isinstance(history_length, jax.core.Tracer):
    return
  if int(history_length) >= MAX_GAME_LENGTH:
    _raise_history_overflow(int(history_length))


def _active_step(state, action):
  """The state after the player to move attempts action on a live game.

  A move onto an occupied cell fails: the board is unchanged, the player
  keeps the move and learns the occupant's colour.

  Args:
    state: The game state dict.
    action: The cell index attempted.

  Returns:
    The new game state dict.
  """
  player = state["current_player"]
  _assert_history_capacity(state["history_length"])
  row = action // BOARD_SIZE
  col = action % BOARD_SIZE
  mark = player + jnp.int8(1)

  current_cell = state["board"][row, col]
  successful = current_cell == EMPTY

  placed_board = state["board"].at[row, col].set(mark)
  new_board = jnp.where(successful, placed_board, state["board"])

  winner = _check_winner(new_board)
  black_reveal = jnp.where(winner == 0, BLACK_WIN_VIEW, BLACK_VIEW)
  white_reveal = jnp.where(winner == 1, WHITE_WIN_VIEW, WHITE_VIEW)
  placed_reveal = jnp.where(player == 0, black_reveal, white_reveal)
  occupied_reveal = jnp.where(
      current_cell == BLACK, jnp.int8(BLACK_VIEW), jnp.int8(WHITE_VIEW)
  )
  revealed = jnp.where(successful, placed_reveal, occupied_reveal)
  new_views = state["views"].at[player, action].set(revealed)

  new_done = winner >= 0
  new_winner = winner

  next_player = jnp.where(successful, jnp.int8(1) - player, player)
  new_player = jnp.where(
      new_done, jnp.int8(game_lib.TERMINAL_PLAYER_ID), next_player
  )
  new_move_count = jnp.where(
      successful, state["move_count"] + jnp.int8(1), state["move_count"]
  )

  history_idx = state["history_length"]
  new_action_sequence = state["action_sequence"].at[history_idx].set(action)
  new_action_players = state["action_players"].at[history_idx].set(player)
  new_history_length = state["history_length"] + jnp.int8(1)

  p0_return = jnp.where(
      new_winner == 0, 1.0, jnp.where(new_winner == 1, -1.0, 0.0)
  )
  new_returns = jnp.array([p0_return, -p0_return], dtype=jnp.float32)

  return {
      "board": new_board,
      "views": new_views,
      "current_player": new_player,
      "done": new_done,
      "winner": new_winner,
      "move_count": new_move_count,
      "action_sequence": new_action_sequence,
      "action_players": new_action_players,
      "history_length": new_history_length,
      "returns": jnp.where(new_done, new_returns, state["returns"]),
  }


def _step(state, action):
  """Returns the state after the player to move tries cell `action`.

  If the cell is occupied on the referee board, the player's view records
  the occupant and the same player moves again. Otherwise the stone is
  placed and the turn passes, unless the move wins. A terminal state is
  returned unchanged.

  Args:
    state: The state.
    action: The cell the player to move tries.

  Returns:
    The next state.
  """
  action = jnp.asarray(action, dtype=jnp.int8)
  return jax.lax.cond(
      state["done"],
      lambda _: state,
      lambda _: _active_step(state, action),
      operand=None,
  )


def _legal_actions_mask(state):
  """Return legal actions for the current player based on their private view.

  Args:
    state: The state.

  Returns:
    bool[num_actions], the legal-action mask.
  """
  player = state["current_player"]
  safe_player = jnp.clip(player, 0, NUM_PLAYERS - 1)
  mask = state["views"][safe_player] == EMPTY
  return jnp.where(state["done"], jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), mask)


def _is_chance_node(state):
  """Always False: dark hex has no chance nodes.

  Args:
    state: The state.

  Returns:
    bool[], False.
  """
  del state
  return jnp.bool_(False)


def _is_terminal(state):
  """Return whether the game is terminal.

  Args:
    state: The state.

  Returns:
    bool[], whether the game is over.
  """
  return state["done"]


def _current_player(state):
  """Returns i32[]: the player to move, or TERMINAL_PLAYER_ID when terminal.

  Args:
    state: The state.

  Returns:
    i32[], the player to move.
  """
  return state["current_player"].astype(jnp.int32)


def _returns(state):
  """Return terminal utilities, or zeros before terminal.

  Args:
    state: The state.

  Returns:
    f32[2], the returns.
  """
  return state["returns"]


def _observation_tensor(state, player):
  """Returns f32[81]: OpenSpiel's `reveal-nothing` observation.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[81], the observation.
  """
  view = state["views"][player]
  indices = view - jnp.int8(MIN_CELL_STATE)
  return (
      jax.nn.one_hot(indices, NUM_CELL_STATES).reshape(-1).astype(jnp.float32)
  )


def _information_state_tensor(state, player):
  """Returns f32[162]: OpenSpiel's `reveal-nothing` information state.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[162], the information state.
  """
  view_features = _observation_tensor(state, player)

  def body_fn(carry, idx):
    """Encodes one history entry if `player` saw it.

    Args:
      carry: (features, next slot).
      idx: The history index.

    Returns:
      ((features, next slot), None) for `jax.lax.scan`.
    """
    features, offset = carry
    is_visible = (
        (state["action_players"][idx] == player)
        & (idx < state["history_length"])
        & (offset < NUM_CELLS)
    )
    action = jnp.maximum(state["action_sequence"][idx], 0)
    updated = features.at[offset].set(jax.nn.one_hot(action, NUM_CELLS))
    features = jnp.where(is_visible, updated, features)
    offset = offset + jnp.where(is_visible, jnp.int8(1), jnp.int8(0))
    return (features, offset), None

  init_features = jnp.zeros((NUM_CELLS, NUM_CELLS), dtype=jnp.float32)
  (seq_features, _), _ = jax.lax.scan(
      body_fn,
      (init_features, jnp.int8(0)),
      jnp.arange(MAX_GAME_LENGTH, dtype=jnp.int8),
  )
  return jnp.concatenate([view_features, seq_features.reshape(-1)])


def _chance_outcomes(state):
  """Return the empty (0,) float32 chance vector; there are no chance nodes.

  Args:
    state: The state.

  Returns:
    f32[0], the empty chance vector.
  """
  del state
  return jnp.zeros((0,), dtype=jnp.float32)


class DarkHex(game_lib.Game, interop.PyspielInterop):
  """Classical dark hex on a 3x3 board."""

  def __init__(self, board_size: int = BOARD_SIZE):
    if board_size != BOARD_SIZE:
      raise ValueError(
          f"dark_hex supports only board_size={BOARD_SIZE}, got {board_size}"
      )

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return NUM_ACTIONS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (OBS_DIM,)

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    return (INFO_STATE_DIM,)

  @property
  def num_players(self):
    """The number of players."""
    return NUM_PLAYERS

  def new_initial_state(self):
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return _init_state()

  def apply_action(self, state, action):
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    return _step(state, action)

  def current_player(self, state):
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return _current_player(state)

  def is_terminal(self, state):
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return _is_terminal(state)

  def is_chance_node(self, state):
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return _is_chance_node(state)

  def chance_outcomes(self, state):
    """Returns the chance distribution at `state`.

    Index i is the probability of chance action id i. At a chance node the
    entries are non-negative and sum to 1; outcomes that cannot occur have
    probability 0. Elsewhere the values are unspecified. The shape is static,
    so the method is safe under `jit` and `vmap`. To enumerate outcomes
    exactly, iterate over the nonzero entries; to simulate, sample an index.

    Games without chance nodes inherit this default, which returns `f32[0]`.

    Args:
      state: The current state.

    Returns:
      f32[max_chance_outcomes]: the chance distribution.

    Raises:
      NotImplementedError: if the game declares chance outcomes but does not
        override this method.
    """
    return _chance_outcomes(state)

  def legal_actions_mask(self, state):
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    return _legal_actions_mask(state)

  def observation_tensor(self, state, observing_player):
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    return _observation_tensor(state, observing_player)

  def information_state_tensor(self, state, observing_player):
    """Returns everything `observing_player` has observed so far.

    In a perfect-information game the current observation determines the
    information state, and the game returns its observation tensor.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*info_state_shape]: the information-state tensor.
    """
    return _information_state_tensor(state, observing_player)

  def returns(self, state):
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    return _returns(state)

  def observation_from_os(self, os_state, player):
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    return jnp.asarray(os_state.observation_tensor(player), dtype=jnp.float32)

  def information_state_from_os(self, os_state, player):
    """Returns the JAX game's information-state tensor of `player`.

    Games whose information state is their observation inherit this
    default, which returns `observation_from_os`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The information-state tensor the JAX game gives `player` in the same
      position.
    """
    return jnp.asarray(
        os_state.information_state_tensor(player), dtype=jnp.float32
    )
