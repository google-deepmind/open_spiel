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

"""Dark hex on a 3x3 board with imperfect recall, OpenSpiel's `dark_hex_ir`.

Both of OpenSpiel's variants are implemented, with OpenSpiel's defaults
(`board_size=3`, `gameversion=cdh`, `obstype=reveal-nothing`). A move to a
cell that is occupied on the referee board reveals the occupant to the mover
without placing a stone; in classical dark hex (`cdh`) the same player then
moves again, in abrupt dark hex (`adh`) the turn passes. Each player sees
only the cells they have played or revealed.

State: a dict with the referee board (0 empty, 1 black, 2 white), each
player's view and the history of attempts. Black (player 0) joins the top
and bottom rows; white joins the left and right columns. Actions: the cell
index row * 3 + col. Observation (f32[81]): per cell a one-hot over OpenSpiel's
nine cell states. Information state (f32[162], imperfect recall): the
observation, then the observer's own attempts in order as cell one-hots.
Returns: +1 / -1; hex has no draws.

Supports only board_size=3 (num_rows = num_cols = 3) and
obstype=reveal-nothing.
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
MAX_ATTEMPTS = NUM_CELLS * 2 - 1

EMPTY = 0
BLACK = 1
WHITE = 2

OPEN_SPIEL_EMPTY_INDEX = 4
OPEN_SPIEL_WHITE_INDEX = 3
OPEN_SPIEL_BLACK_INDEX = 5
OPEN_SPIEL_CELL_STATES = 9


STATE_DTYPES = {
    "board": jnp.int8,
    "p0_view": jnp.int8,
    "p1_view": jnp.int8,
    "current_player": jnp.int8,
    "done": jnp.bool_,
    "winner": jnp.int8,
    "winning_cell": jnp.int8,
    "attempt_count": jnp.int8,
    "action_sequence_players": jnp.int8,
    "action_sequence_actions": jnp.int8,
    "returns": jnp.float32,
}

OBS_DIM = NUM_CELLS * OPEN_SPIEL_CELL_STATES
INFO_STATE_DIM = OBS_DIM + NUM_CELLS * NUM_CELLS


def assert_state_dtypes(state):
  """Raises if a field of `state` has a dtype other than STATE_DTYPES.

  A check for tests.

  Args:
    state: The state dict.
  """
  for field, expected_dtype in STATE_DTYPES.items():
    if field in state:
      chex.assert_type(state[field], expected_dtype)


def _init_state():
  """Initialize a new 3x3 Abrupt Dark Hex state.

  Returns:
    The initial state dict.
  """
  return {
      "board": jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int8),
      "p0_view": jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int8),
      "p1_view": jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int8),
      "current_player": jnp.int8(0),
      "done": jnp.bool_(False),
      "winner": jnp.int8(-1),
      "winning_cell": jnp.int8(-1),
      "attempt_count": jnp.int8(0),
      "action_sequence_players": jnp.full((MAX_ATTEMPTS,), -1, dtype=jnp.int8),
      "action_sequence_actions": jnp.full((MAX_ATTEMPTS,), -1, dtype=jnp.int8),
      "returns": jnp.array([0.0, 0.0], dtype=jnp.float32),
  }


def _expand_neighbors(reached):
  """Expands a mask to its adjacent hex cells.

  Args:
    reached: bool[3, 3], the mask.

  Returns:
    bool[3, 3], the cells adjacent to the mask.
  """
  padded = jnp.pad(reached, 1, constant_values=False)
  return (
      padded[0:BOARD_SIZE, 1 : BOARD_SIZE + 1]
      | padded[0:BOARD_SIZE, 2 : BOARD_SIZE + 2]
      | padded[1 : BOARD_SIZE + 1, 0:BOARD_SIZE]
      | padded[1 : BOARD_SIZE + 1, 2 : BOARD_SIZE + 2]
      | padded[2 : BOARD_SIZE + 2, 0:BOARD_SIZE]
      | padded[2 : BOARD_SIZE + 2, 1 : BOARD_SIZE + 1]
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

  def body_fn(_, current):
    """Grows the connected set by one neighbour step.

    Args:
      _: The iteration; unused.
      current: The connected set.

    Returns:
      The grown set.
    """
    return current | (_expand_neighbors(current) & cells)

  reached = jax.lax.fori_loop(0, NUM_CELLS, body_fn, reached)
  return jnp.any(reached & end_edge)


def _check_winner(board):
  """The winner of the board.

  Black wins by joining top to bottom, white by joining left to right.

  Args:
    board: The board, int8[BOARD_SIZE, BOARD_SIZE].

  Returns:
    An int8 scalar: 0 if black has won, 1 if white has, else -1.
  """
  top = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_).at[0, :].set(True)
  bottom = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
      .at[BOARD_SIZE - 1, :]
      .set(True)
  )
  left = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_).at[:, 0].set(True)
  right = (
      jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
      .at[:, BOARD_SIZE - 1]
      .set(True)
  )

  p0_wins = _check_connected(board, jnp.int8(BLACK), top, bottom)
  p1_wins = _check_connected(board, jnp.int8(WHITE), left, right)
  return jnp.where(
      p0_wins, jnp.int8(0), jnp.where(p1_wins, jnp.int8(1), jnp.int8(-1))
  )


def _view_for_player(state, player):
  """The private view of `player`.

  Args:
    state: The state.
    player: The player.

  Returns:
    int8[3, 3], the view.
  """
  return jnp.where(player == 0, state["p0_view"], state["p1_view"])


def _set_view_for_player(state, player, new_view):
  """Replaces the private view of `player`.

  Args:
    state: The state.
    player: The player.
    new_view: The new view.

  Returns:
    The state with the new view.
  """
  return {
      **state,
      "p0_view": jnp.where(player == 0, new_view, state["p0_view"]),
      "p1_view": jnp.where(player == 1, new_view, state["p1_view"]),
  }


def _compute_returns(winner):
  """The returns of a finished game.

  Args:
    winner: The winner, or -1.

  Returns:
    f32[2], +1/-1 for the winner and loser, zeros for none.
  """
  p0_return = jnp.where(winner == 0, 1.0, jnp.where(winner == 1, -1.0, 0.0))
  return jnp.array([p0_return, -p0_return], dtype=jnp.float32)


GAME_VERSIONS = ("cdh", "adh")


def _step(state, action, abrupt=False):
  """Returns the state after the player to move tries cell `action`.

  An occupied cell is revealed in the mover's view without a stone being
  placed; with `abrupt` (a Python bool, `gameversion=adh`) the turn then
  passes, otherwise (`cdh`) the mover moves again. A terminal state is
  returned unchanged.

  Args:
    state: The state.
    action: The cell the player to move tries.
    abrupt: Whether a move to an occupied cell passes the turn.

  Returns:
    The next state.
  """
  action = jnp.asarray(action, dtype=jnp.int32)
  player = state["current_player"]
  done = state["done"]
  row = action // BOARD_SIZE
  col = action % BOARD_SIZE
  mark = (player + jnp.int8(1)).astype(jnp.int8)
  occupied_mark = state["board"][row, col]
  is_empty = occupied_mark == EMPTY
  active = ~done

  placed_board = state["board"].at[row, col].set(mark)
  new_board = jnp.where(active & is_empty, placed_board, state["board"])
  visible_mark = jnp.where(is_empty, mark, occupied_mark)
  player_view = _view_for_player(state, player)
  new_view = jnp.where(
      active, player_view.at[row, col].set(visible_mark), player_view
  )

  new_state = _set_view_for_player(state, player, new_view)

  sequence_index = state["attempt_count"]
  can_record = active & (sequence_index < MAX_ATTEMPTS)
  new_sequence_players = jnp.where(
      can_record,
      state["action_sequence_players"].at[sequence_index].set(player),
      state["action_sequence_players"],
  )
  new_sequence_actions = jnp.where(
      can_record,
      state["action_sequence_actions"]
      .at[sequence_index]
      .set(jnp.asarray(action).astype(jnp.int8)),
      state["action_sequence_actions"],
  )
  new_attempt_count = jnp.where(
      active,
      jnp.minimum(sequence_index + jnp.int8(1), jnp.int8(MAX_ATTEMPTS)),
      sequence_index,
  )

  winner = _check_winner(new_board)
  new_done = done | ((winner >= 0) & active)
  new_winner = jnp.where(done, state["winner"], winner)
  new_winning_cell = jnp.where(
      new_done & ~done,
      jnp.asarray(action).astype(jnp.int8),
      state["winning_cell"],
  )
  turn_passes = is_empty | abrupt
  next_player = jnp.where(
      new_done,
      jnp.int8(game_lib.TERMINAL_PLAYER_ID),
      jnp.where(turn_passes, 1 - player, player),
  )
  new_returns = jnp.where(
      new_done, _compute_returns(new_winner), state["returns"]
  )

  return {
      **new_state,
      "board": new_board,
      "current_player": next_player,
      "done": new_done,
      "winner": new_winner,
      "winning_cell": new_winning_cell,
      "attempt_count": new_attempt_count,
      "action_sequence_players": new_sequence_players,
      "action_sequence_actions": new_sequence_actions,
      "returns": new_returns,
  }


def _legal_actions_mask(state):
  """Return actions whose cells are empty in the acting player's view.

  Args:
    state: The state.

  Returns:
    bool[num_actions], the legal-action mask.
  """
  player = state["current_player"]
  view = _view_for_player(state, player)
  mask = (view.reshape((NUM_CELLS,)) == EMPTY).astype(jnp.bool_)
  return jnp.where(state["done"], jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), mask)


def _chance_outcomes(state):
  """Return the empty (0,) float32 chance vector; there are no chance nodes.

  Args:
    state: The state.

  Returns:
    f32[0], the empty chance vector.
  """
  del state
  return jnp.zeros((0,), dtype=jnp.float32)


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
  """Return terminal returns.

  Args:
    state: The state.

  Returns:
    f32[2], the returns.
  """
  return state["returns"]


def _open_spiel_cell_index(state, view):
  """The OpenSpiel cell-state index of each cell of `view`.

  Args:
    state: The state.
    view: int8[NUM_CELLS], a player's view.

  Returns:
    int[NUM_CELLS], each cell's index (winning cells get theirs).
  """
  flat_cells = jnp.arange(NUM_CELLS, dtype=jnp.int8)
  base_index = jnp.where(
      view == BLACK,
      OPEN_SPIEL_BLACK_INDEX,
      jnp.where(view == WHITE, OPEN_SPIEL_WHITE_INDEX, OPEN_SPIEL_EMPTY_INDEX),
  )
  is_winning_cell = state["done"] & (flat_cells == state["winning_cell"])
  black_win = is_winning_cell & (state["winner"] == 0) & (view == BLACK)
  white_win = is_winning_cell & (state["winner"] == 1) & (view == WHITE)
  return jnp.where(black_win, 8, jnp.where(white_win, 0, base_index))


def _observation_tensor(state, player):
  """Returns f32[81]: the player's view in OpenSpiel's cell encoding.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[81], the observation.
  """
  view = _view_for_player(state, player).reshape((NUM_CELLS,))
  indices = _open_spiel_cell_index(state, view)
  return (
      jax.nn.one_hot(indices, OPEN_SPIEL_CELL_STATES)
      .reshape(-1)
      .astype(jnp.float32)
  )


def _information_state_tensor(state, player):
  """Returns f32[162]: the observation and the player's own attempts.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[162], the information state.
  """
  obs = _observation_tensor(state, player)
  actions = state["action_sequence_actions"]
  players = state["action_sequence_players"]
  own_action_mask = (players == player) & (actions >= 0)
  own_action_ranks = jnp.cumsum(own_action_mask.astype(jnp.int8)) - jnp.int8(1)

  def encode_slot(slot):
    """One-hot encodes the player's `slot`-th own action.

    Args:
      slot: The slot.

    Returns:
      f32[NUM_CELLS]; zeros for an unused slot.
    """
    matching = own_action_mask & (own_action_ranks == slot)
    action = jnp.sum(jnp.where(matching, actions, 0))
    valid = jnp.any(matching) & (slot < state["attempt_count"])
    encoded = jax.nn.one_hot(jnp.maximum(action, 0), NUM_CELLS)
    return jnp.where(valid, encoded, jnp.zeros((NUM_CELLS,), dtype=jnp.float32))

  sequence = jax.vmap(encode_slot)(jnp.arange(NUM_CELLS))
  return jnp.concatenate([obs, sequence.reshape(-1)]).astype(jnp.float32)


class DarkHexIR(game_lib.Game, interop.PyspielInterop):
  """Dark hex with imperfect recall on a 3x3 board, classical by default.

  Attributes:
    abrupt: Whether a move to an occupied cell passes the turn.
    gameversion: The OpenSpiel gameversion, "cdh" or "adh".
  """

  def __init__(
      self,
      board_size: int = BOARD_SIZE,
      gameversion: str = "cdh",
      obstype: str = "reveal-nothing",
      num_rows: int | None = None,
      num_cols: int | None = None,
  ):
    for name, value in (
        ("board_size", board_size),
        ("num_rows", num_rows),
        ("num_cols", num_cols),
    ):
      if value is not None and value != BOARD_SIZE:
        raise ValueError(
            f"dark_hex_ir supports only {name}={BOARD_SIZE}, got {value}"
        )
    if gameversion not in GAME_VERSIONS:
      raise ValueError(
          f"dark_hex_ir gameversion must be one of {GAME_VERSIONS}, got "
          f"{gameversion!r}"
      )
    if obstype != "reveal-nothing":
      raise ValueError(
          f"dark_hex_ir supports only obstype='reveal-nothing', got {obstype!r}"
      )
    self.gameversion = gameversion
    self.abrupt = gameversion == "adh"

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
    return _step(state, action, self.abrupt)

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
