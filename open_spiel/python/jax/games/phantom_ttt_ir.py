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

"""Phantom tic-tac-toe with imperfect recall, OpenSpiel's `phantom_ttt_ir`.

Both of OpenSpiel's variants are implemented, with OpenSpiel's default
`gameversion=classical`; the observation type is OpenSpiel's default,
`reveal-nothing`.

Each player sees only their own marks and the cells they tried to play and
found occupied. A move to an occupied cell reveals it; in the classical
variant the same player then moves again, in the abrupt variant
(`gameversion=abrupt`) the turn passes. A player may target a cell hidden to
them but not one they already know is occupied.

State: a dict with the referee board (0 empty, 1 x for player 0, 2 o for
player 1), each player's visible board and the history of attempts.
Actions: the cell index 0-8. Observation (f32[27]): three planes over the
observer's visible board: empty, o, x. Information state (f32[108],
imperfect recall): the observation, then the observer's own attempts in
order as cell one-hots. Returns: +1 / -1 for three in a row, 0 for a draw.
"""

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_ACTIONS = 9
NUM_PLAYERS = 2
BOARD_CELLS = 9
OBS_DIM = 27
MAX_TOTAL_ACTION_SEQUENCE = 2 * BOARD_CELLS - 1
MAX_PLAYER_ACTION_SEQUENCE = BOARD_CELLS
INFO_STATE_DIM = OBS_DIM + MAX_PLAYER_ACTION_SEQUENCE * NUM_ACTIONS

EMPTY = 0
PLAYER0_MARK = 1
PLAYER1_MARK = 2
NO_WINNER = -1


STATE_DTYPES = {
    "board": jnp.int8,
    "visible": jnp.int8,
    "action_sequence": jnp.int8,
    "action_sequence_players": jnp.int8,
    "action_sequence_len": jnp.int8,
    "current_player": jnp.int8,
    "done": jnp.bool_,
    "winner": jnp.int8,
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


@functools.cache
def _win_lines():
  """Return Tic-Tac-Toe winning triples.

  Returns:
    int32[8, 3], each line's cells.
  """
  return np.array(
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


def _init_state():
  """Returns the initial state; player 0 (x) moves first.

  Returns:
    The initial state dict.
  """
  return {
      "board": jnp.zeros(BOARD_CELLS, dtype=jnp.int8),
      "visible": jnp.zeros((NUM_PLAYERS, BOARD_CELLS), dtype=jnp.int8),
      "action_sequence": jnp.full(
          MAX_TOTAL_ACTION_SEQUENCE, -1, dtype=jnp.int8
      ),
      "action_sequence_players": jnp.full(
          MAX_TOTAL_ACTION_SEQUENCE, -1, dtype=jnp.int8
      ),
      "action_sequence_len": jnp.int8(0),
      "current_player": jnp.int8(0),
      "done": jnp.bool_(False),
      "winner": jnp.int8(NO_WINNER),
      "returns": jnp.array([0.0, 0.0], dtype=jnp.float32),
  }


def _mark_for_player(player):
  """The board mark of `player`.

  Args:
    player: The player.

  Returns:
    int8[], the mark.
  """
  return jnp.int8(player + 1)


def _check_winner(board, player_mark):
  """Check whether the given mark has any three-in-a-row.

  Args:
    board: The board.
    player_mark: The mark.

  Returns:
    bool[], True with a line.
  """

  def check_line(line):
    """Returns whether `player_mark` holds every cell of `line`.

    Args:
      line: The line's cells.

    Returns:
      bool[].
    """
    return jnp.all(board[line] == player_mark)

  return jnp.any(jax.vmap(check_line)(_win_lines()))


def _compute_returns(winner):
  """The returns of a finished game.

  Args:
    winner: The winner, or -1.

  Returns:
    f32[2], +1/-1 for the winner and loser, zeros for none.
  """
  p0_return = jnp.where(
      winner == 0, jnp.float32(1.0), jnp.where(winner == 1, -1.0, 0.0)
  )
  return jnp.array([p0_return, -p0_return], dtype=jnp.float32)


GAME_VERSIONS = ("classical", "abrupt")


def _step(state, action, abrupt=False):
  """Returns the state after the player to move tries cell `action`.

  `action` must be legal. A terminal state is returned unchanged. `abrupt`
  (a Python bool) selects the abrupt variant: a move to an occupied cell
  passes the turn; in the classical variant the mover moves again.

  Args:
    state: The state.
    action: The cell the player to move tries.
    abrupt: Whether a move to an occupied cell passes the turn.

  Returns:
    The next state.
  """
  return jax.lax.cond(
      state["done"],
      lambda _: state,
      lambda _: _active_step(state, action, abrupt),
      operand=None,
  )


def _active_step(state, action, abrupt):
  """Returns the state after the player to move tries cell `action`.

  The non-terminal branch of `_step`.

  Args:
    state: The state; not terminal.
    action: The cell the player to move tries.
    abrupt: Whether a move to an occupied cell passes the turn.

  Returns:
    The next state.
  """
  board = state["board"]
  visible = state["visible"]
  player = state["current_player"]
  action = jnp.asarray(action, dtype=jnp.int32)

  mark = _mark_for_player(player)
  opponent_mark = _mark_for_player(1 - player)
  cell = board[action]

  hits_empty_cell = cell == EMPTY
  hits_opponent_cell = cell == opponent_mark

  new_board = jnp.where(hits_empty_cell, board.at[action].set(mark), board)

  player_visible = visible[player]
  new_player_visible = jnp.where(
      hits_empty_cell | hits_opponent_cell,
      player_visible.at[action].set(jnp.where(hits_empty_cell, mark, cell)),
      player_visible,
  )
  new_visible = visible.at[player].set(new_player_visible)

  seq_len = state["action_sequence_len"]
  append = seq_len < MAX_TOTAL_ACTION_SEQUENCE
  new_action_sequence = jnp.where(
      append,
      state["action_sequence"]
      .at[seq_len]
      .set(jnp.asarray(action).astype(jnp.int8)),
      state["action_sequence"],
  )
  new_action_sequence_players = jnp.where(
      append,
      state["action_sequence_players"].at[seq_len].set(player),
      state["action_sequence_players"],
  )
  new_seq_len = jnp.where(append, seq_len + jnp.int8(1), seq_len)

  player_won = hits_empty_cell & _check_winner(new_board, mark)
  board_full = jnp.all(new_board != EMPTY)
  draw = board_full & ~player_won
  new_done = player_won | draw
  new_winner = jnp.where(
      player_won, player, jnp.where(draw, jnp.int8(NO_WINNER), state["winner"])
  )
  new_returns = jnp.where(
      new_done, _compute_returns(new_winner), state["returns"]
  )
  turn_passes = hits_empty_cell | abrupt
  next_player = jnp.where(new_done | ~turn_passes, player, jnp.int8(1 - player))

  return {
      "board": new_board,
      "visible": new_visible,
      "action_sequence": new_action_sequence,
      "action_sequence_players": new_action_sequence_players,
      "action_sequence_len": new_seq_len,
      "current_player": next_player,
      "done": new_done,
      "winner": new_winner,
      "returns": new_returns,
  }


def _legal_actions_mask(state):
  """Return actions that are empty from ``state["current_player"]``'s view.

  Args:
    state: The state.

  Returns:
    bool[num_actions], the legal-action mask.
  """
  player = state["current_player"]
  visible_board = state["visible"][player]
  mask = visible_board == EMPTY
  return jnp.where(state["done"], jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), mask)


def _is_terminal(state):
  """Check whether the game is over.

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
  return jnp.where(
      state["done"],
      jnp.int8(game_lib.TERMINAL_PLAYER_ID),
      state["current_player"],
  ).astype(jnp.int32)


def _returns(state):
  """Returns f32[2]: zero before the end.

  Args:
    state: The state.

  Returns:
    f32[2], the returns.
  """
  return state["returns"]


def _observation_tensor(state, player):
  """Returns f32[27]: the observer's visible board as empty, o, x planes.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[27], the observation.
  """
  visible_board = state["visible"][player]
  empty = visible_board == EMPTY
  o_cells = visible_board == PLAYER1_MARK
  x_cells = visible_board == PLAYER0_MARK
  return jnp.concatenate([empty, o_cells, x_cells]).astype(jnp.float32)


def _information_state_tensor(state, player):
  """Returns f32[108]: the observation and the player's own attempts.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[108], the information state.
  """
  obs = _observation_tensor(state, player)

  indices = jnp.arange(MAX_TOTAL_ACTION_SEQUENCE)
  visible = (indices < state["action_sequence_len"]) & (
      state["action_sequence_players"] == player
  )
  player_action_positions = jnp.cumsum(visible.astype(jnp.int32)) - 1
  actions = jnp.maximum(state["action_sequence"], 0).astype(jnp.int32)
  updates = jax.nn.one_hot(actions, NUM_ACTIONS, dtype=jnp.float32)
  updates = updates * visible[:, None].astype(jnp.float32)
  clipped_positions = jnp.clip(
      player_action_positions, 0, MAX_PLAYER_ACTION_SEQUENCE - 1
  )
  encoded_actions = jnp.zeros(
      (MAX_PLAYER_ACTION_SEQUENCE, NUM_ACTIONS), dtype=jnp.float32
  )
  encoded_actions = encoded_actions.at[clipped_positions].add(updates)
  return jnp.concatenate([obs, encoded_actions.reshape(-1)]).astype(jnp.float32)


def _is_chance_node(state):
  """Always False: the game has no chance nodes.

  Args:
    state: The state.

  Returns:
    bool[], False.
  """
  del state
  return jnp.bool_(False)


def _chance_outcomes(state):
  """Return the empty (0,) float32 chance vector; there are no chance nodes.

  Args:
    state: The state.

  Returns:
    f32[0], the empty chance vector.
  """
  del state
  return jnp.zeros((0,), dtype=jnp.float32)


class PhantomTicTacToeIR(game_lib.Game, interop.PyspielInterop):
  """Phantom tic-tac-toe with imperfect recall, classical by default.

  Attributes:
    abrupt: Whether a move to an occupied cell passes the turn.
    gameversion: The OpenSpiel gameversion, "classical" or "abrupt".
  """

  def __init__(
      self, gameversion: str = "classical", obstype: str = "reveal-nothing"
  ):
    if gameversion not in GAME_VERSIONS:
      raise ValueError(
          f"phantom_ttt_ir gameversion must be one of {GAME_VERSIONS}, got "
          f"{gameversion!r}"
      )
    if obstype != "reveal-nothing":
      raise ValueError(
          "phantom_ttt_ir supports only obstype='reveal-nothing', got "
          f"{obstype!r}"
      )
    self.gameversion = gameversion
    self.abrupt = gameversion == "abrupt"

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
