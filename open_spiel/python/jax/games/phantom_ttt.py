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

"""Classical phantom tic-tac-toe, OpenSpiel's `phantom_ttt`.

Phantom tic-tac-toe is imperfect-information tic-tac-toe. A referee board
holds both players' marks; each player sees only their own marks and the
cells they tried to play and found occupied. In the classical variant
(OpenSpiel's default `gameversion=classical`) a move to an occupied cell
reveals it and the same player moves again. Observation type is OpenSpiel's
default, `reveal-nothing`.

State: a dict with the referee board (0 empty, 2 x for player 0, 1 o for
player 1), each player's view and the history of attempts. Actions: the cell
index 0-8. Observation (f32[27]): three planes over the cells of the
observer's view: empty, o, x. Information state (f32[108]): the observation,
then the observer's own attempts in order as cell one-hots. Returns: +1 / -1
for three in a row, 0 for a draw.
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
NUM_CELLS = 9
NUM_CELL_STATES = 3
MAX_TOTAL_ACTION_SEQUENCE = 2 * NUM_CELLS - 1
MAX_PLAYER_ACTION_SEQUENCE = NUM_CELLS

EMPTY = 0
P0_MARK = 2
P1_MARK = 1


OBS_DIM = NUM_CELLS * NUM_CELL_STATES
INFO_STATE_DIM = OBS_DIM + MAX_PLAYER_ACTION_SEQUENCE * NUM_ACTIONS

STATE_DTYPES = {
    "board": jnp.int8,
    "views": jnp.int8,
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


def _init_state():
  """Returns the initial state; player 0 (x) moves first.

  Returns:
    The initial state dict.
  """
  return {
      "board": jnp.zeros(NUM_CELLS, dtype=jnp.int8),
      "views": jnp.zeros((NUM_PLAYERS, NUM_CELLS), dtype=jnp.int8),
      "action_sequence": jnp.full(
          MAX_TOTAL_ACTION_SEQUENCE, -1, dtype=jnp.int8
      ),
      "action_sequence_players": jnp.full(
          MAX_TOTAL_ACTION_SEQUENCE, -1, dtype=jnp.int8
      ),
      "action_sequence_len": jnp.int8(0),
      "current_player": jnp.int8(0),
      "done": jnp.bool_(False),
      "winner": jnp.int8(-1),
      "returns": jnp.array([0.0, 0.0], dtype=jnp.float32),
  }


@functools.cache
def _win_lines():
  """The eight winning lines.

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


def _check_winner(board, mark):
  """Whether `mark` holds a line of `board`.

  Args:
    board: The board.
    mark: The mark.

  Returns:
    bool[], True with a line.
  """
  lines = _win_lines()
  return jnp.any(jnp.all(board[lines] == mark, axis=1))


def _compute_returns(winner):
  """The returns of a finished game.

  Args:
    winner: The winner, or -1.

  Returns:
    f32[2], +1/-1 for the winner and loser, zeros for none.
  """
  p0_return = jnp.where(winner == 0, 1.0, jnp.where(winner == 1, -1.0, 0.0))
  return jnp.array([p0_return, -p0_return], dtype=jnp.float32)


def _step(state, action):
  """Returns the state after the player to move tries cell `action`.

  A terminal state is returned unchanged.

  Args:
    state: The state.
    action: The cell the player to move tries.

  Returns:
    The next state.
  """
  action = jnp.asarray(action, dtype=jnp.int32)
  player = state["current_player"]
  mark = jnp.where(player == 0, jnp.int8(P0_MARK), jnp.int8(P1_MARK))
  board = state["board"]
  done = state["done"]

  occupied = board[action] != EMPTY
  successful = (~done) & (~occupied)

  new_board = jnp.where(successful, board.at[action].set(mark), board)
  revealed_value = new_board[action]
  new_views = jnp.where(
      ~done,
      state["views"].at[player, action].set(revealed_value),
      state["views"],
  )

  seq_len = state["action_sequence_len"]
  append = (~done) & (seq_len < MAX_TOTAL_ACTION_SEQUENCE)
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

  player_won = successful & _check_winner(new_board, mark)
  board_full = jnp.all(new_board != EMPTY)
  draw = successful & board_full & (~player_won)
  new_done = done | player_won | draw
  new_winner = jnp.where(
      player_won,
      player,
      jnp.where(draw, jnp.int8(-1), state["winner"]),
  )
  new_returns = jnp.where(
      new_done, _compute_returns(new_winner), state["returns"]
  )

  next_player = jnp.where(
      new_done,
      player,
      jnp.where(successful, jnp.int8(1) - player, player),
  )

  return {
      "board": new_board,
      "views": new_views,
      "action_sequence": new_action_sequence,
      "action_sequence_players": new_action_sequence_players,
      "action_sequence_len": new_seq_len,
      "current_player": next_player,
      "done": new_done,
      "winner": new_winner,
      "returns": new_returns,
  }


def _legal_actions_mask(state):
  """Return legal actions for the current player's private view.

  Args:
    state: The state.

  Returns:
    bool[num_actions], the legal-action mask.
  """
  player = state["current_player"]
  mask = state["views"][player] == EMPTY
  return jnp.where(state["done"], jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), mask)


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
  return jnp.where(
      state["done"],
      jnp.int8(game_lib.TERMINAL_PLAYER_ID),
      state["current_player"],
  ).astype(jnp.int32)


def _returns(state):
  """Return terminal returns.

  Args:
    state: The state.

  Returns:
    f32[2], the returns.
  """
  return state["returns"]


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


def _observation_tensor(state, player):
  """Returns f32[27]: OpenSpiel's `reveal-nothing` observation.

  Args:
    state: The state.
    player: The observing player.

  Returns:
    f32[27], the observation.
  """
  view = state["views"][player]
  empty = (view == EMPTY).astype(jnp.float32)
  o_mark = (view == P1_MARK).astype(jnp.float32)
  x_mark = (view == P0_MARK).astype(jnp.float32)
  return jnp.concatenate([empty, o_mark, x_mark]).astype(jnp.float32)


def _information_state_tensor(state, player):
  """Returns f32[108]: OpenSpiel's `reveal-nothing` information state.

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


class PhantomTicTacToe(game_lib.Game, interop.PyspielInterop):
  """Classical phantom tic-tac-toe."""

  def __init__(
      self, gameversion: str = "classical", obstype: str = "reveal-nothing"
  ):
    """The parameters are OpenSpiel's; only their defaults are implemented."""
    if gameversion != "classical":
      raise ValueError(
          "phantom_ttt supports only gameversion='classical', got "
          f"{gameversion!r}"
      )
    if obstype != "reveal-nothing":
      raise ValueError(
          f"phantom_ttt supports only obstype='reveal-nothing', got {obstype!r}"
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
