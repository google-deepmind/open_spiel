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

"""Abrupt phantom tic-tac-toe, imperfect recall.

The Env over the JAX engine `phantom_ttt_ir` (OpenSpiel
`phantom_ttt_ir(gameversion=abrupt)`).

Abrupt rules: a seat sees only its own marks and the squares its own failed
attempts revealed; a move onto a square the opponent holds fails, reveals that
square to the mover and PASSES the turn (the classical phantom_ttt lets the
mover retry). Moves are private (Env.visible_action hides every move from the
other seat).

Imperfect recall: OpenSpiel's phantom_ttt_ir information state is its
observation, so the InfoState is the seat and its view only, never the
seat's attempt history. Stateful policies are refused
(Env.admits_stateful_policies False): a carry fed SeatStep.last_action at
every ply would restore the memory of past attempts that imperfect recall
removes.

Conventions normalized here: views use envs.phantom_ttt's marks (0
unseen/empty, 2 X = seat 0, 1 O = seat 1; the ir engine stores 1 X,
2 O), so both phantom variants' views mean the same; the engine's state is a
dict with `done` (no `is_terminal`) and current_player() is -4 at terminal,
normalized to actor -1 by the adapter.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import phantom_ttt_ir as engine
from open_spiel.python.scale.envs import hidden_cells
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.envs import phantom_ttt

NUM_CELLS = phantom_ttt.NUM_CELLS
# 17: a seat's 9th attempt fills the board
MAX_LEN = engine.MAX_TOTAL_ACTION_SEQUENCE
EMPTY = phantom_ttt.EMPTY
P0_MARK = phantom_ttt.P0_MARK  # 2 (X)
P1_MARK = phantom_ttt.P1_MARK  # 1 (O)
assert engine.EMPTY == EMPTY


def to_phantom_marks(cells: jax.Array) -> jax.Array:
  """i32 cells in envs.phantom_ttt's marks from the ir engine's marks.

  Args:
    cells: i32 cells in the ir engine's marks.

  Returns:
    i32 cells in envs.phantom_ttt's marks.
  """
  c = jnp.asarray(cells, jnp.int32)
  return jnp.where(
      c == engine.PLAYER0_MARK,
      jnp.int32(P0_MARK),
      jnp.where(c == engine.PLAYER1_MARK, jnp.int32(P1_MARK), jnp.int32(EMPTY)),
  )


class PhantomTTTIRInfo(NamedTuple):
  """What one seat knows under imperfect recall: its seat and its view.

  Attributes:
    seat: i32.
    view: i32[9]: 0 unseen/empty, 2 X, 1 O (own marks + revealed).
  """

  seat: jax.Array
  view: jax.Array


class PhantomTTTIREnv(jax_game_env.JaxGameEnv):
  """Abrupt phantom tic-tac-toe (imperfect recall): 2 seats, 9 actions."""

  name = "phantom_ttt_ir"
  max_len = MAX_LEN
  zero_sum = True
  admits_stateful_policies = False  # module doc

  def __init__(self):
    super().__init__(engine.PhantomTicTacToeIR(gameversion="abrupt"))
    assert self.max_chance_outcomes == 0
    assert self.num_players == 2 and self.num_actions == NUM_CELLS

  def terminal(self, s) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s["done"], jnp.bool_)

  def info_state(self, s, seat) -> PhantomTTTIRInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return PhantomTTTIRInfo(
        seat=seat, view=to_phantom_marks(jnp.asarray(s["visible"])[seat])
    )

  def labels(self, s, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    seat = jnp.asarray(seat, jnp.int32)
    opp_mark = jnp.where(
        seat == 0,
        jnp.int32(engine.PLAYER1_MARK),
        jnp.int32(engine.PLAYER0_MARK),
    )
    return {"opp_marks": jnp.asarray(s["board"], jnp.int32) == opp_mark}

  def visible_action(self, s, a, seat) -> jax.Array:
    """What `seat` learns of action `a` taken at state `s`.

    Args:
      s: The state.
      a: The player action id.
      seat: The seat (i32).

    Returns:
      i32 action as `seat` sees it.
    """
    return hidden_cells.private_visible_action(self, s, a, seat)


def hide_opponent_marks(env: PhantomTTTIREnv, s, seat, key):
  """Privacy hook for check_env: envs.phantom_ttt's rewrite.

  (a random permutation of the cells `seat` has not seen, moving the board, the
  opponent's view and attempt log); may reach no playable state, and
  info_state(seat) must be blind to it.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The rewritten state.
  """
  del env
  return phantom_ttt.permute_unseen(s, seat, key, "visible")
