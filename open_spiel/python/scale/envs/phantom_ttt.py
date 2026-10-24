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

"""Phantom tic-tac-toe (classical) Env over the JAX engine `phantom_ttt`.

Classical rules (OpenSpiel `phantom_ttt`, reveal-nothing): a seat sees only its
own marks and the squares its own failed attempts revealed. A move onto a
square the opponent holds fails, reveals that square to the mover, and the
mover moves again. Moves are private: Env.visible_action hides every move from
the other seat. Stateful policies are refused (Env.admits_stateful_policies
False; envs.dark_hex's module doc gives the example).

The engine's phantom_ttt state is a dict with a `done` flag (no `is_terminal`)
and current_player returns -4 at terminal; the
wrapper reads `done` and the adapter normalizes actor() to -1. Marks follow
the engine's: 0 empty, 2 X (seat 0), 1 O (seat 1).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import phantom_ttt
from open_spiel.python.scale.envs import hidden_cells
from open_spiel.python.scale.envs import jax_game_env

NUM_CELLS = 9
MAX_LEN = phantom_ttt.MAX_TOTAL_ACTION_SEQUENCE  # 17 decisions at most
MAX_OWN_ACTIONS = phantom_ttt.MAX_PLAYER_ACTION_SEQUENCE  # 9
EMPTY = phantom_ttt.EMPTY
P0_MARK = phantom_ttt.P0_MARK
P1_MARK = phantom_ttt.P1_MARK


class PhantomTTTInfo(NamedTuple):
  """What one seat knows: its own view and its own attempts, in order.

  Attributes:
    seat: i32.
    view: i32[9]: 0 unknown/empty, 2 X, 1 O (own marks + revealed).
    own_actions: i32[9]: this seat's attempted cells in order, -1 pad.
    num_own_actions: i32.
  """

  seat: jax.Array
  view: jax.Array
  own_actions: jax.Array
  num_own_actions: jax.Array


def own_action_sequence(s: Any, seat) -> tuple[jax.Array, jax.Array]:
  """The seat's attempts.

  (i32[9] seat's attempts in order with -1 padding, i32 count).

  Args:
    s: The state.
    seat: The seat.

  Returns:
    (i32[9] attempts in order, -1 padded; i32 count).
  """
  return hidden_cells.own_actions(
      s["action_sequence"],
      s["action_sequence_players"],
      s["action_sequence_len"],
      seat,
      MAX_OWN_ACTIONS,
  )


class PhantomTTTEnv(jax_game_env.JaxGameEnv):
  """Classical phantom tic-tac-toe: 2 seats, 9 actions."""

  name = "phantom_ttt"
  max_len = MAX_LEN
  zero_sum = True
  # A failed attempt keeps the turn, so a carry updated at every ply could
  # count the opponent's private retries: stateful policies are refused.
  admits_stateful_policies = False

  def __init__(self):
    super().__init__(phantom_ttt.PhantomTicTacToe())
    assert self.max_chance_outcomes == 0

  def terminal(self, s) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s["done"], jnp.bool_)

  def info_state(self, s, seat) -> PhantomTTTInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    own, n = own_action_sequence(s, seat)
    return PhantomTTTInfo(
        seat=seat,
        view=jnp.asarray(s["views"], jnp.int32)[seat],
        own_actions=own,
        num_own_actions=n,
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
    opp_mark = jnp.where(seat == 0, jnp.int32(P1_MARK), jnp.int32(P0_MARK))
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


def hide_opponent_marks(env: PhantomTTTEnv, s, seat, key):
  """Privacy hook for check_env: permute every cell `seat` has not seen.

  Privacy hook for check_env: permute, at random, every cell `seat` has not
  seen (its view is empty there), moving the opponent's hidden marks and the
  opponent's own view and attempt log with them. The result may be a state
  play cannot reach (e.g. an opponent line with done False); info_state(seat)
  must be blind to it all the same.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The rewritten state.
  """
  del env
  return permute_unseen(s, seat, key, "views")


def permute_unseen(s, seat, key, views_key: str):
  """The privacy rewrite of both phantom tic-tac-toe variants.

  A random permutation of the cells `seat` has not seen (s[views_key][seat] is
  EMPTY there) applied to the board, the opponent's view and the opponent's
  attempt log. Both JAX engines use 0 for an unseen cell.

  Args:
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.
    views_key: The key of the view array.

  Returns:
    The rewritten state.
  """
  return hidden_cells.permute_unseen(
      s,
      seat,
      key,
      views=views_key,
      actions="action_sequence",
      players="action_sequence_players",
      empty=EMPTY,
  )
