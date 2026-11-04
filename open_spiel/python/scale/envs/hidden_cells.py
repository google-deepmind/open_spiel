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

"""Shared helpers of the board games whose opponent moves are hidden.

(phantom tic-tac-toe, dark hex): what a seat sees of an action, a seat's own
action log, and the privacy rewrite that permutes the cells a seat has not seen.

  own, n = hidden_cells.own_actions(actions, players, length, seat, n_own)
  s = hidden_cells.permute_unseen(s, seat, key, views="views",
  actions="action_sequence",
                                  players="action_sequence_players")
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types


def private_visible_action(env, s, a, seat) -> jax.Array:
  """A seat sees its own moves; the other seat's moves are hidden.

  Args:
    env: The env.
    s: The state.
    a: The player action id.
    seat: The seat.

  Returns:
    i32 action as `seat` sees it (NO_ACTION when hidden).
  """
  own = jnp.asarray(seat, jnp.int32) == env.actor(s)
  return jnp.where(own, jnp.asarray(a, jnp.int32), jnp.int32(types.NO_ACTION))


def unseen_permutation(unseen, key) -> jax.Array:
  """A uniformly random permutation of the unseen cells.

  i32[N]: a uniformly random permutation of the cells where `unseen`
  (bool[N]), identity elsewhere (perm[c] is where cell c's content goes).

  Args:
    unseen: bool[N] the cells to permute.
    key: PRNG key.

  Returns:
    i32[N] perm, identity elsewhere (perm[c] is where cell c's content goes).
  """
  n = unseen.shape[-1]
  cells = jnp.arange(n, dtype=jnp.int32)
  r = jax.random.uniform(key, (n,))
  # unseen first
  shuffled = jnp.argsort(jnp.where(unseen, r, 2.0 + cells)).astype(jnp.int32)
  ordered = jnp.argsort(jnp.where(unseen, cells, n + cells)).astype(jnp.int32)
  k = jnp.sum(unseen, dtype=jnp.int32)
  first_k = jnp.arange(n) < k
  return cells.at[jnp.where(first_k, ordered, n)].set(shuffled, mode="drop")


def move_cells(x, perm) -> jax.Array:
  """Moves the cells of x by perm: new[perm[c]] = x[c].

  new[perm[c]] = x[c] over the cells (x of any shape with perm.size cells).

  Args:
    x: An array of any shape with perm.size cells.
    perm: i32[N] the permutation.

  Returns:
    The moved array.
  """
  flat = x.reshape(perm.shape[0])
  return jnp.zeros_like(flat).at[perm].set(flat).reshape(x.shape)


def move_actions(actions, players, opp, perm) -> jax.Array:
  """An action log with `opp`'s cell actions moved by `perm`.

  (others as they are).

  Args:
    actions: The action log.
    players: The seat of each action.
    opp: The opponent seat.
    perm: i32[N] the permutation.

  Returns:
    The moved log (other actions as they are).
  """
  a = jnp.asarray(actions, jnp.int32)
  is_opp = (jnp.asarray(players, jnp.int32) == opp) & (a >= 0)
  return jnp.where(is_opp, perm[jnp.maximum(a, 0)], a).astype(actions.dtype)


def own_actions(
    actions, players, length, seat, n_own: int
) -> tuple[jax.Array, jax.Array]:
  """`seat`'s actions among the first `length` of the log.

  (i32[n_own]: `seat`'s actions among the first `length` of the log, in
  order and -1 padded; i32: their count).

  Args:
    actions: The action log.
    players: The seat of each action.
    length: The log length.
    seat: The seat.
    n_own: The output length.

  Returns:
    (i32[n_own] the actions in order, -1 padded; i32 their count).
  """
  i32 = jnp.int32
  seat = jnp.asarray(seat, i32)
  idx = jnp.arange(actions.shape[0], dtype=i32)
  mine = (idx < jnp.asarray(length, i32)) & (jnp.asarray(players, i32) == seat)
  # out of range: dropped
  pos = jnp.where(mine, jnp.cumsum(mine, dtype=i32) - 1, i32(n_own))
  own = (
      jnp.full((n_own,), -1, i32)
      .at[pos]
      .set(jnp.asarray(actions, i32), mode="drop")
  )
  return own, jnp.sum(mine, dtype=i32)


def permute_unseen(
    s: Any, seat, key, *, views: str, actions: str, players: str, empty: int = 0
) -> dict[str, Any]:
  """The privacy rewrite of a dict state with a per-seat view array.

  S[views] (i32[2, cells], `empty` where unseen): a random permutation of the
  cells `seat` has not seen, applied to s["board"], the opponent's view and the
  opponent's entries of the action log s[actions] (players s[players]).

  Args:
    s: The dict state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.
    views: The key of the view array.
    actions: The key of the action log.
    players: The key of the action log's seats.
    empty: The view value of an unseen cell.

  Returns:
    The rewritten state.
  """
  seat = jnp.asarray(seat, jnp.int32)
  perm = unseen_permutation(
      jnp.asarray(s[views], jnp.int32)[seat] == empty, key
  )
  opp = 1 - seat
  out = dict(s)
  out["board"] = move_cells(s["board"], perm)
  out[views] = s[views].at[opp].set(move_cells(s[views][opp], perm))
  out[actions] = move_actions(s[actions], s[players], opp, perm)
  return out
