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

"""SeatStep construction and per-ply policy keys.

The rollout drivers, serving replay, search, shard readers and the policy
checkers all build SeatSteps and derive policy keys here, so a replayed
decision sees exactly the inputs and keys the rollout gave it (row by row;
sequence.scan_evaluate shares one key across a batch).

  steps = plies.seat_steps(env, states, new_hand, new_match, last_action,
                           last_actor)
  eval_key, decide_key = plies.ply_keys(seat_key, row_ply)
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

# The init_carry key of a seat is fold_in64's key at the high word 0xFFFFFFFF
# (counter -1 as int64): a policy key folds in a non-negative int64 row ply,
# whose high word is at most 0x7FFFFFFF, so the two key domains never meet.
INIT_KEY_WORD = 0xFFFFFFFF


def seat_step(
    env: types.Env,
    states: Any,
    seat: Any,
    new_hand: Any,
    new_match: Any,
    last_action: Any,
    last_actor: Any,
) -> types.SeatStep:
  """SeatStep [B] of `seat` (i32[B] or scalar) over batched env states [B].

  Flags and last_* broadcast from scalars or [B]. last_action is what this
  seat sees (callers apply Env.visible_action). On a terminal state to_move is
  -1 and the legal mask is all False.

  Args:
    env: the env.
    states: batched env states [B].
    seat: i32[B] or a scalar seat.
    new_hand: bool flags: a hand starts at this ply.
    new_match: bool flags: a match starts at this ply.
    last_action: the last action as the seat sees it.
    last_actor: the seat that took the last action.

  Returns:
    the SeatStep [B].
  """
  actor = jax.vmap(env.actor)(states)
  legal = jax.vmap(env.legal_mask)(states)
  b = actor.shape[0]

  def full(v):
    """Broadcasts `v` to the batch.

    Args:
      v: A scalar or [b] value.

    Returns:
      [b], the broadcast value.
    """
    return jnp.broadcast_to(jnp.asarray(v), (b,))

  seat = full(seat).astype(jnp.int32)
  return types.SeatStep(
      info=jax.vmap(env.info_state)(states, seat),
      legal_mask=jnp.where((actor == seat)[:, None], legal, False),
      seat=seat,
      to_move=actor.astype(jnp.int32),
      new_hand=full(new_hand).astype(jnp.bool_),
      new_match=full(new_match).astype(jnp.bool_),
      last_action=full(last_action).astype(jnp.int32),
      last_actor=full(last_actor).astype(jnp.int32),
  )


def seat_steps(
    env: types.Env,
    states: Any,
    new_hand: Any,
    new_match: Any,
    last_action: Any,
    last_actor: Any,
) -> types.SeatStep:
  """SeatStep for every seat, leaves [B, P, ...].

  new_hand/new_match/last_actor: [B]; last_action: [B] (public) or [B, P]
  (per seat, after Env.visible_action).

  Args:
    env: the env.
    states: batched env states [B].
    new_hand: bool flags: a hand starts at this ply.
    new_match: bool flags: a match starts at this ply.
    last_action: the last action as the seat sees it.
    last_actor: the seat that took the last action.

  Returns:
    the SeatStep with leaves [B, P, ...].
  """
  p = env.num_players
  b = jnp.shape(new_hand)[0]
  last_action = jnp.asarray(last_action)
  if last_action.ndim == 1:
    last_action = jnp.broadcast_to(last_action[:, None], (b, p))
  assert last_action.shape == (b, p), last_action.shape

  def one(q, la):
    """The SeatStep of seat q.

    Args:
      q: the seat.
      la: its last_action.

    Returns:
      the SeatStep [B].
    """
    return seat_step(
        env,
        states,
        jnp.full((b,), q, jnp.int32),
        new_hand,
        new_match,
        la,
        last_actor,
    )

  per_seat = jax.vmap(one, in_axes=(0, 1), out_axes=1)(
      jnp.arange(p, dtype=jnp.int32), last_action
  )
  return per_seat


def visible_actions(env: types.Env, states: Any, action: Any) -> jax.Array:
  """Env.visible_action(s, a, seat) for every seat, i32[B, P].

  Args:
    env: the env.
    states: batched env states [B].
    action: i32[B], the action taken.

  Returns:
    i32[B, P].
  """
  seats = jnp.arange(env.num_players, dtype=jnp.int32)
  return jax.vmap(
      lambda s, a: jax.vmap(lambda q: env.visible_action(s, a, q))(seats)
  )(states, action).astype(jnp.int32)


def policy_key(seat_key: Any, row_ply: Any) -> Any:
  """The policy key of one seat at one row ply (row_ply is an int64 counter).

  Args:
    seat_key: the seat's policy key.
    row_ply: the row's int64 ply counter.

  Returns:
    the key.
  """
  return x64.fold_in64(seat_key, row_ply)


def ply_keys(seat_key: Any, row_ply: Any):
  """(evaluate key, decide key) of one seat at one row ply.

  Args:
    seat_key: the seat's policy key.
    row_ply: the row's int64 ply counter.

  Returns:
    (evaluate key, decide key).
  """
  pk = policy_key(seat_key, row_ply)
  return jax.random.fold_in(pk, 1), jax.random.fold_in(pk, 2)


def init_key(seat_key: Any) -> Any:
  """The init_carry key of one seat; no row ply's policy key equals it.

  Args:
    seat_key: the seat's policy key.

  Returns:
    the key.
  """
  low = jax.random.fold_in(seat_key, INIT_KEY_WORD)
  return jax.random.fold_in(low, INIT_KEY_WORD)
