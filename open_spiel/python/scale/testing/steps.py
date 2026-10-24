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

"""SeatStep builders for contract tests, from any Env (via core.plies)."""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types

seat_step = plies.seat_step


def two_ply_steps(env: types.Env, batch: int, seat: int, seed: int = 0):
  """Returns (step1, step2) for check_policy.

  step1 opens a match (new_hand, new_match True); step2 follows one random
  legal action by the opening actor (flags False).

  Args:
    env: The env.
    batch: The number of rows.
    seat: The observing seat.
    seed: The seed.

  Returns:
    (step1, step2): the opening ply, then the ply after one action.
  """
  key = jax.random.key(seed)
  kd, ka, kc = jax.random.split(key, 3)
  s0 = jax.vmap(env.init)(jax.random.split(kd, batch))
  step1 = seat_step(env, s0, seat, True, True, types.NO_ACTION, types.NO_ACTOR)
  legal = jax.vmap(env.legal_mask)(s0)
  logits = jnp.where(legal, jnp.float32(0.0), jnp.float32(-jnp.inf))
  a = jax.random.categorical(ka, logits, axis=-1).astype(jnp.int32)
  actor0 = jax.vmap(env.actor)(s0)
  s1 = jax.vmap(env.step)(s0, a, jax.random.split(kc, batch))
  seen = jax.vmap(env.visible_action, in_axes=(0, 0, None))(s0, a, seat)
  step2 = seat_step(env, s1, seat, False, False, seen, actor0)
  return step1, step2
