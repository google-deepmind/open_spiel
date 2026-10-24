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

"""The chance resolver: samples chance nodes for Env.init, Env.step, Arena.

Chance event n, counting from `count` (an int64), is sampled with
core.x64.fold_in64(base_key, n), so an outcome depends only on the key and
the event's index, never on how many player plies came before it. The Arena
keys a hand's events by the hand and carries the count across plies, which
makes duplicate deals share every chance outcome of a hand (the private deal
and every later board card) whatever the seats play.

  state, count = chance.resolve(env, state, hand_key, count)
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64


def sample_outcome(env: types.Env, s: Any, key: Any) -> jax.Array:
  """An i32 outcome drawn from env.chance_probs(s).

  Args:
    env: the env.
    s: a chance node.
    key: the draw's key.

  Returns:
    the outcome.
  """
  probs = env.chance_probs(s)
  probs = jnp.asarray(probs, jnp.float32)
  logp = jnp.where(
      probs > 0,
      jnp.log(jnp.maximum(probs, jnp.float32(1e-30))),
      jnp.float32(-jnp.inf),
  )
  return jax.random.categorical(key, logp).astype(jnp.int32)


def resolve(env: types.Env, s: Any, base_key: Any, count: Any = 0):
  """Samples chance nodes until a decision or terminal node.

  Returns (state, count + number of chance events resolved), count int64.

  Args:
    env: the env.
    s: the state.
    base_key: the hand's chance key; event n is keyed by fold_in64(base_key, n).
    count: the int64 count of chance events already resolved.

  Returns:
    (state, count + number of chance events resolved), count int64.
  """
  count = jnp.asarray(count)
  assert jnp.issubdtype(count.dtype, jnp.integer), count.dtype
  count = count.astype(jnp.int64)
  if env.max_chance_outcomes == 0:
    return s, count

  def cond(carry):
    """Whether the state is a non-terminal chance node.

    Args:
      carry: (state, count).

    Returns:
      bool.
    """
    st, _ = carry
    return env.is_chance(st) & ~env.terminal(st)

  def body(carry):
    """Samples and applies one chance event.

    Args:
      carry: (state, count).

    Returns:
      (next state, count + 1).
    """
    st, n = carry
    o = sample_outcome(env, st, x64.fold_in64(base_key, n))
    return env.apply_chance(st, o), n + jnp.int64(1)

  return jax.lax.while_loop(cond, body, (s, count))
