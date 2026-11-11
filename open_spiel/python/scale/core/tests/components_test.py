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

"""Contract tests of the small core components, alone.

chance.resolve, plies (SeatStep construction and keys), sequence (the scan
defaults).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import chance
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.testing import fixtures


def test_resolve_keys_event_n_by_fold_in_and_counts_events():
  """Resolve keys chance event n by fold_in and counts the events."""
  env = kuhn.KuhnEnv()
  key = jax.random.key(3)
  s, n = chance.resolve(env, env.root(), key)
  assert int(n) == 2 and not bool(env.is_chance(s))
  # Known answer: replay with the documented keys.
  r = env.root()
  for i in range(2):
    r = env.apply_chance(
        r, chance.sample_outcome(env, r, jax.random.fold_in(key, i))
    )
  for x, y in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(r)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  # Starting the count at 5 uses events 5, 6.
  s5, n5 = chance.resolve(env, env.root(), key, 5)
  assert int(n5) == 7
  r = env.root()
  for i in (5, 6):
    r = env.apply_chance(
        r, chance.sample_outcome(env, r, jax.random.fold_in(key, i))
    )
  np.testing.assert_array_equal(np.asarray(s5.p0_card), np.asarray(r.p0_card))
  # A decision node is left alone.
  s2, n2 = chance.resolve(env, s, key, 9)
  assert int(n2) == 9 and int(s2.p0_card) == int(s.p0_card)


def test_resolve_deterministic_and_jittable():
  """Resolve is deterministic and jittable."""
  env = kuhn.KuhnEnv()
  keys = jax.random.split(jax.random.key(0), 64)
  f = jax.jit(jax.vmap(lambda k: chance.resolve(env, env.root(), k)))
  a, b = f(keys), f(keys)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_seat_steps_shapes_masks_and_privacy():
  """seat_steps' shapes, legal masks and privacy."""
  env = kuhn.KuhnEnv()
  s = jax.vmap(env.init)(jax.random.split(jax.random.key(1), 4))
  la = jnp.array([[0, -1], [1, 1], [-1, -1], [0, 0]], jnp.int32)
  st = plies.seat_steps(
      env,
      s,
      jnp.ones(4, bool),
      jnp.zeros(4, bool),
      la,
      jnp.full((4,), types.NO_ACTOR, jnp.int32),
  )
  assert st.legal_mask.shape == (4, 2, 2) and st.info.card.shape == (4, 2)
  np.testing.assert_array_equal(np.asarray(st.seat), [[0, 1]] * 4)
  np.testing.assert_array_equal(np.asarray(st.last_action), np.asarray(la))
  # Seat 0 acts first in kuhn: only its mask is non-empty.
  assert (
      np.asarray(st.legal_mask)[:, 0].all()
      and not np.asarray(st.legal_mask)[:, 1].any()
  )
  np.testing.assert_array_equal(
      np.asarray(st.info.card[:, 0]), np.asarray(s.p0_card)
  )
  np.testing.assert_array_equal(
      np.asarray(st.info.card[:, 1]), np.asarray(s.p1_card)
  )
  # Terminal: to_move -1, nothing legal.
  t = jax.vmap(
      lambda x: env.step(
          env.step(x, 1, jax.random.key(0)), 1, jax.random.key(0)
      )
  )(s)
  tst = plies.seat_steps(
      env,
      t,
      jnp.zeros(4, bool),
      jnp.zeros(4, bool),
      jnp.ones(4, jnp.int32),
      jnp.ones(4, jnp.int32),
  )
  assert (np.asarray(tst.to_move) == -1).all() and not np.asarray(
      tst.legal_mask
  ).any()


def test_ply_keys_are_distinct_and_reproducible():
  """ply_keys are distinct and reproducible."""
  k = jax.random.key(0)
  e1, d1 = plies.ply_keys(k, 3)
  e2, d2 = plies.ply_keys(k, 3)
  e3, _ = plies.ply_keys(k, 4)

  def raw(x):
    """The key data of `x`.

    Args:
      x: A typed key.

    Returns:
      The numpy key data.
    """
    return np.asarray(jax.random.key_data(x))

  np.testing.assert_array_equal(raw(e1), raw(e2))
  np.testing.assert_array_equal(raw(d1), raw(d2))
  assert not np.array_equal(raw(e1), raw(d1)) and not np.array_equal(
      raw(e1), raw(e3)
  )
  for ply in (0, 2**31 - 1, 2**32 - 1, 2**63 - 1):
    assert not np.array_equal(
        raw(plies.init_key(k)), raw(plies.policy_key(k, ply))
    ), ply


def test_scan_evaluate_equals_per_ply_loop():
  """scan_evaluate equals a per-ply loop."""
  env = kuhn.KuhnEnv()
  pol = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  params = pol.make_params(w=jnp.array([[0.3, -0.2], [0.1, 0.4]]))
  s = jax.vmap(env.init)(jax.random.split(jax.random.key(2), 3))
  st0 = plies.seat_step(env, s, 0, True, True, -1, -1)
  s1 = jax.vmap(lambda x: env.step(x, 1, jax.random.key(0)))(s)
  st1 = plies.seat_step(env, s1, 0, False, False, 1, 0)
  seq = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b], 1), st0, st1)
  c0 = pol.init_carry(params, 3, st0.seat, None)
  c_end, outs = pol.evaluate_seq(params, c0, seq, None)
  c = c0
  for t, st in enumerate((st0, st1)):
    c = pol.observe(params, c, st)
    np.testing.assert_allclose(
        np.asarray(outs.logits[:, t]),
        np.asarray(pol.evaluate(params, c, st, None).logits),
    )
  np.testing.assert_array_equal(np.asarray(c_end["count"]), [2.0, 2.0, 2.0])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
