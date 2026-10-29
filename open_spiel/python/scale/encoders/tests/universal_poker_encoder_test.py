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

"""Tests for universal_poker_info_state.

A hand-computed known vector, the declared shape / dtype under x64, batch ==
per-example, jit == eager, values in [0, 1], seat dependence, and agreement
with the JAX engine's information_state_tensor on what both encode (seat, own
hole cards, board).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.encoders import universal_poker as enc_lib
from open_spiel.python.scale.envs import universal_poker as up
from open_spiel.python.scale.testing import check_env

NAME = "universal_poker_info_state"
C = up.CALL


def card(rank: str, suit: int) -> int:
  """The card id of `rank` in `suit`.

  Args:
    rank: one of "23456789TJQKA".
    suit: 0..3.

  Returns:
    4 * rank index + suit.
  """
  return 4 * "23456789TJQKA".index(rank) + suit


KK = (card("K", 0), card("K", 1))
AA = (card("A", 0), card("A", 1))
FLOP = (card("9", 2), card("J", 2), card("4", 3))


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A no-limit hold'em env.

  Returns:
    The env.
  """
  return up.NoLimitHoldemEnv()


def _raw(env, events):
  """Applies `events`, chance outcomes at chance nodes, from the root.

  Args:
    env: The env.
    events: The chance outcomes and actions.

  Returns:
    The state.
  """
  s = env.root()
  for e in events:
    s = env.apply_chance(s, e) if bool(env.is_chance(s)) else env.apply(s, e)
  return s


INV = np.float32(1.0 / 200)


def chips(*x):
  """Chip amounts scaled as the encoder scales them.

  Args:
    *x: chip amounts.

  Returns:
    f32[len(x)], times 1/200.
  """
  return np.asarray(x, np.float32) * INV


def _onehot(idx, n):
  """A multi-hot of the indices `idx`.

  Args:
    idx: The index or indices.
    n: The length.

  Returns:
    f32[n].
  """
  v = np.zeros(n, np.float32)
  for i in np.atleast_1d(idx):
    v[i] = 1.0
  return v


def test_registered_and_declared():
  """Registered and declared."""
  enc = registry.get(NAME)
  assert enc is enc_lib.INFO_STATE
  assert enc.game == "universal_poker"
  assert enc.shape == (968,) and enc.dtype == jnp.float32
  assert registry.names(game="universal_poker") == (NAME,)


def test_known_vector(env):
  # KK (seat 0, BB) v AA: SB raises to 6, BB re-raises to 18, SB calls; flop;
  # seat 0 checks, seat 1 bets to 30. Seat 0 to act.
  """Known vector."""
  s = _raw(env, list(KK) + list(AA) + [6, 18, C] + list(FLOP) + [C, 30])
  head = np.concatenate(
      [[1, 0], _onehot(KK, 52), _onehot(FLOP, 52), _onehot(1, 4), chips(18, 30)]
  )
  slots = np.zeros((up.MAX_LEN, enc_lib.SLOT), np.float32)
  for i, (kind, size, rnd) in enumerate(
      [(2, 6, 0), (2, 18, 0), (1, 0, 0), (1, 0, 1), (2, 30, 1)]
  ):
    slots[i] = np.concatenate([_onehot(kind, 3), chips(size), _onehot(rnd, 4)])
  want = np.concatenate([head, slots.reshape(-1)]).astype(np.float32)
  got = registry.get(NAME)(env.info_state(s, 0))
  np.testing.assert_array_equal(np.asarray(got), want)
  # Seat 1's vector differs only in the seat one-hot and the hole cards.
  got1 = np.asarray(registry.get(NAME)(env.info_state(s, 1)))
  diff = np.flatnonzero(got1 != want)
  assert set(diff) == {0, 1} | {2 + c for c in KK + AA}


def test_fold_slot_and_empty_history(env):
  """Fold slot and empty history."""
  s0 = _raw(env, list(KK) + list(AA))
  v = np.asarray(registry.get(NAME)(env.info_state(s0, 1)))
  assert v[112:].sum() == 0  # no action yet
  np.testing.assert_array_equal(v[110:112], chips(2, 1))  # the blinds (BB, SB)
  s = _raw(env, list(KK) + list(AA) + [up.FOLD])
  v = np.asarray(registry.get(NAME)(env.info_state(s, 0)))
  np.testing.assert_array_equal(v[112:120], [1, 0, 0, 0, 1, 0, 0, 0])


def test_agrees_with_engine_tensor_on_cards(env):
  """Agrees with engine tensor on cards."""
  s = _raw(env, list(KK) + list(AA) + [6, 18, C] + list(FLOP) + [C, 30])
  for seat in (0, 1):
    ours = np.asarray(registry.get(NAME)(env.info_state(s, seat)))
    theirs = np.asarray(env.game.information_state_tensor(s, jnp.int32(seat)))
    np.testing.assert_array_equal(ours[:106], theirs[:106])  # seat, hole, board


def test_contract_on_random_playouts(env):
  """Contract on random playouts."""
  enc = registry.get(NAME)
  kd, kp = jax.random.split(jax.random.key(5))
  # Reaches private _playout of the code under test.
  traj = jax.jit(jax.vmap(lambda d, q: check_env._playout(env, d, q)))(  # pylint: disable=protected-access
      jax.random.split(kd, 16), jax.random.split(kp, 16)
  )
  flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)
  n = jax.tree_util.tree_leaves(flat)[0].shape[0]
  outs = []
  for seat in (0, 1):
    info = jax.vmap(env.info_state)(flat, jnp.full((n,), seat, jnp.int32))
    eager = enc.batch(info)
    jitted = jax.jit(enc.batch)(info)
    assert eager.shape == (n, 968) and eager.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
    e = np.asarray(eager)
    assert np.isfinite(e).all() and (e >= 0).all() and (e <= 1).all()
    one = enc(jax.tree_util.tree_map(lambda x: x[3], info))
    np.testing.assert_array_equal(np.asarray(one), e[3])
    outs.append(e)
  assert not np.array_equal(outs[0], outs[1])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
