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

"""Tests for Encoders.

Hand-computed known answers, and agreement of the info-state
encoders with the JAX engine's (pyspiel-equivalent) information_state_tensor.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.envs import tic_tac_toe
from open_spiel.python.scale.testing import check_env

F, C, R = 0, 1, 2
J0, J1, Q0, Q1, K0, K1 = range(6)


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


def _play(env, actions):
  """Plays `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  s = env.init(jax.random.key(0))
  for a in actions:
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


def _onehot(i, n):
  """A one-hot of `i` (none for -1).

  Args:
    i: The index, or -1.
    n: The length.

  Returns:
    f32[n].
  """
  v = np.zeros(n, np.float32)
  if i >= 0:
    v[i] = 1.0
  return v


def _check(name, info, expected):
  enc = registry.get(name)
  out = enc(info)
  assert out.dtype == enc.dtype
  np.testing.assert_array_equal(
      np.asarray(out, np.float32),
      np.asarray(expected, np.float32).reshape(enc.shape),
  )


@pytest.fixture(name="envs", scope="module")
def _envs_fixture():
  """The envs with known answers.

  Returns:
    {name: env}.
  """
  return dict(
      leduc=leduc.LeducEnv(),
      ttt=tic_tac_toe.TicTacToeEnv(),
      c4=connect_four.ConnectFourEnv(),
      pttt=phantom_ttt.PhantomTTTEnv(),
  )


def test_leduc_known_vectors(envs):
  """Leduc known vectors."""
  env = envs["leduc"]
  s = _raw(env, (Q0, K1, R, C, J0, C))  # round 2, public J0, p1 to act
  hdr0 = np.concatenate([[1, 0], _onehot(Q0, 6), _onehot(J0, 6)])
  hdr1 = np.concatenate([[0, 1], _onehot(K1, 6), _onehot(J0, 6)])
  r1 = [0, 1, 1, 0, 0, 0, 0, 0]  # raise, call
  r2 = [1, 0, 0, 0, 0, 0, 0, 0]  # call
  _check(
      "leduc_poker_info_state",
      env.info_state(s, 0),
      np.concatenate([hdr0, r1, r2]),
  )
  _check(
      "leduc_poker_info_state",
      env.info_state(s, 1),
      np.concatenate([hdr1, r1, r2]),
  )
  _check(
      "leduc_poker_observation_tensor",
      env.info_state(s, 0),
      np.concatenate([hdr0, [3, 3]]),
  )
  s = _raw(env, (Q0, K1, R))  # round 1, p1 facing a raise: no public card
  hdr1 = np.concatenate([[0, 1], _onehot(K1, 6), np.zeros(6)])
  _check(
      "leduc_poker_observation_tensor",
      env.info_state(s, 1),
      np.concatenate([hdr1, [3, 1]]),
  )
  _check(
      "leduc_poker_info_state",
      env.info_state(s, 1),
      np.concatenate([hdr1, [0, 1] + [0] * 6, [0] * 8]),
  )


def test_tic_tac_toe_known_vectors(envs):
  """Tic tac toe known vectors."""
  env = envs["ttt"]
  s = _play(env, (4, 0, 8))  # X at 4, 8; O at 0
  mine = np.zeros(9)
  mine[0] = 1
  theirs = np.zeros(9)
  theirs[[4, 8]] = 1
  _check(
      "tic_tac_toe_planes", env.info_state(s, 1), np.stack([mine, theirs], -1)
  )
  _check(
      "tic_tac_toe_observation_tensor",
      env.info_state(s, 1),
      np.stack([mine, theirs, np.ones(9)], -1),
  )
  _check(
      "tic_tac_toe_observation_tensor",
      env.info_state(s, 0),
      np.stack([theirs, mine, np.zeros(9)], -1),
  )


def test_connect_four_known_vectors(envs):
  """Connect four known vectors."""
  env = envs["c4"]
  s = _play(env, (3, 3))  # seat 0 at cell 38, seat 1 at cell 31 (row 0 = top)
  board = -np.ones(42)
  board[38] = 0
  board[31] = 1
  m0 = np.zeros(42)
  m0[38] = 1
  m1 = np.zeros(42)
  m1[31] = 1
  empty = (board == -1).astype(np.float32)
  _check(
      "connect_four_planes",
      env.info_state(s, 1),
      np.stack([m1, m0], -1).reshape(6, 7, 2),
  )
  _check(
      "connect_four_player_planes",
      env.info_state(s, 0),
      np.stack([m0, m1, empty], -1).reshape(6, 7, 3),
  )
  _check(
      "connect_four_player_planes",
      env.info_state(s, 1),
      np.stack([m1, m0, empty], -1).reshape(6, 7, 3),
  )
  _check(
      "connect_four_flat_planes",
      env.info_state(s, 0),
      np.concatenate([empty, m0, m1]),
  )


def test_phantom_ttt_known_vectors(envs):
  """Phantom ttt known vectors."""
  env = envs["pttt"]
  s = _play(env, (4, 4, 0))  # p1 failed at 4 (sees X), then placed O at 0
  view = np.array([1, 0, 0, 0, 2, 0, 0, 0, 0])
  planes = np.concatenate([view == 0, view == 1, view == 2]).astype(np.float32)
  attempts = np.zeros((9, 9))
  attempts[0, 4] = 1
  attempts[1, 0] = 1
  _check(
      "phantom_ttt_info_state",
      env.info_state(s, 1),
      np.concatenate([planes, attempts.reshape(-1), [0, 1]]),
  )
  _check(
      "phantom_ttt_observation",
      env.info_state(s, 1),
      np.concatenate([planes, [0, 1]]),
  )
  view0 = np.array([0, 0, 0, 0, 2, 0, 0, 0, 0])
  planes0 = np.concatenate([view0 == 0, view0 == 1, view0 == 2]).astype(
      np.float32
  )
  _check(
      "phantom_ttt_observation",
      env.info_state(s, 0),
      np.concatenate([planes0, [1, 0]]),
  )


def _random_states(env, n, seed):
  """Every state of n uniform-random hands, flattened, with a terminal flag.

  Args:
    env: The env.
    n: The hands.
    seed: The seed.

  Returns:
    (states, terminal flags).
  """
  # Reaches private _playout of the code under test.
  # pylint: disable=protected-access
  kd, kp = jax.random.split(jax.random.key(seed))
  traj = jax.jit(jax.vmap(lambda d, q: check_env._playout(env, d, q)))(
      jax.random.split(kd, n), jax.random.split(kp, n)
  )
  flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)
  return flat, np.asarray(jax.vmap(env.terminal)(flat))


@pytest.mark.parametrize("which", ["leduc", "pttt"])
def test_info_state_equals_engine_information_state_tensor(envs, which):
  """Info state equals engine information state tensor.

  leduc_poker_info_state == the engine's information_state_tensor (30);
  phantom_ttt_info_state[:108] == the engine's information_state_tensor (108).

  Args:
    envs: the envs fixture.
    which: the env's key in `envs`.
  """
  env = envs[which]
  name = {"leduc": "leduc_poker_info_state", "pttt": "phantom_ttt_info_state"}[
      which
  ]
  width = {"leduc": 30, "pttt": 108}[which]
  enc = registry.get(name)
  flat, _ = _random_states(env, 256, seed=11)
  for seat in (0, 1):
    seats = jnp.full(
        (jax.tree_util.tree_leaves(flat)[0].shape[0],), seat, jnp.int32
    )
    ours = jax.jit(lambda s, q: enc.batch(jax.vmap(env.info_state)(s, q)))(
        flat, seats
    )
    ref = jax.jit(jax.vmap(env.game.information_state_tensor))(flat, seats)
    assert ref.dtype == jnp.float32 and ref.shape[1] == width
    np.testing.assert_array_equal(np.asarray(ours[:, :width]), np.asarray(ref))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
