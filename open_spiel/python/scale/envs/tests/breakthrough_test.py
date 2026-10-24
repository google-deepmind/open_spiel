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

"""BreakthroughEnv tests.

The Env contract, pyspiel action ids, normalized terminal, known answers,
perfect-information world_from_info.

Actions are pyspiel's (768 on 8x8): ((row * cols + col) * 6 + dir) * 2 +
capture, row 0 at the TOP (rank 8), seat 0 (black) on rows 0-1 moving down
(dirs 0-2: down-left, down, down-right), seat 1 (white) on the bottom two
rows moving up (dirs 3-5: up-left, up, up-right).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import breakthrough
from open_spiel.python.scale.envs.tests import board_helpers
from open_spiel.python.scale.testing import check_env

A = breakthrough.action_id  # pylint: disable=invalid-name
# Seat 0 races the a-file: a7-a3, then captures b2 and a1 (reaches row 7).
P0_RACE = (
    A(1, 0, 1),
    A(6, 7, 4),
    A(2, 0, 1),
    A(5, 7, 4),
    A(3, 0, 1),
    A(4, 7, 4),
    A(4, 0, 1),
    A(3, 7, 4),
    A(5, 0, 2, 1),
    A(2, 7, 3, 1),
    A(6, 1, 0, 1),
)
# Seat 1 races the a-file up while seat 0 shuffles d/e/f pawns.
P1_RACE = (
    A(1, 3, 1),
    A(6, 0, 4),
    A(1, 4, 1),
    A(5, 0, 4),
    A(2, 3, 1),
    A(4, 0, 4),
    A(2, 4, 1),
    A(3, 0, 4),
    A(1, 5, 1),
    A(2, 0, 5, 1),
    A(0, 3, 1),
    A(1, 1, 3, 1),
)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A jitted breakthrough env.

  Returns:
    The env.
  """
  return board_helpers.JitEnv(breakthrough.BreakthroughEnv())


def test_sizes(env):
  """Sizes.

  Args:
    env: The module `env` fixture.
  """
  assert env.num_actions == 8 * 8 * 6 * 2 == 768
  assert env.num_players == 2 and env.max_chance_outcomes == 0
  # 16 pawns a side, each advancing at most to the row before the goal, +1
  # winning move a side: 2 * (8 * 6 + 8 * 5 + 1).
  assert env.max_len == 178
  assert (env.rows, env.columns) == (8, 8)


def test_action_id_known_values():
  """Action id known values."""
  # pyspiel breakthrough 8x8 action strings (open_spiel 1.6.12).
  assert A(1, 0, 1) == 98  # a7a6
  assert A(1, 0, 2) == 100  # a7b6
  assert A(1, 1, 0) == 108  # b7a6
  assert A(6, 0, 4) == 584  # a2a3
  assert A(6, 7, 4) == 668  # h2h3
  assert A(5, 0, 2, 1) == 485


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env.env, num_playouts=32)


def test_check_env_small_board():
  """Check env small board."""
  check_env.check_env(
      breakthrough.BreakthroughEnv(rows=6, columns=5), num_playouts=64
  )


def _play(env, actions):
  """Plays `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  return board_helpers.play(env, actions)


def test_initial_legal_set_is_pyspiels(env):
  """Initial legal set is pyspiels.

  Args:
    env: The module `env` fixture.
  """
  s = env.init(jax.random.key(0))
  want = [
      98,
      100,
      108,
      110,
      112,
      120,
      122,
      124,
      132,
      134,
      136,
      144,
      146,
      148,
      156,
      158,
      160,
      168,
      170,
      172,
      180,
      182,
  ]
  assert np.flatnonzero(np.asarray(env.legal_mask(s))).tolist() == want
  s = env.step(s, jnp.int32(98), jax.random.key(0))
  want1 = [
      584,
      586,
      594,
      596,
      598,
      606,
      608,
      610,
      618,
      620,
      622,
      630,
      632,
      634,
      642,
      644,
      646,
      654,
      656,
      658,
      666,
      668,
  ]
  assert np.flatnonzero(np.asarray(env.legal_mask(s))).tolist() == want1
  assert int(env.actor(s)) == 1


@pytest.mark.parametrize(
    "actions,expected", [(P0_RACE, [1.0, -1.0]), (P1_RACE, [-1.0, 1.0])]
)
def test_known_terminal_returns(env, actions, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  s = _play(env, actions)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  for k in range(len(actions)):
    st = _play(env, actions[:k])
    assert not bool(env.terminal(st))
    assert int(env.actor(st)) == k % 2
    np.testing.assert_array_equal(np.asarray(env.returns(st)), [0.0, 0.0])


def test_capture_bit_is_the_target(env):
  """Capture bit is the target.

  Before the capture a5xb2 the capture id is legal and the non-capture id
  of the same move is not; a straight move onto a piece is illegal.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, P0_RACE[:8])
  m = np.asarray(env.legal_mask(s))
  assert m[A(5, 0, 2, 1)] and not m[A(5, 0, 2, 0)]
  assert not m[A(5, 0, 1, 0)] and not m[A(5, 0, 1, 1)]


def test_info_state_board_layout(env):
  """Info state board layout.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (A(1, 0, 1),))
  for seat in (0, 1):
    i = env.info_state(s, seat)
    assert int(i.seat) == seat and int(i.to_move) == 1
    b = np.asarray(i.board).reshape(8, 8)
    assert (
        (b[0] == 0).all()
        and b[1, 0] == -1
        and (b[1, 1:] == 0).all()
        and b[2, 0] == 0
    )
    assert (b[6:] == 1).all() and (b[3:6] == -1).all()
  t = _play(env, P0_RACE)
  assert int(env.info_state(t, 0).to_move) == -1


def test_world_from_info_round_trips(env):
  """World from info round trips.

  Args:
    env: The module `env` fixture.
  """
  for actions in ((), P0_RACE[:5], P0_RACE[:9], P1_RACE[:7]):
    st = _play(env, actions)
    for seat in (0, 1):
      w = env.world_from_info(env.info_state(st, seat))
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  for actions in (P0_RACE, P1_RACE):  # terminal: the info round-trips
    st = _play(env, actions)
    info = env.info_state(st, 0)
    w = env.world_from_info(info)
    assert bool(env.terminal(w))
    np.testing.assert_array_equal(
        np.asarray(env.returns(w)), np.asarray(env.returns(st))
    )
    for x, y in zip(info, env.info_state(w, 0)):
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_bad_sizes_fail_loud():
  """Bad sizes fail loud."""
  # pyspiel gives each side ONE home row below 6 rows; the engine always two.
  with pytest.raises(ValueError, match="rows"):
    breakthrough.BreakthroughEnv(rows=5)
  with pytest.raises(ValueError, match="columns"):
    breakthrough.BreakthroughEnv(columns=1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
