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

"""ConnectFourEnv tests.

The Env contract, normalized terminal, known answers, perfect-information
world_from_info.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.testing import check_env

P0_VERTICAL = (0, 1, 0, 1, 0, 1, 0)
P1_VERTICAL = (0, 1, 0, 1, 0, 1, 6, 1)
P0_HORIZONTAL = (0, 0, 1, 1, 2, 2, 3)
DRAW = (
    3,
    3,
    0,
    2,
    5,
    4,
    5,
    6,
    4,
    2,
    2,
    3,
    3,
    5,
    5,
    2,
    2,
    4,
    3,
    3,
    1,
    1,
    2,
    1,
    5,
    5,
    0,
    4,
    0,
    4,
    4,
    6,
    6,
    6,
    6,
    0,
    1,
    0,
    6,
    0,
    1,
    1,
)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Connect Four env.

  Returns:
    The env.
  """
  return connect_four.ConnectFourEnv()


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, num_playouts=64)


def _play(env, actions):
  """Plays legal `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  s = env.init(jax.random.key(0))
  for a in actions:
    assert not bool(env.terminal(s)), actions
    assert bool(env.legal_mask(s)[a])
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


@pytest.mark.parametrize(
    "actions,expected",
    [
        (P0_VERTICAL, [1.0, -1.0]),
        (P1_VERTICAL, [-1.0, 1.0]),
        (P0_HORIZONTAL, [1.0, -1.0]),
        (DRAW, [0.0, 0.0]),
    ],
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
  assert not bool(env.terminal(_play(env, actions[:-1])))


def test_draw_fills_the_board(env):
  """Draw fills the board.

  Args:
    env: The module `env` fixture.
  """
  assert len(DRAW) == 42
  s = _play(env, DRAW)
  assert (np.asarray(env.info_state(s, 0).board) >= 0).all()


def test_full_column_is_illegal(env):
  """Full column is illegal.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (0, 0, 0, 0, 0, 0))
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [False] + [True] * 6
  )


def test_info_state_board_layout(env):
  """Info state board layout.

  Row 0 is the top (the engine's layout): the first piece in col 3 lands at 38.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (3, 3))
  b = np.asarray(env.info_state(s, 0).board)
  assert b[38] == 0 and b[31] == 1 and (np.delete(b, [31, 38]) == -1).all()
  assert int(env.info_state(s, 1).seat) == 1
  assert int(env.actor(s)) == 0


def test_world_from_info_round_trips(env):
  """World from info round trips.

  Args:
    env: The module `env` fixture.
  """
  for actions in (
      (),
      (3, 3),
      P0_VERTICAL,
      P1_VERTICAL,
      P0_HORIZONTAL,
      DRAW,
      DRAW[:20],
  ):
    st = _play(env, actions)
    for seat in (0, 1):
      w = env.world_from_info(env.info_state(st, seat))
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
