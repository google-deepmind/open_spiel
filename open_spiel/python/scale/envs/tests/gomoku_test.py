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

"""GomokuEnv tests.

The Env contract, normalized terminal, known answers (five in a row, an
overline, a diagonal, a full-board draw on 3x3 connect 3), perfect- information
world_from_info. Cells are pyspiel's: row * size + col, black (seat 0) first.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.envs import gomoku
from open_spiel.python.scale.envs.tests import board_helpers
from open_spiel.python.scale.testing import check_env

N = 15
P0_ROW = (0, 100, 1, 101, 2, 102, 3, 103, 4)  # black a1..e1
# Black 0,1,2,3,5 then 4: six in a row (pyspiel 1.6.12 scores it a win).
P0_OVERLINE = (0, 100, 1, 101, 2, 102, 3, 103, 5, 120, 4)
# white on the main diagonal
P1_DIAG = (224, 0, 223, 16, 222, 32, 221, 48, 219, 64)
DRAW_3X3 = (0, 1, 2, 4, 3, 5, 7, 6, 8)  # the tic-tac-toe draw


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A jitted gomoku env.

  Returns:
    The env.
  """
  return board_helpers.JitEnv(gomoku.GomokuEnv())


def test_sizes(env):
  """Sizes.

  Args:
    env: The module `env` fixture.
  """
  assert (env.board_size, env.connect) == (15, 5)
  assert env.num_actions == 225 and env.max_len == 225 and env.num_players == 2
  assert env.max_chance_outcomes == 0


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env.env, num_playouts=16)


def test_check_env_small_board():
  """Check env small board."""
  check_env.check_env(
      gomoku.GomokuEnv(board_size=7, connect=4), num_playouts=64
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


@pytest.mark.parametrize(
    "actions,expected",
    [(P0_ROW, [1.0, -1.0]), (P0_OVERLINE, [1.0, -1.0]), (P1_DIAG, [-1.0, 1.0])],
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
    assert not bool(env.terminal(st)) and int(env.actor(st)) == k % 2


def test_full_board_draw():
  """Full board draw."""
  env = board_helpers.JitEnv(gomoku.GomokuEnv(board_size=3, connect=3))
  s = _play(env, DRAW_3X3)
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [0.0, 0.0])
  assert not bool(env.terminal(_play(env, DRAW_3X3[:-1])))


def test_occupied_cell_is_illegal(env):
  """Occupied cell is illegal.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (112, 0))
  m = np.asarray(env.legal_mask(s))
  assert not m[112] and not m[0] and m.sum() == 223


def test_info_state_and_world_from_info(env):
  """Info state and world from info.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (112, 0, 113))
  i1 = env.info_state(s, 1)
  assert int(i1.seat) == 1
  b = np.full(225, -1)
  b[[112, 113]] = 0
  b[0] = 1
  np.testing.assert_array_equal(np.asarray(i1.board), b)
  np.testing.assert_array_equal(np.asarray(env.info_state(s, 0).board), b)
  for actions in ((), (112, 0, 113), P0_ROW, P0_OVERLINE, P1_DIAG, P1_DIAG[:5]):
    st = _play(env, actions)
    for seat in (0, 1):
      w = env.world_from_info(env.info_state(st, seat))
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  env3 = board_helpers.JitEnv(gomoku.GomokuEnv(board_size=3, connect=3))
  st = _play(env3, DRAW_3X3)
  w = env3.world_from_info(env3.info_state(st, 0))
  for x, y in zip(jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_bad_sizes_fail_loud():
  """Bad sizes fail loud."""
  with pytest.raises(ValueError, match="connect"):
    gomoku.GomokuEnv(board_size=4, connect=5)
  with pytest.raises(ValueError, match="connect"):
    gomoku.GomokuEnv(board_size=5, connect=1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
