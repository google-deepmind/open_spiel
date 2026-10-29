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

"""HexEnv tests.

The Env contract, ABSOLUTE pyspiel cells for both seats (the engine's hex
transposes seat 1's actions; the wrapper undoes it), normalized terminal, known
answers, perfect-information world_from_info.

pyspiel hex: cell = row * size + col (a1 = 0, b1 = 1, ...), each row shifted
right of the one above, so (r, c) touches (r-1, c), (r-1, c+1), (r, c-1),
(r, c+1), (r+1, c-1), (r+1, c). Seat 0 (x) joins top and bottom, seat 1 (o)
left and right. No swap rule (pyspiel's default).
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.envs import hex as hex_env
from open_spiel.python.scale.envs.tests import board_helpers
from open_spiel.python.scale.testing import check_env

# 3x3, recorded with pyspiel 1.6.12 (see module docstring for adjacency).
P0_COLUMN_3 = (0, 4, 3, 5, 6)  # x: a1 a2 a3 (column 0) top to bottom
P1_ROW_3 = (0, 3, 1, 4, 8, 5)  # o: a2 b2 c2 (row 1) left to right
# x on b1, a2 (touching: (0,1)-(1,0)), a3: a zig-zag top-bottom chain.
P0_ZIGZAG_3 = (1, 4, 3, 5, 6)


def _column_11_moves() -> tuple[int, ...]:
  """Moves where black fills column 0 and white column 5.

  Returns:
    The moves.
  """
  moves = []
  for r in range(11):
    moves.append(r * 11)
    if r < 10:
      moves.append(r * 11 + 5)
  return tuple(moves)


# 11x11: x plays column 0 top to bottom; o plays column 5 (no left-right chain).
P0_COLUMN_11 = _column_11_moves()


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A jitted 11x11 hex env.

  Returns:
    The env.
  """
  return board_helpers.JitEnv(hex_env.HexEnv())


@pytest.fixture(name="env3", scope="module")
def _env3_fixture():
  """A jitted 3x3 hex env.

  Returns:
    The env.
  """
  return board_helpers.JitEnv(hex_env.HexEnv(board_size=3))


def test_sizes(env):
  """Sizes.

  Args:
    env: The module `env` fixture.
  """
  assert env.board_size == 11 and env.num_actions == 121 and env.max_len == 121
  assert env.num_players == 2 and env.max_chance_outcomes == 0


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env.env, num_playouts=16)


def test_check_env_small_board():
  """Check env small board."""
  check_env.check_env(hex_env.HexEnv(board_size=5), num_playouts=64)


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
    [
        (P0_COLUMN_3, [1.0, -1.0]),
        (P1_ROW_3, [-1.0, 1.0]),
        (P0_ZIGZAG_3, [1.0, -1.0]),
    ],
)
def test_known_terminal_returns_3x3(env3, actions, expected):
  """Known terminal returns 3x3.

  Args:
    env3: The module `env3` fixture.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  s = _play(env3, actions)
  assert bool(env3.terminal(s)) and int(env3.actor(s)) == -1
  assert not bool(env3.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env3.returns(s)), expected)
  for k in range(len(actions)):
    st = _play(env3, actions[:k])
    assert not bool(env3.terminal(st)) and int(env3.actor(st)) == k % 2


def test_known_terminal_returns_11x11(env):
  """Known terminal returns 11x11.

  Args:
    env: The module `env` fixture.
  """
  assert len(P0_COLUMN_11) == 21
  s = _play(env, P0_COLUMN_11)
  assert bool(env.terminal(s))
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [1.0, -1.0])
  assert not bool(env.terminal(_play(env, P0_COLUMN_11[:-1])))


def test_non_neighbours_do_not_connect(env3):
  """(0,0) and (1,1) are not adjacent in hex: x on a1, b2, b3 has no chain."""
  s = _play(env3, (0, 2, 4, 5, 7))
  assert not bool(env3.terminal(s))


def test_seat1_cells_are_absolute(env3):
  """Seat 1's action 1 marks cell 1 (b1), not its transpose (a2)."""
  s = _play(env3, (4, 1))
  b = np.asarray(env3.info_state(s, 0).board)
  np.testing.assert_array_equal(b, [-1, 1, -1, -1, 0, -1, -1, -1, -1])
  np.testing.assert_array_equal(np.asarray(env3.legal_mask(s)), b == -1)


def test_world_from_info_round_trips(env, env3):
  """World from info round trips.

  Args:
    env: The module `env` fixture.
    env3: The module `env3` fixture.
  """
  for e, actions in (
      (env3, ()),
      (env3, (4, 1)),
      (env3, P0_COLUMN_3),
      (env3, P1_ROW_3),
      (env3, P0_ZIGZAG_3),
      (env3, P1_ROW_3[:4]),
      (env, P0_COLUMN_11),
      (env, P0_COLUMN_11[:12]),
  ):
    st = _play(e, actions)
    for seat in (0, 1):
      info = e.info_state(st, seat)
      assert int(info.seat) == seat
      w = e.world_from_info(info)
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_bad_size_fails_loud():
  """Bad size fails loud."""
  with pytest.raises(ValueError, match="board_size"):
    hex_env.HexEnv(board_size=1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
