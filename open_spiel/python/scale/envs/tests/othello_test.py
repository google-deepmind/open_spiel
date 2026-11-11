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

"""OthelloEnv tests.

The Env contract, pyspiel action ids (row * 8 + col, row 0 = rank 1; 64 = pass),
a normalized board (-1 empty, 0 black, 1 white), normalized terminal, known
answers, perfect-information world_from_info.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.envs import othello
from open_spiel.python.scale.envs.tests import board_helpers
from open_spiel.python.scale.testing import check_env


def _sq(name):
  """The cell of a square name such as d3.

  Args:
    name: The square name.

  Returns:
    The cell index.
  """
  return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])


# The shortest Othello game: black wipes white out in 9 moves (13-0).
WIPEOUT = tuple(_sq(m) for m in "e6 f4 e3 f6 g5 d6 e7 f5 c5".split())
# A random pyspiel 1.6.12 game with one pass (white, ply 57: pass is its only
# legal action); white wins.
PASS_GAME = (
    19,
    34,
    43,
    20,
    29,
    51,
    59,
    10,
    41,
    30,
    22,
    48,
    31,
    42,
    33,
    38,
    18,
    15,
    11,
    49,
    47,
    3,
    57,
    24,
    14,
    40,
    26,
    17,
    50,
    21,
    16,
    56,
    1,
    52,
    12,
    5,
    37,
    23,
    32,
    8,
    13,
    58,
    60,
    44,
    61,
    53,
    7,
    6,
    46,
    45,
    39,
    25,
    54,
    62,
    55,
    9,
    4,
    64,
    2,
)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A jitted othello env.

  Returns:
    The env.
  """
  return board_helpers.JitEnv(othello.OthelloEnv())


def test_sizes(env):
  """Sizes.

  Args:
    env: The module `env` fixture.
  """
  assert (
      env.num_actions == 65
      and env.num_players == 2
      and env.max_chance_outcomes == 0
  )
  # <= 60 placements; a pass is always followed by a placement.
  assert env.max_len == 120


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env.env, num_playouts=32)


def _play(env, actions):
  """Plays `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  return board_helpers.play(env, actions)


def test_initial_position(env):
  """Initial position.

  Args:
    env: The module `env` fixture.
  """
  s = env.init(jax.random.key(0))
  assert np.flatnonzero(np.asarray(env.legal_mask(s))).tolist() == [
      19,
      26,
      37,
      44,
  ]
  b = np.asarray(env.info_state(s, 0).board).reshape(8, 8)
  assert b[3, 3] == 1 and b[4, 4] == 1 and b[3, 4] == 0 and b[4, 3] == 0
  assert (b == -1).sum() == 60
  assert int(env.actor(s)) == 0


def test_wipeout(env):
  """Wipeout.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, WIPEOUT)
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [1.0, -1.0])
  b = np.asarray(env.info_state(s, 1).board)
  assert (b == 0).sum() == 13 and (b == 1).sum() == 0
  for k in range(len(WIPEOUT)):
    st = _play(env, WIPEOUT[:k])
    assert not bool(env.terminal(st)) and int(env.actor(st)) == k % 2


def test_pass_game(env):
  """Pass game.

  Args:
    env: The module `env` fixture.
  """
  i = PASS_GAME.index(othello.PASS)
  before = _play(env, PASS_GAME[:i])
  assert int(env.actor(before)) == 1
  assert np.flatnonzero(np.asarray(env.legal_mask(before))).tolist() == [
      othello.PASS
  ]
  after = _play(env, PASS_GAME[: i + 1])
  assert int(env.actor(after)) == 0  # the passer's opponent moves again
  assert int(env.info_state(after, 1).passes) == 1
  assert not np.asarray(env.legal_mask(after))[othello.PASS]
  s = _play(env, PASS_GAME)
  assert bool(env.terminal(s))
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [-1.0, 1.0])
  # 58 discs placed + 4: two empty squares neither side can take ends it.
  assert (np.asarray(env.info_state(s, 0).board) == -1).sum() == 2
  assert int(env.info_state(s, 0).passes) == 0


def test_world_from_info_round_trips(env):
  """World from info round trips.

  Args:
    env: The module `env` fixture.
  """
  i = PASS_GAME.index(othello.PASS)
  for actions in (
      (),
      WIPEOUT[:4],
      PASS_GAME[:i],
      PASS_GAME[: i + 1],
      PASS_GAME[:30],
  ):
    st = _play(env, actions)
    for seat in (0, 1):
      info = env.info_state(st, seat)
      w = env.world_from_info(info)
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  for actions in (WIPEOUT, PASS_GAME):  # terminal: the info round-trips
    st = _play(env, actions)
    info = env.info_state(st, 1)
    w = env.world_from_info(info)
    assert bool(env.terminal(w))
    np.testing.assert_array_equal(
        np.asarray(env.returns(w)), np.asarray(env.returns(st))
    )
    for x, y in zip(info, env.info_state(w, 1)):
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
