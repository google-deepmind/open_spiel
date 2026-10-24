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

"""TicTacToeEnv tests.

The Env contract, normalized terminal, known answers, perfect-information
world_from_info.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import tic_tac_toe
from open_spiel.python.scale.testing import check_env

X_ROW = (0, 3, 1, 4, 2)
O_DIAG = (0, 4, 1, 2, 8, 6)  # O takes 2-4-6
DRAW = (0, 1, 2, 4, 3, 5, 7, 6, 8)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A tic-tac-toe env.

  Returns:
    The env.
  """
  return tic_tac_toe.TicTacToeEnv()


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, num_playouts=128)


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
    assert not bool(env.terminal(s))
    assert bool(env.legal_mask(s)[a])
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


@pytest.mark.parametrize(
    "actions,expected",
    [(X_ROW, [1.0, -1.0]), (O_DIAG, [-1.0, 1.0]), (DRAW, [0.0, 0.0])],
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
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  for k in range(len(actions)):
    assert not bool(env.terminal(_play(env, actions[:k])))


@pytest.mark.parametrize("actions", [X_ROW, O_DIAG, DRAW])
def test_terminal_normalized(env, actions):
  """Terminal normalized.

  The engine's ttt keeps flipping current_player after terminal;
  the wrapper reports actor -1 and an empty legal mask.

  Args:
    env: The module `env` fixture.
    actions: The moves of a finished game.
  """
  s = _play(env, actions)
  assert int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())


def test_actor_alternates(env):
  """Actor alternates.

  Args:
    env: The module `env` fixture.
  """
  for k in range(len(DRAW)):
    assert int(env.actor(_play(env, DRAW[:k]))) == k % 2


def test_info_state_and_world_from_info(env):
  """Info state and world from info.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (4, 0, 8))
  i1 = env.info_state(s, 1)
  assert int(i1.seat) == 1
  np.testing.assert_array_equal(
      np.asarray(i1.board), [1, -1, -1, -1, 0, -1, -1, -1, 0]
  )
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, 0).board), np.asarray(i1.board)
  )
  assert not dict(env.labels(s, 0))
  for actions in ((), (4, 0, 8), X_ROW, O_DIAG, DRAW):
    st = _play(env, actions)
    w = env.world_from_info(env.info_state(st, 0))
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
