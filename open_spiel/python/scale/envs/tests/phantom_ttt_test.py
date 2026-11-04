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

"""PhantomTTTEnv (classical) tests.

The Env contract, privacy, private moves (visible_action), normalized terminal
and known answers.

Classical rules: a move onto a square the opponent holds fails, reveals that
square to the mover, and the mover moves again.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.testing import check_env

# p0 takes 4; p1 tries 4 (fails, sees X there), takes 0; p0 tries 0 (fails),
# takes 3; p1 takes 1; p0 takes 5: row 3-4-5.
P0_WINS_WITH_FAILS = (4, 4, 0, 0, 3, 1, 5)
P1_DIAG = (0, 4, 1, 2, 8, 6)
DRAW = (0, 1, 2, 4, 3, 5, 7, 6, 8)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A phantom tic-tac-toe env.

  Returns:
    The env.
  """
  return phantom_ttt.PhantomTTTEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(
      env, hide_fn=phantom_ttt.hide_opponent_marks, num_playouts=128
  )


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
    assert bool(env.legal_mask(s)[a]), (actions, a)
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


@pytest.mark.parametrize(
    "actions,expected",
    [
        (P0_WINS_WITH_FAILS, [1.0, -1.0]),
        (P1_DIAG, [-1.0, 1.0]),
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
  assert int(env.actor(s)) == -1  # The engine says -4; normalized
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert not bool(env.terminal(_play(env, actions[:-1])))


def test_failed_move_keeps_the_turn_and_reveals(env):
  """Failed move keeps the turn and reveals.

  Args:
    env: The module `env` fixture.
  """
  actors = [
      int(env.actor(_play(env, P0_WINS_WITH_FAILS[:k]))) for k in range(7)
  ]
  assert actors == [0, 1, 1, 0, 0, 1, 0]
  s = _play(env, P0_WINS_WITH_FAILS[:3])
  i1 = env.info_state(s, 1)
  # p1 sees X (2) at 4 (revealed by its failed try) and its own O (1) at 0.
  np.testing.assert_array_equal(
      np.asarray(i1.view), [1, 0, 0, 0, 2, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(np.asarray(i1.own_actions), [4, 0] + [-1] * 7)
  assert int(i1.num_own_actions) == 2
  i0 = env.info_state(s, 0)
  np.testing.assert_array_equal(
      np.asarray(i0.view), [0, 0, 0, 0, 2, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(np.asarray(i0.own_actions), [4] + [-1] * 8)
  # p1 can no longer try 4 or 0; p0 can still try 0 (it does not know).
  assert not bool(env.legal_mask(s)[4])


def test_visible_action_hides_the_opponents_moves(env):
  """Visible action hides the opponents moves.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (4,))  # p1 to move
  assert int(env.visible_action(s, jnp.int32(2), 1)) == 2
  assert int(env.visible_action(s, jnp.int32(2), 0)) == types.NO_ACTION


def test_info_state_ignores_opponent_moves(env):
  """p1 placed at 0 or at 8 (p0 cannot tell): p0's InfoState is identical."""
  a = env.info_state(_play(env, (4, 0)), 0)
  b = env.info_state(_play(env, (4, 8)), 0)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  lab = env.labels(_play(env, (4, 0)), 0)
  np.testing.assert_array_equal(
      np.asarray(lab["opp_marks"]), [True] + [False] * 8
  )
  assert env.world_from_info(a) is None


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
