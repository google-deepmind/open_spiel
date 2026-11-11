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

"""PhantomTTTIREnv tests.

The game is OpenSpiel phantom_ttt_ir(gameversion=abrupt). Tested: the Env
contract, privacy, private moves, imperfect recall and known answers.

Abrupt rules: a move onto a square the opponent holds fails, reveals that
square to the mover, and passes the turn. Imperfect recall (the OpenSpiel
game's information state is its observation): the InfoState is the seat and
its view, never its attempt history. Views use envs.phantom_ttt's marks:
0 unseen/empty, 2 X (seat 0), 1 O (seat 1) (the ir engine has 1 X,
2 O; the wrapper normalizes).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import phantom_ttt_ir
from open_spiel.python.scale.testing import check_env

# p0 takes 4; p1 tries 4 (fails, sees X, turn passes); p0 takes 0; p1 tries 0
# (fails); p0 takes 8: diagonal 0-4-8.
P0_DIAG_WITH_FAILS = (4, 4, 0, 0, 8)
P1_DIAG = (0, 4, 1, 2, 5, 6)
DRAW = (0, 1, 2, 4, 3, 5, 7, 6, 8)
GAME = "phantom_ttt_ir(gameversion=abrupt)"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """An abrupt phantom tic-tac-toe env.

  Returns:
    The env.
  """
  return phantom_ttt_ir.PhantomTTTIREnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(
      env, hide_fn=phantom_ttt_ir.hide_opponent_marks, num_playouts=128
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


KNOWN = [
    (P0_DIAG_WITH_FAILS, [1.0, -1.0]),
    (P1_DIAG, [-1.0, 1.0]),
    (DRAW, [0.0, 0.0]),
]


@pytest.mark.parametrize("actions,expected", KNOWN)
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


def test_known_returns_match_pyspiel(env):
  """Known returns match pyspiel.

  Args:
    env: The module `env` fixture.
  """
  del env  # Unused.
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game(GAME)
  for actions, expected in KNOWN:
    st = game.new_initial_state()
    for a in actions:
      st.apply_action(a)
    assert st.is_terminal()
    np.testing.assert_array_equal(
        np.asarray(st.returns(), np.float32), expected
    )


def test_failed_move_passes_the_turn_and_reveals(env):
  """Failed move passes the turn and reveals.

  Args:
    env: The module `env` fixture.
  """
  actors = [
      int(env.actor(_play(env, P0_DIAG_WITH_FAILS[:k]))) for k in range(5)
  ]
  assert actors == [0, 1, 0, 1, 0]  # abrupt: strict alternation
  s = _play(env, P0_DIAG_WITH_FAILS[:2])
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, 1).view), [0, 0, 0, 0, 2, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, 0).view), [0, 0, 0, 0, 2, 0, 0, 0, 0]
  )
  s = _play(env, P0_DIAG_WITH_FAILS[:3])  # p1 to move: it may not retry 4
  assert int(env.actor(s)) == 1 and not bool(env.legal_mask(s)[4])
  assert bool(env.legal_mask(s)[0])  # p1 does not know p0 holds 0
  s = _play(env, (0, 4))  # p1 placed O at 4
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, 1).view), [0, 0, 0, 0, 1, 0, 0, 0, 0]
  )


def test_imperfect_recall_info_state_is_the_view(env):
  """Imperfect recall info state is the view.

  p0 took 0 then 1, or 1 then 0 (p1 took 8 between): the same view, so the
  same InfoState; the perfect-recall phantom_ttt would tell them apart.

  Args:
    env: The module `env` fixture.
  """
  a = env.info_state(_play(env, (0, 8, 1)), 0)
  b = env.info_state(_play(env, (1, 8, 0)), 0)
  assert set(a._fields) == {"seat", "view"}
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  np.testing.assert_array_equal(np.asarray(a.view), [2, 2, 0, 0, 0, 0, 0, 0, 0])


def test_info_state_ignores_opponent_moves(env):
  """Info state ignores opponent moves.

  Args:
    env: The module `env` fixture.
  """
  a = env.info_state(_play(env, (4, 0)), 0)
  b = env.info_state(_play(env, (4, 8)), 0)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  lab = env.labels(_play(env, (4, 0)), 0)
  np.testing.assert_array_equal(
      np.asarray(lab["opp_marks"]), [True] + [False] * 8
  )
  assert env.world_from_info(a) is None


def test_visible_action_hides_the_opponents_moves(env):
  """Visible action hides the opponents moves.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, (4,))  # p1 to move
  assert int(env.visible_action(s, jnp.int32(2), 1)) == 2
  assert int(env.visible_action(s, jnp.int32(2), 0)) == types.NO_ACTION


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
