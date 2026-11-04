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

"""Dark Hex 3x3 envs: classical (dark_hex) and abrupt with imperfect recall.

(dark_hex_ir = OpenSpiel dark_hex_ir(gameversion=adh)). The Env contract with
privacy hooks, private moves (visible_action), known answers, and agreement with
pyspiel on random games.

Classical: a move onto a cell the opponent holds fails, reveals the cell to
the mover, and the mover moves again. Abrupt: the failed move reveals the
cell and passes the turn. Black (seat 0) joins top to bottom, white (seat 1)
left to right; cells are row-major.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import dark_hex
from open_spiel.python.scale.envs.tests import differential_vs_openspiel as pd
from open_spiel.python.scale.testing import check_env

B_COLUMN = (0, 1, 3, 2, 6)  # black 0-3-6 joins top to bottom
# white 3-4-5 joins left to right; black 0, 1, 8 do not
W_ROW = (0, 3, 1, 4, 8, 5)


@pytest.fixture(name="classical", scope="module")
def _classical_fixture():
  """A classical dark hex env.

  Returns:
    The env.
  """
  return dark_hex.DarkHexEnv()


@pytest.fixture(name="abrupt", scope="module")
def _abrupt_fixture():
  """An abrupt dark hex env.

  Returns:
    The env.
  """
  return dark_hex.DarkHexIREnv()


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
    assert bool(env.legal_mask(s)[a]), (actions, a)
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


def _same(a, b):
  """Whether two pytrees have equal leaves.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    True when equal.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  return len(la) == len(lb) and all(
      np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
  )


def test_names_and_sizes(classical, abrupt):
  """Names and sizes.

  Args:
    classical: The module `classical` fixture.
    abrupt: The module `abrupt` fixture.
  """
  assert classical.name == "dark_hex"
  for env in (classical, abrupt):
    assert env.num_actions == 9 and env.num_players == 2 and env.max_len == 17
    assert env.max_chance_outcomes == 0 and env.zero_sum
  assert abrupt.name == "dark_hex_ir"


def test_check_env_classical_with_privacy(classical):
  """Check env classical with privacy.

  Args:
    classical: The module `classical` fixture.
  """
  check_env.check_env(
      classical, hide_fn=dark_hex.hide_unseen_cells, num_playouts=128
  )


def test_check_env_abrupt_with_privacy(abrupt):
  """Check env abrupt with privacy.

  Args:
    abrupt: The module `abrupt` fixture.
  """
  check_env.check_env(
      abrupt, hide_fn=dark_hex.hide_unseen_cells_ir, num_playouts=128
  )


@pytest.mark.parametrize(
    "actions,expected", [(B_COLUMN, [1.0, -1.0]), (W_ROW, [-1.0, 1.0])]
)
def test_known_terminal_returns(classical, abrupt, actions, expected):
  """Known terminal returns.

  Args:
    classical: The module `classical` fixture.
    abrupt: The module `abrupt` fixture.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  for env in (classical, abrupt):  # no failed moves: both variants agree
    s = _play(env, actions)
    assert bool(env.terminal(s)) and int(env.actor(s)) == -1
    assert not bool(env.legal_mask(s).any())
    np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
    assert not bool(env.terminal(_play(env, actions[:-1])))


def test_failed_move_keeps_the_turn_classical(classical):
  """Failed move keeps the turn classical.

  Args:
    classical: The module `classical` fixture.
  """
  seq = (4, 4, 0)  # black 4; white tries 4 (fails, sees black there), white 0
  actors = [int(classical.actor(_play(classical, seq[:k]))) for k in range(4)]
  assert actors == [0, 1, 1, 0]
  s = _play(classical, seq)
  i1 = classical.info_state(s, 1)
  # White's view: black (+1) at 4 revealed by its failed try, its own (-1) at 0.
  np.testing.assert_array_equal(
      np.asarray(i1.view), [-1, 0, 0, 0, 1, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(np.asarray(i1.own_actions), [4, 0] + [-1] * 7)
  assert int(i1.num_own_actions) == 2
  i0 = classical.info_state(s, 0)
  np.testing.assert_array_equal(
      np.asarray(i0.view), [0, 0, 0, 0, 1, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(np.asarray(i0.own_actions), [4] + [-1] * 8)
  assert bool(classical.legal_mask(s)[0])  # black does not know white holds 0


def test_failed_move_passes_the_turn_abrupt(abrupt):
  """Failed move passes the turn abrupt.

  Args:
    abrupt: The module `abrupt` fixture.
  """
  seq = (4, 4, 0)  # black 4; white tries 4 (fails, turn passes); black 0
  actors = [int(abrupt.actor(_play(abrupt, seq[:k]))) for k in range(4)]
  assert actors == [0, 1, 0, 1]
  s = _play(abrupt, seq)
  # black (1) seen at 4
  np.testing.assert_array_equal(
      np.asarray(abrupt.info_state(s, 1).view), [0, 0, 0, 0, 1, 0, 0, 0, 0]
  )
  np.testing.assert_array_equal(
      np.asarray(abrupt.info_state(s, 0).view), [1, 0, 0, 0, 1, 0, 0, 0, 0]
  )
  assert not bool(abrupt.legal_mask(s)[4])  # white knows 4 is taken


def test_visible_action_hides_the_opponents_moves(classical, abrupt):
  """Visible action hides the opponents moves.

  Args:
    classical: The module `classical` fixture.
    abrupt: The module `abrupt` fixture.
  """
  for env in (classical, abrupt):
    s = _play(env, (4,))  # white to move
    assert int(env.visible_action(s, jnp.int32(2), 1)) == 2
    assert int(env.visible_action(s, jnp.int32(2), 0)) == types.NO_ACTION


def test_info_state_ignores_opponent_moves(classical, abrupt):
  """Info state ignores opponent moves.

  Args:
    classical: The module `classical` fixture.
    abrupt: The module `abrupt` fixture.
  """
  for env in (classical, abrupt):
    a = env.info_state(_play(env, (4, 0)), 0)
    b = env.info_state(_play(env, (4, 8)), 0)
    assert _same(a, b)
    lab = env.labels(_play(env, (4, 0)), 0)
    np.testing.assert_array_equal(
        np.asarray(lab["opp_stones"]), [True] + [False] * 8
    )
    assert env.world_from_info(a) is None


def test_ir_forgets_the_order_of_own_moves(abrupt):
  """Ir forgets the order of own moves.

  Imperfect recall: black 0 then 2, or 2 then 0 (white elsewhere), is one
  InfoState.

  Args:
    abrupt: The module imperfect-recall DarkHexIREnv fixture.
  """
  a = abrupt.info_state(_play(abrupt, (0, 6, 2, 7)), 0)
  b = abrupt.info_state(_play(abrupt, (2, 6, 0, 7)), 0)
  assert _same(a, b)


@pytest.mark.parametrize(
    "name,params", [("dark_hex", {}), ("dark_hex_ir", {"gameversion": "adh"})]
)
def test_agrees_with_pyspiel_on_random_games(name, params):
  """Agrees with pyspiel on random games.

  Args:
    name: The parametrized case.
    params: The parametrized case.
  """
  pyspiel = pytest.importorskip("pyspiel")
  env = dark_hex.DarkHexEnv() if name == "dark_hex" else dark_hex.DarkHexIREnv()
  game = pyspiel.load_game(name, params)
  assert game.max_game_length() == env.max_len
  stats = pd.check_random_games(env, game, 300, seed=0)
  assert stats["terminals"] == 300 and len(stats["outcomes"]) == 2


def test_classical_alias_is_the_same_game(classical):
  """Classical alias is the same game.

  Args:
    classical: The module `classical` fixture.
  """
  ref = dark_hex.DarkHexEnv()
  for seq in (B_COLUMN, W_ROW, (4, 4, 0, 1)):
    assert _same(_play(ref, seq), _play(classical, seq))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
