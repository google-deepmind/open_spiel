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

"""NPlayerMatrixEnv.

The Env contract on every variant, the declared structure of each table,
matching_pennies_3p equal to OpenSpiel's, and exact expected returns and
NashConv (game_tree) of the known equilibria. The variants that predate
team_dilemma_2v2 and odd_side_3v1v1 keep their tables bit for bit, so
adding variants leaves each earlier variant's rules unchanged.
"""

import ast
import hashlib
import inspect
import itertools
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import game_tree
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import nplayer_matrix as nm
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.testing import check_env

VARIANTS = sorted(nm.GAMES)


def _play(env, actions):
  """Applies `actions` from the root.

  Args:
    env: The env.
    actions: The actions, none at a terminal.

  Returns:
    The state after them.
  """
  s = env.root()
  for a in actions:
    assert not bool(env.terminal(s))
    s = env.apply(s, jnp.int32(a))
  return s


@pytest.mark.parametrize("variant", VARIANTS)
def test_check_env_passes(variant):
  """Check env passes."""
  check_env.check_env(
      nm.NPlayerMatrixEnv(variant),
      hide_fn=nm.hide_other_actions,
      num_playouts=64,
  )


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_joint_action_returns_its_table_entry(variant):
  """Every joint action returns its table entry."""
  env = nm.NPlayerMatrixEnv(variant)
  g = nm.GAMES[variant]
  for joint in itertools.product(
      range(env.num_actions), repeat=env.num_players
  ):
    s = env.root()
    for seat, a in enumerate(joint):
      assert int(env.actor(s)) == seat and not bool(env.terminal(s))
      np.testing.assert_array_equal(
          np.asarray(env.returns(s)), np.zeros(env.num_players)
      )
      s = env.apply(s, jnp.int32(a))
    assert bool(env.terminal(s)) and int(env.actor(s)) == -1
    assert not np.asarray(env.legal_mask(s)).any()
    np.testing.assert_array_equal(np.asarray(env.returns(s)), g.payoffs[joint])


def test_declared_structure():
  """Declared structure."""
  want = {  # variant: (players, side_of_seat, sum constant, zero_sum)
      "matching_pennies_3p": (3, (0, 1, 2), None, False),
      "odd_one_out_3p": (3, (0, 1, 2), 0.0, True),
      "odd_one_out_3p_c3": (3, (0, 1, 2), 3.0, False),
      "odd_one_out_4p": (4, (0, 1, 2, 3), 0.0, True),
      "team_pennies_2v2": (4, (0, 1, 0, 1), 0.0, True),
      "team_pennies_2v1": (3, (0, 1, 0), None, False),
      "team_dilemma_2v2": (4, (0, 1, 0, 1), 0.0, True),
      "odd_side_3v1v1": (5, (0, 1, 0, 2, 0), None, False),
  }
  assert set(want) == set(VARIANTS)
  for v, (p, sides, c, zs) in want.items():
    env = nm.NPlayerMatrixEnv(v)
    assert (
        env.num_players,
        env.side_of_seat,
        env.sum_constant,
        env.zero_sum,
    ) == (p, sides, c, zs)
    side_c = {"team_pennies_2v1": 0.0, "odd_side_3v1v1": 0.0}.get(v)
    assert env.side_sum_constant == side_c
    assert (
        env.constant_sum == (c is not None or side_c is not None)
        and env.max_len == p
    )
  mp = nm.GAMES["matching_pennies_3p"].payoffs
  assert len(np.unique(mp.sum(-1))) > 1  # general-sum
  t = nm.GAMES["team_pennies_2v2"].payoffs
  np.testing.assert_array_equal(t[..., 0], t[..., 2])
  np.testing.assert_array_equal(t[..., 1], t[..., 3])
  np.testing.assert_array_equal(t[..., 0], -t[..., 1])
  # A table that contradicts its declaration is refused.
  bad = nm.MatrixGame(
      "bad", nm.GAMES["odd_one_out_3p"].payoffs, (0, 1, 2), 1.0, True
  )
  with pytest.raises(AssertionError, match="constant sum"):
    nm._check_game(bad)  # pylint: disable=protected-access
  bad = nm.MatrixGame(
      "bad", nm.GAMES["odd_one_out_4p"].payoffs, (0, 1, 0, 1), 0.0, True
  )
  with pytest.raises(AssertionError, match="partners"):
    nm._check_game(bad)  # pylint: disable=protected-access
  bad = nm.MatrixGame("bad", t, (0, 1, 0, 1), 0.0, True, equal_partners=False)
  with pytest.raises(AssertionError, match="unequal partners"):
    nm._check_game(bad)  # pylint: disable=protected-access


#: sha256 of each pre-existing variant's table and declaration, as read
#: before team_dilemma_2v2 and odd_side_3v1v1 were added.
PRE_EXISTING = {
    "matching_pennies_3p": (
        "72f337d55ef08c3f1e5eeed24f73c4a5feacc440b37afb8f57e552314a021b09"
    ),
    "odd_one_out_3p": (
        "7cbc2cb865312d416d7b22d0e7d67bfc9018672cf91338e56bddb3c353e6557d"
    ),
    "odd_one_out_3p_c3": (
        "038fcc648126220a730b5c3ecf35f91297384b629bd7835ffb4d8d029597d67a"
    ),
    "odd_one_out_4p": (
        "ae3f4d2ebe8829ade2730e386f8ea124eb2c4135a22a3e6740372e4e1df79de5"
    ),
    "team_pennies_2v2": (
        "1a6cf2f5f47e01a9494e009785b5c7502c7fd1ddf1c74199b6612757034c779b"
    ),
    "team_pennies_2v1": (
        "215c6dae6f940e0670f11d32a028563cb9a76f91861c70f5bcee6e9c8616a7e3"
    ),
}


def test_pre_existing_variants_keep_their_tables():
  """The variants before the S8/3v1v1 additions are unchanged bit for bit."""
  for v, want in PRE_EXISTING.items():
    g = nm.GAMES[v]
    decl = (
        g.payoffs.shape,
        g.side_of_seat,
        g.sum_constant,
        g.rotatable,
        g.side_sum_constant,
    )
    got = hashlib.sha256(g.payoffs.tobytes() + repr(decl).encode())
    assert got.hexdigest() == want, v
    assert g.equal_partners, v


def test_team_dilemma_seat_and_side_returns_disagree():
  """Seat returns reward defecting; side returns reward cooperating."""
  g = nm.GAMES["team_dilemma_2v2"]
  sides = nm.NPlayerMatrixEnv("team_dilemma_2v2").sides
  for joint in itertools.product((0, 1), repeat=4):
    for i in range(4):
      flip = list(joint)
      flip[i] = 1 - flip[i]
      d_seat = g.payoffs[tuple(flip)][i] - g.payoffs[joint][i]
      d_side = (
          sides.side_returns(g.payoffs[tuple(flip)])[sides.side(i)]
          - sides.side_returns(g.payoffs[joint])[sides.side(i)]
      )
      # action 0 cooperates: moving to 1 gains 1 for the seat, loses 0.5
      # for its side
      sign = 1.0 if joint[i] == 0 else -1.0
      assert d_seat == sign * 1.0 and d_side == sign * -0.5, (joint, i)


def test_odd_side_3v1v1_known_entries():
  """odd_side_3v1v1: a few hand-checked joint actions."""
  g = nm.GAMES["odd_side_3v1v1"].payoffs
  # team bit 0, lone bits 0, 0: all equal
  np.testing.assert_array_equal(g[0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
  # team bit 1 (one team 1), lone 0, 0: team odd; no bonus
  np.testing.assert_array_equal(g[1, 0, 0, 0, 0], [2, -1, 2, -1, 2])
  # team 0, seat 1 plays 1 (odd), seat 3 0: side 1 +2+1, side 2 -1-1
  np.testing.assert_array_equal(g[0, 1, 0, 0, 0], [-1, 3, -1, -2, -1])
  # team 1 (seats 0, 2, 4 all 1), seat 1 1, seat 3 0: seat 3 odd; bonus
  np.testing.assert_array_equal(g[1, 1, 1, 0, 1], [-1, 0, -1, 1, -1])


def test_known_entries():
  """Known entries."""
  g = nm.GAMES
  np.testing.assert_array_equal(
      g["matching_pennies_3p"].payoffs[0, 0, 0], [1, 1, -1]
  )
  np.testing.assert_array_equal(
      g["matching_pennies_3p"].payoffs[0, 1, 0], [-1, -1, -1]
  )
  np.testing.assert_array_equal(
      g["odd_one_out_3p"].payoffs[0, 0, 1], [-1, -1, 2]
  )
  np.testing.assert_array_equal(g["odd_one_out_3p"].payoffs[1, 1, 1], [0, 0, 0])
  np.testing.assert_array_equal(
      g["odd_one_out_3p_c3"].payoffs[1, 0, 1], [0, 3, 0]
  )
  np.testing.assert_array_equal(
      g["odd_one_out_4p"].payoffs[1, 1, 0, 1], [-1, -1, 3, -1]
  )
  np.testing.assert_array_equal(
      g["odd_one_out_4p"].payoffs[1, 1, 0, 0], [0, 0, 0, 0]
  )
  np.testing.assert_array_equal(
      g["team_pennies_2v2"].payoffs[1, 0, 0, 1], [1, -1, 1, -1]
  )
  np.testing.assert_array_equal(
      g["team_pennies_2v2"].payoffs[1, 0, 0, 0], [-1, 1, -1, 1]
  )
  # 2v1: the pair (seats 0, 2) wins when a_0 XOR a_2 equals the lone seat's a_1.
  np.testing.assert_array_equal(
      g["team_pennies_2v1"].payoffs[1, 1, 0], [1, -1, 1]
  )
  np.testing.assert_array_equal(
      g["team_pennies_2v1"].payoffs[1, 0, 1], [1, -1, 1]
  )
  np.testing.assert_array_equal(
      g["team_pennies_2v1"].payoffs[0, 1, 0], [-1, 1, -1]
  )
  np.testing.assert_array_equal(
      g["team_pennies_2v1"].payoffs[1, 0, 0], [-1, 1, -1]
  )
  # A wrong side-sum declaration is refused.
  bad = nm.MatrixGame(
      "bad", g["team_pennies_2v1"].payoffs, (0, 1, 0), None, True, 1.0
  )
  with pytest.raises(AssertionError, match="side sum"):
    nm._check_game(bad)  # pylint: disable=protected-access


def test_matching_pennies_3p_equals_openspiel():
  """Matching pennies 3p equals openspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game("matching_pennies_3p")
  table = nm.GAMES["matching_pennies_3p"].payoffs
  for joint in itertools.product(range(2), repeat=3):
    st = game.new_initial_state()
    st.apply_actions(list(joint))
    np.testing.assert_array_equal(
        table[joint], np.asarray(st.returns(), np.float32)
    )


def test_privacy_no_seat_sees_another_seats_action():
  """Privacy no seat sees another seats action."""
  env = nm.NPlayerMatrixEnv("odd_one_out_4p")
  s = _play(env, (1, 0))
  for seat in range(4):
    want = 1 if seat == 2 else types.NO_ACTION  # seat 2 is the actor
    assert int(env.visible_action(s, jnp.int32(1), jnp.int32(seat))) == want
  a, b = _play(env, (0, 0, 0)), _play(env, (1, 1, 0))
  for x, y in zip(env.info_state(a, 2), env.info_state(b, 2)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  assert (
      int(env.info_state(a, 0).own_action) == 0
      and int(env.info_state(b, 0).own_action) == 1
  )


def _table(tree, strategies):
  """The tabular strategy of each infoset's player.

  Args:
    tree: The game tree.
    strategies: Each player's strategy, f64[A].

  Returns:
    f64[num_infosets, num_actions].
  """
  t = np.zeros((tree.num_infosets, tree.num_actions))
  for row in range(tree.num_infosets):
    t[row] = strategies[int(tree.player[row])]
  return t


@pytest.mark.parametrize("variant", VARIANTS)
def test_known_equilibria_have_nash_conv_zero(variant):
  """Known equilibria have nash conv zero."""
  env = nm.NPlayerMatrixEnv(variant)
  tree = game_tree.GameTree(env)
  assert tree.num_infosets == env.num_players  # one decision infoset per seat
  for eq in nm.KNOWN_EQUILIBRIA[variant]:
    table = _table(tree, eq)
    np.testing.assert_allclose(
        tree.expected_returns(table), nm.exact_returns(variant, eq), atol=1e-12
    )
    assert abs(tree.nash_conv(table)) < 1e-12, (
        variant,
        eq,
        tree.nash_conv(table),
    )
  # Every seat on action 0 is not an equilibrium of any variant.
  pure = tuple((1.0, 0.0) for _ in range(env.num_players))
  assert tree.nash_conv(_table(tree, pure)) > 0.5


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_returns_equal_the_tree_under_random_strategies(variant):
  """Exact returns equal the tree under random strategies."""
  env = nm.NPlayerMatrixEnv(variant)
  tree = game_tree.GameTree(env)
  rng = np.random.default_rng(len(variant))
  for _ in range(5):
    strat = tuple(
        tuple(x) for x in rng.dirichlet(np.ones(2), size=env.num_players)
    )
    got = tree.expected_returns(_table(tree, strat))
    np.testing.assert_allclose(
        got, nm.exact_returns(variant, strat), atol=1e-12
    )
    if env.sum_constant is not None:
      assert abs(got.sum() - env.sum_constant) < 1e-12


def test_equilibrium_values():
  """Equilibrium values."""
  u3, u4 = ((0.5, 0.5),) * 3, ((0.5, 0.5),) * 4
  np.testing.assert_allclose(
      nm.exact_returns("matching_pennies_3p", u3), [0, 0, 0], atol=0
  )
  np.testing.assert_allclose(
      nm.exact_returns("odd_one_out_3p_c3", u3), [1, 1, 1], atol=0
  )
  np.testing.assert_allclose(
      nm.exact_returns("odd_one_out_4p", u4), [0] * 4, atol=0
  )
  np.testing.assert_allclose(
      nm.exact_returns("odd_one_out_3p", ((1, 0), (1, 0), (0, 1))),
      [-1, -1, 2],
      atol=0,
  )
  with pytest.raises(ValueError, match="one distribution"):
    nm.exact_returns("odd_one_out_4p", u3)


def test_sides_are_the_declared_partition():
  """env.sides is the declared core.sides.Sides.

  Never the individual default (module doc).
  """
  for v in VARIANTS:
    env = nm.NPlayerMatrixEnv(v)
    s = sides_lib.sides_of(env)
    assert s == sides_lib.Sides(
        env.side_of_seat, env.constant_sum, env.rotatable
    ), v
  assert sides_lib.sides_of(nm.NPlayerMatrixEnv("team_pennies_2v2")) == (
      sides_lib.Sides.partnerships(4, 2, constant_sum=True)
  )
  s21 = sides_lib.sides_of(nm.NPlayerMatrixEnv("team_pennies_2v1"))
  assert s21.side_sizes == (2, 1) and s21.constant_sum and not s21.equal_sizes
  assert s21.rotation_classes == ((0,), (1,)) and s21.rotation_period == 1
  c3 = sides_lib.sides_of(nm.NPlayerMatrixEnv("odd_one_out_3p_c3"))
  assert c3.constant_sum and c3.num_sides == 3
  assert not sides_lib.sides_of(
      nm.NPlayerMatrixEnv("matching_pennies_3p")
  ).constant_sum


def test_registered_with_variant_option():
  """Registered with variant option."""
  assert registry.spec("nplayer_matrix").kwargs == {
      "variant": "matching_pennies_3p"
  }
  shas = set()
  for v in VARIANTS:
    env = registry.make("nplayer_matrix", variant=v)
    assert registry.spec_of(env) == registry.spec("nplayer_matrix", variant=v)
    shas.add(registry.rules_sha(env))
  assert len(shas) == len(VARIANTS)
  with pytest.raises(ValueError, match="variant must be one of"):
    registry.spec("nplayer_matrix", variant="chess")
  with pytest.raises(ValueError, match="variant must be one of"):
    nm.NPlayerMatrixEnv("chess")
  assert not registry.game("nplayer_matrix").served


def test_the_module_imports_no_game_package():
  """The module imports no game package."""
  tree = ast.parse(inspect.getsource(nm))
  mods = {
      n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
  }
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):
      mods |= {a.name for a in n.names}
  allowed = (
      "__future__",
      "typing",
      "dataclasses",
      "itertools",
      "jax",
      "numpy",
      subprocess_env.PACKAGE,
  )
  bad = sorted(
      m
      for m in mods
      if not any(m == a or m.startswith(a + ".") for a in allowed)
  )
  assert not bad, bad


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
