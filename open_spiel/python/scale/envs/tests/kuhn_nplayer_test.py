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

"""KuhnNPlayerEnv.

The Env contract, known payoffs, equality with pyspiel kuhn_poker(players=N) on
every node of the N=3 tree and of random N=2, 4, 10 games, equality with the
two-player built-in at players=2, and exact expected returns and NashConv
against OpenSpiel's own on N=3 and N=4.
"""

import ast
import inspect
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import game_tree
from open_spiel.python.scale.envs import kuhn_jax
from open_spiel.python.scale.envs import kuhn_nplayer
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.envs.tests import differential_vs_openspiel
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.testing import check_env

P, B = kuhn_nplayer.PASS, kuhn_nplayer.BET


def _env(n):
  """The n-player kuhn env.

  Args:
    n: The players.

  Returns:
    The KuhnNPlayerEnv.
  """
  return kuhn_nplayer.KuhnNPlayerEnv(players=n)


def _play(env, s, actions):
  """Applies `actions` from state `s`.

  Args:
    env: The env.
    s: The state.
    actions: The actions, none at a terminal.

  Returns:
    The state after them.
  """
  for a in actions:
    assert not bool(env.terminal(s))
    s = env.apply(s, jnp.int32(a))
  return s


@pytest.mark.parametrize("n", [2, 3, 4])
def test_check_env_passes(n):
  """Check env passes."""
  check_env.check_env(
      _env(n), hide_fn=kuhn_nplayer.hide_other_cards, num_playouts=128
  )


def test_sizes():
  """Sizes."""
  for n in (2, 3, 4, 10):
    env = _env(n)
    assert (
        env.num_players,
        env.num_actions,
        env.max_len,
        env.max_chance_outcomes,
    ) == (n, 2, 2 * n - 1, n + 1)
    assert env.zero_sum and not hasattr(env, "sides")


@pytest.mark.parametrize(
    "cards,actions,expected,actors",
    [
        ((3, 1, 0), (P, P, P), [2, -1, -1], (0, 1, 2)),
        ((0, 1, 2), (P, P, P), [-1, -1, 2], (0, 1, 2)),
        ((2, 3, 0), (B, P, P), [2, -1, -1], (0, 1, 2)),
        ((2, 3, 0), (B, B, P), [-2, 3, -1], (0, 1, 2)),
        ((2, 3, 0), (B, B, B), [-2, 4, -2], (0, 1, 2)),
        ((1, 0, 2), (P, B, B, P), [-1, -2, 3], (0, 1, 2, 0)),
        ((1, 0, 2), (P, B, P, B), [3, -2, -1], (0, 1, 2, 0)),
        ((3, 1, 0), (P, P, B, B, B), [4, -2, -2], (0, 1, 2, 0, 1)),
        ((3, 1, 0), (P, P, B, P, P), [-1, -1, 2], (0, 1, 2, 0, 1)),
    ],
)
def test_known_payoffs_three_players(cards, actions, expected, actors):
  """Known payoffs three players."""
  env = _env(3)
  s = env.deal(*cards)
  seen = []
  for a in actions:
    seen.append(int(env.actor(s)))
    s = _play(env, s, (a,))
  assert tuple(seen) == actors
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  np.testing.assert_array_equal(
      np.asarray(env.returns(s)), np.asarray(expected, np.float32)
  )


def test_chance_layout():
  """Chance layout."""
  env = _env(3)
  s = env.root()
  assert bool(env.is_chance(s)) and not bool(env.terminal(s))
  np.testing.assert_allclose(env.chance_probs(s), [0.25] * 4)
  s = env.apply_chance(s, 2)
  np.testing.assert_allclose(env.chance_probs(s), [1 / 3, 1 / 3, 0, 1 / 3])
  s = env.apply_chance(env.apply_chance(s, 0), 3)
  assert not bool(env.is_chance(s)) and int(env.actor(s)) == 0
  np.testing.assert_array_equal(np.asarray(s.cards), [2, 0, 3])
  info = env.info_state(s, 1)
  assert int(info.card) == 0 and int(info.seat) == 1
  np.testing.assert_array_equal(
      np.asarray(env.labels(s, 1)["opp_cards"]), [3, 2]
  )
  with pytest.raises(ValueError, match="distinct"):
    env.deal(1, 1, 2)


def test_bad_player_counts_fail_loud():
  """Bad player counts fail loud."""
  for bad in (1, 11):
    with pytest.raises(ValueError, match="players"):
      _env(bad)
  with pytest.raises(TypeError, match="players"):
    kuhn_nplayer.KuhnNPlayerEnv(players=True)
  with pytest.raises(ValueError, match="players"):
    registry.spec("kuhn_nplayer", players=11)


def test_registered_with_players_option():
  """Registered with players option."""
  assert registry.spec("kuhn_nplayer").kwargs == {"players": 3}
  e4 = registry.make("kuhn_nplayer", players=4)
  assert isinstance(e4, kuhn_nplayer.KuhnNPlayerEnv) and e4.num_players == 4
  assert registry.spec_of(e4) == registry.spec("kuhn_nplayer", players=4)
  shas = {
      registry.rules_sha(registry.make("kuhn_nplayer", players=n))
      for n in (2, 3, 4)
  }
  assert len(shas) == 3 and all(len(x) == 64 for x in shas)
  assert registry.rules_sha(_env(2)) != registry.rules_sha(
      kuhn_jax.KuhnJaxEnv()
  )
  assert (
      not registry.game("kuhn_nplayer").served
      and registry.game("kuhn_nplayer").exact_eval
  )
  assert "kuhn_nplayer" in registry.BUILTIN_STANDALONE


def test_the_module_imports_no_game_package():
  """The module imports no game package."""
  tree = ast.parse(inspect.getsource(kuhn_nplayer))
  mods = {
      n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
  }
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):
      mods |= {a.name for a in n.names}
  allowed = ("__future__", "typing", "jax", subprocess_env.PACKAGE)
  bad = sorted(
      m
      for m in mods
      if not any(m == a or m.startswith(a + ".") for a in allowed)
  )
  assert not bad, bad


def test_two_players_equal_the_builtin_on_every_state():
  """Two players equal the builtin on every state."""
  env, ref = _env(2), kuhn_jax.KuhnJaxEnv()
  histories = ((P, P), (B, B), (B, P), (P, B, P), (P, B, B))
  for c0, c1 in itertools.permutations(range(3), 2):
    a, b = env.deal(c0, c1), ref.deal(c0, c1)
    for hist in histories:
      for k in range(len(hist) + 1):
        sa, sb = _play(env, a, hist[:k]), _play(ref, b, hist[:k])
        for f in ("terminal", "actor", "legal_mask", "returns"):
          np.testing.assert_array_equal(
              np.asarray(getattr(env, f)(sa)),
              np.asarray(getattr(ref, f)(sb)),
              err_msg=f"{f} {hist[:k]}",
          )
        for seat in (0, 1):
          for x, y in zip(env.info_state(sa, seat), ref.info_state(sb, seat)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  np.testing.assert_allclose(
      env.chance_probs(env.apply_chance(env.root(), 1)),
      ref.chance_probs(ref.apply_chance(ref.root(), 1)),
  )


def _walk(env, game):
  """Every node of the full pyspiel tree with the env replayed alongside.

  Depth first; yields differential_vs_openspiel.Node.

  Args:
    env: the KuhnNPlayerEnv.
    game: the pyspiel game it replays.

  Yields:
    differential_vs_openspiel.Node per node, depth first.
  """
  fns = differential_vs_openspiel._fns(env)  # pylint: disable=protected-access
  stack = [(game.new_initial_state(), fns.root(), 0)]
  while stack:
    st, ours, moves = stack.pop()
    is_chance = bool(fns.node(ours)[0])
    assert is_chance == st.is_chance_node(), st.history()
    if st.is_chance_node():
      probs = np.asarray(env.chance_probs(ours))
      want = np.zeros_like(probs)
      for o, p in st.chance_outcomes():
        want[o] = p
      np.testing.assert_allclose(
          probs, want, rtol=1e-6, err_msg=str(st.history())
      )
      for o, _ in st.chance_outcomes():
        stack.append((st.child(o), fns.apply_chance(ours, np.int32(o)), moves))
      continue
    yield differential_vs_openspiel.Node(st, ours, moves)
    if not st.is_terminal():
      for a in st.legal_actions():
        stack.append((st.child(a), fns.apply(ours, np.int32(a)), moves + 1))


def test_every_node_of_the_three_player_tree_equals_pyspiel():
  """Every node of the three player tree equals pyspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  env = _env(3)
  game = pyspiel.load_game("kuhn_poker", {"players": 3})
  nodes = terminals = 0
  outcomes = set()
  for node in _walk(env, game):
    differential_vs_openspiel.assert_same_node(env, node)
    nodes += 1
    if node.ref.is_terminal():
      terminals += 1
      outcomes.add(tuple(node.ref.returns()))
  # 24 deals x 13 betting sequences that end the hand.
  assert terminals == 24 * 13, terminals
  assert len(outcomes) > 10 and nodes > terminals


@pytest.mark.parametrize("n,games", [(2, 200), (4, 400), (10, 60)])
def test_random_games_equal_pyspiel(n, games):
  """Random games equal pyspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  env = _env(n)
  out = differential_vs_openspiel.check_random_games(
      env, pyspiel.load_game("kuhn_poker", {"players": n}), games, seed=n
  )
  assert out["terminals"] == games and len(out["outcomes"]) > 3


def _info_string(info) -> str:
  """Pyspiel's kuhn_poker information_state_string of one KuhnNInfo row.

  Args:
    info: One KuhnNInfo row.

  Returns:
    The string.
  """
  return str(int(info.card)) + "".join(
      "pb"[int(b)] for b in info.bets[: int(info.num_bets)]
  )


def _tabular(pyspiel, tree, table):
  """Table as a pyspiel.TabularPolicy keyed by information-state string.

  A missing key fails inside pyspiel's evaluation.

  Args:
    pyspiel: the pyspiel module.
    tree: the env's GameTree.
    table: f[infosets, A] action probabilities, one row per tree infoset.

  Returns:
    The pyspiel.TabularPolicy.
  """
  rows = {}
  for row in range(tree.num_infosets):
    info = jax.tree_util.tree_map(lambda x: x[row], tree.info)  # pylint: disable=cell-var-from-loop
    rows[_info_string(info)] = [
        (a, float(table[row, a])) for a in range(table.shape[1])
    ]
  assert len(rows) == tree.num_infosets
  return pyspiel.TabularPolicy(rows)


@pytest.mark.parametrize("n", [3, 4])
def test_exact_values_and_nash_conv_equal_openspiel(n):
  """Exact values and nash conv equal openspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game("kuhn_poker", {"players": n})
  tree = game_tree.GameTree(_env(n))
  rng = np.random.default_rng(n)
  tables = [tree.uniform_table()]
  for _ in range(2):
    t = rng.dirichlet(np.ones(2), size=tree.num_infosets)
    tables.append(t * tree.legal / (t * tree.legal).sum(1, keepdims=True))
  for table in tables:
    pol = _tabular(pyspiel, tree, table)
    want_values = pyspiel.expected_returns(
        game.new_initial_state(), [pol] * n, -1, True
    )
    np.testing.assert_allclose(
        tree.expected_returns(table), want_values, atol=1e-12
    )
    np.testing.assert_allclose(
        tree.nash_conv(table), pyspiel.nash_conv(game, pol, False), atol=1e-10
    )
  # uniform play is far from an equilibrium
  assert tree.nash_conv(tables[0]) > 0.1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
