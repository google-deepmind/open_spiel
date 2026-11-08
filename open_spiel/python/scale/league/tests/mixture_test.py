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

"""Population mixtures as behaviour policies, and symmetric payoffs.

Population mixtures as behaviour policies (Kuhn's theorem), and exact
symmetric payoffs, over eval.exploitability.GameTree.

Known answers: a mixture with all
weight on one table is that table; a mixture of identical tables is the
table; the mixture's behaviour table is realization-equivalent to the
mixture: against any fixed opponent its expected return equals the
weighted sum of the members' returns, in either seat (exact, f64); own
reach is the product of the seat's own action probabilities on the path.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.league import mixture


@pytest.fixture(name="tree", scope="module")
def _tree_fixture():
  """The Kuhn game tree.

  Returns:
    The GameTree.
  """
  return expl.GameTree(kuhn.KuhnEnv())


def _random_table(tree, seed):
  """A random policy table.

  Args:
    tree: The game tree.
    seed: The seed.

  Returns:
    The table.
  """
  rng = np.random.default_rng(seed)
  t = rng.random(tree.legal.shape) * tree.legal
  return t / t.sum(1, keepdims=True)


def _pure_table(tree, seed):
  """A random pure policy table.

  Args:
    tree: The game tree.
    seed: The seed.

  Returns:
    The table.
  """
  rng = np.random.default_rng(seed)
  t = np.zeros(tree.legal.shape)
  for r in range(t.shape[0]):
    acts = np.flatnonzero(tree.legal[r])
    t[r, rng.choice(acts)] = 1.0
  return t


def test_own_reach_known_values(tree):
  """Own reach known values."""
  always_bet = np.zeros(tree.legal.shape)
  always_bet[:, 1] = 1.0
  for p in (0, 1):
    r = mixture.own_reach(tree, always_bet, p)
    assert r.shape == (tree.num_infosets,) and r.dtype == np.float64
    rows = tree.player == p
    # Seat 0 never reaches its "pb" infosets (it never passes first); seat 1
    # reaches all of its first-decision infosets.
    for row in np.flatnonzero(rows):
      own_moves = [
          a
          for (is_ch, a), who in zip(tree.history(row), _actors(tree, row))
          if not is_ch and who == p
      ]
      assert r[row] == (1.0 if all(a == 1 for a in own_moves) else 0.0)
    assert np.all(r[~rows] == 0.0)


def _actors(tree, row):
  """The actor at each step of infoset row's first history.

  Args:
    tree: The game tree.
    row: The infoset row.

  Returns:
    The actors.
  """
  n = int(tree.members[row][0])
  path = []
  while tree.parent[n] >= 0:
    path.append(int(tree.actor[tree.parent[n]]))
    n = int(tree.parent[n])
  return list(reversed(path))


def test_degenerate_mixtures(tree):
  """Degenerate mixtures."""
  a, b = _random_table(tree, 0), _random_table(tree, 1)
  np.testing.assert_allclose(
      mixture.mixture_table(tree, [a, b], [1.0, 0.0]), a, atol=1e-15
  )
  np.testing.assert_allclose(
      mixture.mixture_table(tree, [a, a], [0.3, 0.7]), a, atol=1e-15
  )
  with pytest.raises(ValueError):
    mixture.mixture_table(tree, [a, b], [0.5, 0.6])


@pytest.mark.parametrize("make", [_random_table, _pure_table])
def test_mixture_is_realization_equivalent(tree, make):
  """Mixture is realization equivalent."""
  tables = [make(tree, s) for s in range(4)]
  w = np.array([0.1, 0.2, 0.3, 0.4])
  mix = mixture.mixture_table(tree, tables, w)
  tree.check_table(mix)
  for opp_seed in (10, 11):
    opp = _random_table(tree, opp_seed)
    for seat in (0, 1):

      def seats(t, opp=opp, seat=seat):
        """The seat tables with `t` at `seat`.

        Args:
          t: The table.
          opp: The opponent's table.
          seat: The seat of `t`.

        Returns:
          [seat 0 table, seat 1 table].
        """
        return [t, opp] if seat == 0 else [opp, t]

      want = sum(
          wk * tree.expected_returns(seats(t))[seat] for wk, t in zip(w, tables)
      )
      got = tree.expected_returns(seats(mix))[seat]
      assert got == pytest.approx(want, abs=1e-12)


def test_symmetric_payoff_is_antisymmetric_and_exact(tree):
  """Symmetric payoff is antisymmetric and exact."""
  a, b = _random_table(tree, 2), _random_table(tree, 3)
  ab = mixture.symmetric_payoff(tree, a, b)
  ba = mixture.symmetric_payoff(tree, a=b, b=a)  # the seats swapped
  assert ab == pytest.approx(-ba, abs=1e-12)
  want = 0.5 * (
      tree.expected_returns([a, b])[0] + tree.expected_returns([b, a])[1]
  )
  assert ab == pytest.approx(want, abs=1e-15)
  assert mixture.symmetric_payoff(tree, a, a) == pytest.approx(0.0, abs=1e-12)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
