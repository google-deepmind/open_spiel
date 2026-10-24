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

"""Meta-solvers on known matrices.

Every solver takes the row
player's payoff matrix of a two-player zero-sum meta-game (the column player
gets -P) and returns a MetaSolution(row, col, value) of f64 strategies.

Known answers:
  * rock-paper-scissors and matching pennies: Nash is uniform, value 0;
  * generalised RPS [[0,-a,b],[a,0,-c],[-b,c,0]]: Nash x ~ (c, b, a);
  * a 2x2 game [[2,-1],[-1,1]]: row = col = (2/5, 3/5), value 1/5;
  * a dominant-strategy game: Nash, FP, RM, PRD and alpha-rank put (almost)
    all mass on the dominant strategy;
  * alpha-rank: uniform on RPS (symmetry); the two-strategy closed form
    pi_0 / pi_1 = rho(0 invades 1) / rho(1 invades 0); a transitive game
    ranks by strength.
"""

import math
import sys

import numpy as np
import pytest

from open_spiel.python.scale.league import meta_solvers

RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
# a=1 b=2 c=1
BIASED_RPS = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
BIASED_NASH = np.array([0.25, 0.5, 0.25])
TWO_BY_TWO = np.array([[2.0, -1.0], [-1.0, 1.0]])
# 0 dominates
DOMINANT = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]])
UNIFORM3 = np.full(3, 1.0 / 3.0)


def _is_dist(x, n):
  x = np.asarray(x)
  assert x.dtype == np.float64 and x.shape == (n,)
  assert np.all(x >= 0.0) and abs(x.sum() - 1.0) < 1e-9


def test_every_solver_is_registered_and_returns_f64_distributions():
  """Every solver is registered and returns f64 distributions."""
  assert set(meta_solvers.SOLVERS) == {
      "uniform",
      "nash",
      "fictitious_play",
      "regret_matching",
      "alpha_rank",
      "prd",
      "max_entropy_nash",
  }
  for name in meta_solvers.SOLVERS:
    sol = meta_solvers.solve(name, RPS)
    assert isinstance(sol, meta_solvers.MetaSolution), name
    _is_dist(sol.row, 3)
    _is_dist(sol.col, 3)


def test_unknown_solver_and_bad_matrices_fail_loud():
  """Unknown solver and bad matrices fail loud."""
  with pytest.raises(KeyError):
    meta_solvers.solve("nope", RPS)
  with pytest.raises(ValueError):
    meta_solvers.nash(np.array([[0.0, np.nan], [1.0, 0.0]]))
  with pytest.raises(ValueError):
    meta_solvers.nash(np.zeros((3,)))
  with pytest.raises(ValueError):
    # alpha-rank is single-population: square only
    meta_solvers.alpha_rank(np.zeros((2, 3)))


def test_uniform():
  """Uniform."""
  sol = meta_solvers.uniform(np.zeros((4, 2)))
  np.testing.assert_array_equal(sol.row, np.full(4, 0.25))
  np.testing.assert_array_equal(sol.col, np.full(2, 0.5))
  assert sol.value is None


@pytest.mark.parametrize(
    "payoff,row,col,value",
    [
        (RPS, UNIFORM3, UNIFORM3, 0.0),
        (PENNIES, np.full(2, 0.5), np.full(2, 0.5), 0.0),
        (BIASED_RPS, BIASED_NASH, BIASED_NASH, 0.0),
        (TWO_BY_TWO, np.array([0.4, 0.6]), np.array([0.4, 0.6]), 0.2),
        (DOMINANT, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0),
    ],
)
def test_nash_lp_known_answers(payoff, row, col, value):
  """Nash lp known answers."""
  sol = meta_solvers.nash(payoff)
  np.testing.assert_allclose(sol.row, row, atol=1e-8)
  np.testing.assert_allclose(sol.col, col, atol=1e-8)
  assert sol.value == pytest.approx(value, abs=1e-8)
  assert meta_solvers.nash_conv(payoff, sol.row, sol.col) == pytest.approx(
      0.0, abs=1e-8
  )


def test_nash_non_square():
  # Row has a dominated third strategy (row 2 is row 0 minus 1); column mixes.
  """Nash non square."""
  p = np.array([[2.0, -1.0], [-1.0, 1.0], [1.0, -2.0]])
  sol = meta_solvers.nash(p)
  np.testing.assert_allclose(sol.row, [0.4, 0.6, 0.0], atol=1e-8)
  np.testing.assert_allclose(sol.col, [0.4, 0.6], atol=1e-8)
  assert sol.value == pytest.approx(0.2, abs=1e-8)


def test_nash_conv_known_values():
  # Pure rock against RPS: the paper best response wins 1, the column's best
  # reply
  # to rock is paper (-1 for row): nash_conv = max_i(P col)_i - min_j(row P)_j.
  """Nash conv known values."""
  rock = np.array([1.0, 0.0, 0.0])
  assert meta_solvers.nash_conv(RPS, rock, rock) == pytest.approx(2.0)
  assert meta_solvers.nash_conv(RPS, UNIFORM3, UNIFORM3) == pytest.approx(
      0.0, abs=1e-12
  )


@pytest.mark.parametrize("name", ["fictitious_play", "regret_matching"])
@pytest.mark.parametrize(
    "payoff,nash",
    [
        (RPS, UNIFORM3),
        (BIASED_RPS, BIASED_NASH),
        (TWO_BY_TWO, np.array([0.4, 0.6])),
    ],
)
def test_iterative_nash_solvers_converge(name, payoff, nash):
  # Fictitious play's averages converge at about t**-0.5 (regret matching+
  # much faster), so FP gets the looser bound at the same 20k iterations.
  """Iterative nash solvers converge."""
  tol = 3e-2 if name == "fictitious_play" else 1e-2
  sol = meta_solvers.solve(name, payoff, iterations=20000)
  np.testing.assert_allclose(sol.row, nash, atol=tol)
  np.testing.assert_allclose(sol.col, nash, atol=tol)
  assert meta_solvers.nash_conv(payoff, sol.row, sol.col) < tol


@pytest.mark.parametrize(
    "name", ["fictitious_play", "regret_matching", "prd", "alpha_rank"]
)
def test_dominant_strategy_gets_the_mass(name):
  """Dominant strategy gets the mass."""
  sol = meta_solvers.solve(name, DOMINANT)
  assert sol.row[0] > 0.99, (name, sol.row)
  assert sol.col[0] > 0.99, (name, sol.col)


def test_iterative_solvers_are_deterministic():
  """Iterative solvers are deterministic."""
  for name in ("fictitious_play", "regret_matching", "prd", "alpha_rank"):
    a, b = meta_solvers.solve(name, BIASED_RPS), meta_solvers.solve(
        name, BIASED_RPS
    )
    np.testing.assert_array_equal(a.row, b.row)
    np.testing.assert_array_equal(a.col, b.col)


def test_prd_time_average_on_rps_and_biased_rps():
  # Replicator orbits cycle around an interior equilibrium; their time
  # average converges to it (PRD averages its iterates).
  """Prd time average on rps and biased rps."""
  np.testing.assert_allclose(meta_solvers.prd(RPS).row, UNIFORM3, atol=2e-2)
  np.testing.assert_allclose(
      meta_solvers.prd(BIASED_RPS).row, BIASED_NASH, atol=3e-2
  )


def test_prd_gamma_floor():
  """Prd gamma floor."""
  sol = meta_solvers.prd(DOMINANT, gamma=0.03)
  assert np.all(sol.row >= 0.01 - 1e-12)  # gamma / n = 0.01 floor per strategy


def test_max_entropy_nash():
  """Max entropy nash."""
  np.testing.assert_allclose(
      meta_solvers.max_entropy_nash(RPS).row, UNIFORM3, atol=1e-5
  )
  # A duplicated strategy: the plain LP may put all mass on one copy; the
  # entropy-regularised maxmin spreads it over both.
  dup = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-1.0, -1.0, 0.0]])
  sol = meta_solvers.max_entropy_nash(dup)
  assert abs(sol.row[0] - sol.row[1]) < 1e-4 and sol.row[2] < 0.2


def test_alpha_rank_rps_is_uniform():
  """Alpha rank rps is uniform."""
  for m in (2, 10, 50):
    np.testing.assert_allclose(
        meta_solvers.alpha_rank(RPS, alpha=5.0, population_size=m).row,
        UNIFORM3,
        atol=1e-9,
    )


def _fixation(p, r, s, alpha, m):
  """Closed form: the fixation probability of one r mutant among s players.

  Args:
    p: The payoff matrix.
    r: The mutant strategy.
    s: The resident strategy.
    alpha: The selection intensity.
    m: The population size.

  Returns:
    The fixation probability.
  """
  total = 1.0
  acc = 0.0
  for l in range(1, m):
    k = l
    fr = ((k - 1) * p[r, r] + (m - k) * p[r, s]) / (m - 1)
    fs = (k * p[s, r] + (m - k - 1) * p[s, s]) / (m - 1)
    acc += -alpha * (fr - fs)
    total += math.exp(acc)
  return 1.0 / total


@pytest.mark.parametrize("m", [2, 5, 20])
def test_alpha_rank_two_strategy_closed_form(m):
  """Alpha rank two strategy closed form."""
  p = np.array([[0.3, -0.2], [0.5, 0.1]])  # general-sum symmetric 2x2
  alpha = 1.7
  pi = meta_solvers.alpha_rank(p, alpha=alpha, population_size=m).row
  ratio = _fixation(p, 0, 1, alpha, m) / _fixation(p, 1, 0, alpha, m)
  assert pi[0] / pi[1] == pytest.approx(ratio, rel=1e-9)


def test_alpha_rank_neutral_game_is_uniform_and_transitive_game_ranks():
  """Alpha rank neutral game is uniform and transitive game ranks."""
  np.testing.assert_allclose(
      meta_solvers.alpha_rank(np.zeros((4, 4))).row,
      np.full(4, 0.25),
      atol=1e-12,
  )
  # Transitive: i beats j by (j - i) for i < j. Higher strength -> more mass.
  n = 4
  p = np.array([[float(j - i) for j in range(n)] for i in range(n)])
  pi = meta_solvers.alpha_rank(p, alpha=0.5, population_size=10).row
  assert np.all(np.diff(pi) < 0), pi
  assert meta_solvers.alpha_rank(p, alpha=50.0).row[0] > 0.999


def test_alpha_rank_huge_alpha_does_not_overflow():
  """Alpha rank huge alpha does not overflow."""
  pi = meta_solvers.alpha_rank(
      DOMINANT * 1e3, alpha=1e3, population_size=50
  ).row
  assert np.all(np.isfinite(pi)) and pi[0] > 0.999


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
