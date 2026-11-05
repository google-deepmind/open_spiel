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

"""Meta-solvers at more than two sides.

Capabilities and the symmetric meta-game.

Known answers:
  * Capabilities: every two-player zero-sum solver (the LP ones included) is
    refused at three sides and for a general-sum two-sided game.
  * SymmetricGame.fitness is the exact expectation over iid co-players
    (brute-force sum over ordered profiles); at two players it is U @ sigma.
  * Replicator dynamics converge to the interior strict ESS p* = sqrt(2) - 1
    of a 3-player game with f_1 - f_0 = (1 - p)^2 - 2 p^2 (from uniform, which
    is not a rest point), and to uniform on nplayer_matrix
    odd_one_out_4p; on a weighted rock-paper-scissors game (fitness 2 A
    sigma, A antisymmetric, a conserved quantity, closed orbits) they raise
    NotConvergedError instead of reporting a point; fictitious play's average
    reaches its equilibrium (3, 2, 1) / 6.
  * alpha-rank: the stationary distribution of the 2-strategy monomorphic
    chain equals pi_1 / pi_0 = rho(0 -> 1) / rho(1 -> 0), each fixation
    probability computed here by enumerating co-player draws of an explicit
    population (independent of the hypergeometric formula); at two players it
    equals alpha_rank on the matrix.
"""

import itertools
import math
import sys

import numpy as np
import pytest

from open_spiel.python.scale.envs import nplayer_matrix as nm
from open_spiel.python.scale.league import meta_solvers as ms


# -- capabilities --------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ms.SOLVERS) - {"uniform"}))
def test_two_player_solvers_refuse_more_sides_and_general_sum(name):
  """Two player solvers refuse more sides and general sum."""
  p = np.array([[0.0, 1.0], [-1.0, 0.0]])
  with pytest.raises(ms.CapabilityError, match="3 sides"):
    ms.solve(name, p, num_sides=3)
  with pytest.raises(ms.CapabilityError, match="general-sum"):
    ms.solve(name, p, num_sides=2, constant_sum=False)
  ms.check_capability(name, 2, True)  # the two-player zero-sum case stands


def test_lp_refusal_names_the_lp_and_the_alternative():
  """Lp refusal names the lp and the alternative."""
  with pytest.raises(
      ms.CapabilityError, match="linear program.*symmetric meta-solver"
  ):
    ms.check_capability("nash", 4, True)
  ms.check_capability("uniform", 4, False)  # reads no payoff
  sym = ms.SymmetricGame.from_tensor(np.zeros((2, 2, 2)))
  with pytest.raises(ms.CapabilityError, match="symmetric"):
    ms.solve_symmetric("nash", sym)


# -- the symmetric game --------------------------------------------------------
def _rand_symmetric(k, t, seed):
  """A random symmetric payoff tensor (symmetric in the others).

  Args:
    k: The strategies.
    t: The players.
    seed: The seed.

  Returns:
    f64[k] * t.
  """
  rng = np.random.default_rng(seed)
  u = rng.normal(size=(k,) * t)
  sym = np.zeros_like(u)
  perms = list(itertools.permutations(range(1, t)))
  for perm in perms:
    sym += np.transpose(u, (0,) + perm)
  return sym / len(perms)


@pytest.mark.parametrize("k,t", [(3, 2), (3, 3), (2, 4), (4, 3)])
def test_fitness_is_the_iid_expectation(k, t):
  """Fitness is the iid expectation."""
  u = _rand_symmetric(k, t, k * 10 + t)
  g = ms.SymmetricGame.from_tensor(u)
  sigma = np.random.default_rng(1).dirichlet(np.ones(k))
  brute = np.zeros(k)
  for prof in itertools.product(range(k), repeat=t - 1):
    brute += np.prod(sigma[list(prof)]) * u[(slice(None),) + prof]
  np.testing.assert_allclose(g.fitness(sigma), brute, rtol=0, atol=1e-12)
  if t == 2:
    np.testing.assert_allclose(g.fitness(sigma), u @ sigma, atol=1e-12)
  sub = g.restrict([g.names[-1], g.names[0]])
  idx = [k - 1, 0]
  np.testing.assert_allclose(
      sub.table,
      ms.SymmetricGame.from_tensor(u[np.ix_(*([idx] * t))]).table,
      atol=0,
  )


def test_from_tensor_refuses_an_asymmetric_tensor():
  """From tensor refuses an asymmetric tensor."""
  u = np.zeros((2, 2, 2))
  u[0, 0, 1] = 1.0
  with pytest.raises(ValueError, match="not symmetric"):
    ms.SymmetricGame.from_tensor(u)


P_STAR = np.sqrt(2.0) - 1.0


def _ess_game():
  """A three-player game with a strict interior ESS.

  Returns:
    The SymmetricGame.
  """
  # u(0; .) = 0; u(1; a ones among 2 others) = (1, 0, -2)[a]:
  # f1 - f0 = (1 - p)^2 - 2 p^2, zero at p* = sqrt(2) - 1, decreasing: a strict
  # ESS.
  return ms.SymmetricGame.from_function(
      3, ("a", "b"), lambda i, m: 0.0 if i == 0 else (1.0, 0.0, -2.0)[sum(m)]
  )


def test_replicator_converges_to_the_interior_ess():
  """Replicator converges to the interior ess."""
  sol = ms.solve_symmetric("replicator", _ess_game())
  np.testing.assert_allclose(sol.sigma, [1.0 - P_STAR, P_STAR], atol=1e-5)
  assert sol.regret < 1e-8 and sol.iterations > 0


def test_replicator_on_odd_one_out_4p_is_uniform():
  """Replicator on odd one out 4p is uniform."""
  g = nm.GAMES["odd_one_out_4p"]
  # seat 0's payoff, co-players symmetric
  u = np.asarray(g.payoffs[..., 0], np.float64)
  sol = ms.solve_symmetric("replicator", ms.SymmetricGame.from_tensor(u))
  np.testing.assert_allclose(sol.sigma, [0.5, 0.5], atol=1e-9)


def _weighted_rps(a=1.0, b=2.0, c=3.0):
  """Weighted rock-paper-scissors, three players, and its equilibrium.

  Args:
    a: The first weight.
    b: The second weight.
    c: The third weight.

  Returns:
    (the SymmetricGame, its equilibrium f64[3]).
  """
  big_a = np.array([[0.0, -a, b], [a, 0.0, -c], [-b, c, 0.0]])
  u = big_a[:, :, None] + big_a[:, None, :]  # u(i; j, k) = A[i, j] + A[i, k]
  return ms.SymmetricGame.from_tensor(u), np.array([c, b, a]) / (a + b + c)


def test_replicator_reports_non_convergence_on_a_cyclic_game():
  """Replicator reports non convergence on a cyclic game."""
  g, _ = _weighted_rps()
  with pytest.raises(ms.NotConvergedError):
    ms.solve_symmetric("replicator", g, iterations=40000)


def test_fictitious_play_average_reaches_the_cyclic_equilibrium():
  """Fictitious play average reaches the cyclic equilibrium."""
  g, ne = _weighted_rps()
  assert g.regret(ne) == pytest.approx(0.0, abs=1e-12)
  sol = ms.solve_symmetric("fictitious_play", g, iterations=20000)
  np.testing.assert_allclose(sol.sigma, ne, atol=0.03)
  assert sol.regret < 0.05


# -- alpha-rank ----------------------------------------------------------------
def _brute_fitness(g, types, who):
  """Mean payoff of individual `who` over every draw of co-players.

  Every ordered draw of T-1 distinct co-players from the rest of the explicit
  population `types`.

  Args:
    g: the SymmetricGame.
    types: the explicit population, a strategy per individual.
    who: the individual's index.

  Returns:
    The mean payoff.
  """
  rest = [x for i, x in enumerate(types) if i != who]
  draws = list(itertools.permutations(range(len(rest)), g.num_players - 1))
  return np.mean([g.payoff(types[who], [rest[d] for d in dr]) for dr in draws])


def _brute_rho(g, r, s, alpha, m):
  """The Moran fixation probability of r into s, by brute force.

  Args:
    g: The SymmetricGame.
    r: The mutant.
    s: The resident.
    alpha: The selection intensity.
    m: The population size.

  Returns:
    The fixation probability.
  """
  total, prod = 1.0, 1.0
  for k in range(1, m):
    pop = [r] * k + [s] * (m - k)
    fr, fs = _brute_fitness(g, pop, 0), _brute_fitness(g, pop, m - 1)
    prod *= math.exp(-alpha * (fr - fs))
    total += prod
  return 1.0 / total


@pytest.mark.parametrize("m,alpha", [(3, 1.0), (5, 2.5), (6, 0.3)])
def test_alpha_rank_matches_the_two_strategy_chain(m, alpha):
  """Alpha rank matches the two strategy chain."""
  # u(i; a co-players playing 1) = table[3 i + a]
  table = (
      0.3,
      -0.2,
      0.9,
      1.4,
      0.1,
      -0.7,
  )
  g = ms.SymmetricGame.from_function(
      3, ("a", "b"), lambda i, ot: table[3 * i + sum(ot)]
  )
  sol = ms.solve_symmetric("alpha_rank", g, alpha=alpha, population_size=m)
  ratio = _brute_rho(g, 1, 0, alpha, m) / _brute_rho(g, 0, 1, alpha, m)
  np.testing.assert_allclose(
      sol.sigma, np.array([1.0, ratio]) / (1.0 + ratio), atol=1e-12
  )


def test_alpha_rank_at_two_players_is_the_matrix_alpha_rank():
  """Alpha rank at two players is the matrix alpha rank."""
  p = np.random.default_rng(3).normal(size=(4, 4))
  g = ms.SymmetricGame(2, "abcd", p)
  for m in (2, 7):
    a = ms.solve_symmetric("alpha_rank", g, alpha=3.0, population_size=m).sigma
    np.testing.assert_allclose(
        a,
        ms.solve("alpha_rank", p, alpha=3.0, population_size=m).row,
        atol=1e-12,
    )


def test_alpha_rank_refuses_a_population_smaller_than_the_table():
  """Alpha rank refuses a population smaller than the table."""
  with pytest.raises(ValueError, match="population_size 2 < 3"):
    ms.solve_symmetric("alpha_rank", _ess_game(), population_size=2)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
