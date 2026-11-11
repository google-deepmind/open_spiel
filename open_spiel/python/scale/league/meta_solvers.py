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

"""Meta-solvers: a meta-game payoff matrix -> meta-strategies.

PSRO (Lanctot et al. 2017) mixes its population by a meta-strategy computed
from the population's payoff matrix; the choice of solver is a parameter of
the algorithm. Every solver here takes the row player's payoff matrix P
(f64[n, m]) of a two-player zero-sum meta-game (the column player gets -P;
a payoff matrix over one population is antisymmetric, so row and column
strategies coincide) and returns MetaSolution(row f64[n], col f64[m], value),
value being the row player's game value when the solver computes one.

    sol = meta_solvers.solve("nash", payoff)        # MetaSolution
    meta_solvers.nash_conv(payoff, sol.row, sol.col)  # 0 at an equilibrium

Solvers (pure numpy and scipy, f64, deterministic):
  uniform            1/n each.
  nash               the maxmin linear program (scipy HiGHS), exact.
  fictitious_play    simultaneous fictitious play, average strategies (ties:
                     lowest index).
  regret_matching    alternating regret matching+ with linear averaging.
  alpha_rank         single-population alpha-rank (Omidshafiei et al. 2019):
                     the monomorphic Markov chain whose transitions are the
                     fixation probabilities of a mutant in a population of
                     `population_size` under Fermi selection of strength
                     `alpha`; square P only.
  prd                projected replicator dynamics (Lanctot et al. 2017):
                     Euler steps of the two-population replicator equation,
                     projected onto {x : x_i >= gamma / n}; returns the
                     average of the last `average_last` iterates.
  max_entropy_nash   maxmin with an entropy bonus: maximise v + temperature *
                     H(x) under the maxmin constraints (SLSQP), payoffs scaled
                     by max |P|.

`nash_conv(P, row, col)` = max_i (P col)_i - min_j (row P)_j is the
meta-game exploitability, 0 exactly at a Nash equilibrium of the meta-game;
it is not the exploitability in the underlying game.

Capabilities (`check_capability(name, num_sides, constant_sum)`). Every
solver above reads P as a two-player zero-sum game, so each is valid only
at two sides of a constant-sum game (the league centres a constant sum
into an antisymmetric matrix). More sides, or general-sum two sides, have no
maxmin value: the LP solvers (nash, max_entropy_nash) and the other
two-player solvers are refused there, loudly. What remains is the
symmetric meta-game below, whose solutions are labelled relative to their
field sigma, never an equilibrium claim of the underlying game.

Symmetric meta-games (any number of sides T >= 2, general-sum allowed).
A `SymmetricGame` over K strategies is the heuristic payoff table of a
symmetric T-player game: u(i; M), the expected side return of strategy i
when the other T - 1 sides hold the multiset M (a sorted tuple of strategy
indices), averaged over the side positions i holds and over the
arrangements of M (a rotation-balanced design makes a sampled table
symmetric; hearts, for instance, is symmetric only after that average).
Its `fitness(sigma)` f64[K] is vs_field(i, sigma): i against T - 1 sides
drawn iid from sigma. At T = 2 the table is a K x K matrix and fitness is
U @ sigma.

  sym = SymmetricGame.from_tensor(u)         # u f64[K]*T, seat 0's payoff
  sol = meta_solvers.solve_symmetric("replicator", sym)   # SymmetricSolution
  sol.sigma, sol.regret                      # regret: max_i f_i - sigma . f

  replicator         Euler steps of the symmetric replicator equation
                     x_i' = x_i (f_i(x) - x . f(x)) from uniform; converges
                     only to an asymptotically stable rest point (a strict
                     ESS, for instance). If the iterates still move after
                     `iterations`, or stop at a point with regret above
                     `regret_tol`, it raises NotConvergedError (a cyclic game,
                     rock-paper-scissors-like, has no point to report).
  fictitious_play    symmetric fictitious play: sigma_t is the average of
                     the best responses to sigma_0..sigma_{t-1} (ties: lowest
                     index); returns the average with its regret (not
                     raised: an average is a well-defined answer even when
                     play cycles, and its regret says how far from a
                     symmetric equilibrium it is).
  alpha_rank         single-population alpha-rank with T-player fitness
                     (Omidshafiei et al. 2019): a mutant r among m - k
                     residents s meets T - 1 co-players drawn without
                     replacement from the other m - 1 individuals; fixation
                     under Fermi selection of strength alpha; the stationary
                     distribution of the monomorphic chain. Needs
                     population_size >= T. At T = 2 equal to alpha_rank on
                     the matrix.
  uniform            1/K each.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np
from scipy import optimize

F64 = np.float64


class MetaSolution(NamedTuple):
  """A meta-game solution.

  Attributes:
    row: the row player's strategy, f64.
    col: the column player's strategy, f64.
    value: the row player's game value, or None when the solver has none.
  """

  row: np.ndarray  # f64[n]
  col: np.ndarray  # f64[m]
  value: Optional[float]  # the row player's value, when the solver computes it


def _check(payoff, square: bool = False) -> np.ndarray:
  """The payoff as a validated f64 matrix.

  Args:
    payoff: the row player's payoff matrix.
    square: whether the solver needs a square matrix.

  Returns:
    The f64 matrix.

  Raises:
    ValueError: the payoff is not a non-empty 2-D matrix, has a non-finite
      entry, or is not square when `square`.
  """
  p = np.asarray(payoff, F64)
  if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1:
    raise ValueError(
        f"payoff must be a non-empty 2-D matrix, got shape {p.shape}"
    )
  if not np.all(np.isfinite(p)):
    raise ValueError("payoff has a non-finite entry")
  if square and p.shape[0] != p.shape[1]:
    raise ValueError(
        f"this solver needs a square (single-population) matrix, got {p.shape}"
    )
  return p


def _normalize(x: np.ndarray, what: str) -> np.ndarray:
  """Normalises nonnegative masses to sum to 1.

  Args:
    x: The masses.
    what: The caller, for errors.

  Returns:
    The distribution.

  Raises:
    RuntimeError: The total mass is not positive.
  """
  x = np.maximum(np.asarray(x, F64), 0.0)
  s = x.sum()
  if not np.isfinite(s) or s <= 0.0:
    raise RuntimeError(f"{what}: non-positive total mass")
  return x / s


def nash_conv(payoff, row, col) -> float:
  """max_i (P col)_i - min_j (row P)_j >= 0; 0 at a Nash equilibrium.

  Args:
    payoff: the row player's payoff matrix P.
    row: the row strategy.
    col: the column strategy.

  Returns:
    The NashConv.
  """
  p = _check(payoff)
  row, col = np.asarray(row, F64), np.asarray(col, F64)
  assert row.shape == (p.shape[0],) and col.shape == (p.shape[1],), (
      row.shape,
      col.shape,
  )
  return float(np.max(p @ col) - np.min(row @ p))


def uniform(payoff) -> MetaSolution:
  """Uniform strategies for both players.

  Args:
    payoff: the row player's payoff matrix.

  Returns:
    The MetaSolution.
  """
  n, m = _check(payoff).shape
  return MetaSolution(np.full(n, 1.0 / n), np.full(m, 1.0 / m), None)


def _maxmin_lp(p: np.ndarray) -> tuple[np.ndarray, float]:
  """argmax_x min_j (x P)_j over the simplex, and its value.

  Args:
    p: f64[n, m], the payoff matrix.

  Returns:
    (the maxmin strategy, its value).

  Raises:
    RuntimeError: The LP fails.
  """
  n, m = p.shape
  c = np.zeros(n + 1)
  c[-1] = -1.0
  a_ub = np.column_stack([-p.T, np.ones(m)])  # v - (x P)_j <= 0
  a_eq = np.zeros((1, n + 1))
  a_eq[0, :n] = 1.0
  res = optimize.linprog(
      c,
      A_ub=a_ub,
      b_ub=np.zeros(m),
      A_eq=a_eq,
      b_eq=np.ones(1),
      bounds=[(0.0, None)] * n + [(None, None)],
      method="highs",
  )
  if not res.success:
    raise RuntimeError(f"maxmin LP failed: {res.message}")
  return _normalize(res.x[:n], "maxmin LP"), float(res.x[-1])


def nash(payoff) -> MetaSolution:
  """A Nash equilibrium of the zero-sum matrix game, by linear programming.

  Args:
    payoff: the row player's payoff matrix.

  Returns:
    The MetaSolution, with the game value.
  """
  p = _check(payoff)
  row, value = _maxmin_lp(p)
  col, _ = _maxmin_lp(-p.T)
  return MetaSolution(row, col, value)


def fictitious_play(payoff, iterations: int = 20000) -> MetaSolution:
  """Simultaneous fictitious play from uniform beliefs; returns the averages.

  Args:
    payoff: the row player's payoff matrix.
    iterations: the number of iterations.

  Returns:
    The MetaSolution of the average strategies.
  """
  p = _check(payoff)
  assert iterations >= 1, iterations
  n, m = p.shape
  row_counts, col_counts = np.zeros(n), np.zeros(m)
  row_avg, col_avg = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
  for _ in range(int(iterations)):
    i = int(np.argmax(p @ col_avg))  # first maximal index
    j = int(np.argmin(row_avg @ p))
    row_counts[i] += 1.0
    col_counts[j] += 1.0
    row_avg = row_counts / row_counts.sum()
    col_avg = col_counts / col_counts.sum()
  return MetaSolution(row_avg, col_avg, None)


def regret_matching(payoff, iterations: int = 20000) -> MetaSolution:
  """Alternating regret matching+ with linearly weighted averages.

  Args:
    payoff: the row player's payoff matrix.
    iterations: the number of iterations.

  Returns:
    The MetaSolution of the average strategies.
  """
  p = _check(payoff)
  assert iterations >= 1, iterations
  n, m = p.shape
  rr, rc = np.zeros(n), np.zeros(m)
  sum_r, sum_c = np.zeros(n), np.zeros(m)

  def strat(reg, k):
    """The regret-matching strategy of cumulative regrets `reg`.

    Args:
      reg: the cumulative regrets.
      k: the number of strategies.

    Returns:
      The positive regrets normalised; uniform when none is positive.
    """
    pos = np.maximum(reg, 0.0)
    s = pos.sum()
    return pos / s if s > 0.0 else np.full(k, 1.0 / k)

  for t in range(1, int(iterations) + 1):
    x = strat(rr, n)
    y = strat(rc, m)
    u = p @ y
    rr = np.maximum(rr + u - x @ u, 0.0)
    x = strat(rr, n)
    v = -(x @ p)
    rc = np.maximum(rc + v - v @ y, 0.0)
    y = strat(rc, m)
    sum_r += t * x
    sum_c += t * y
  return MetaSolution(
      _normalize(sum_r, "regret_matching"),
      _normalize(sum_c, "regret_matching"),
      None,
  )


def _log_fixation(p: np.ndarray, r: int, s: int, alpha: float, m: int) -> float:
  """Log of the fixation probability of one r mutant in an s population of m.

  Args:
    p: The payoff matrix.
    r: The mutant strategy.
    s: The resident strategy.
    alpha: The selection intensity.
    m: The population size.

  Returns:
    The log fixation probability.
  """
  k = np.arange(1, m, dtype=F64)
  fr = ((k - 1.0) * p[r, r] + (m - k) * p[r, s]) / (m - 1.0)
  fs = (k * p[s, r] + (m - k - 1.0) * p[s, s]) / (m - 1.0)
  # log of 1 + sum_l prod_k
  terms = np.concatenate([[0.0], np.cumsum(-alpha * (fr - fs))])
  return -float(np.logaddexp.reduce(terms))


def alpha_rank(
    payoff, alpha: float = 10.0, population_size: int = 50
) -> MetaSolution:
  """Alpha-Rank's stationary distribution of a symmetric meta-game.

  Args:
    payoff: a square payoff matrix.
    alpha: the selection intensity.
    population_size: the evolutionary population size.

  Returns:
    The MetaSolution (the same distribution for both players).
  """
  p = _check(payoff, square=True)
  assert alpha >= 0.0 and np.isfinite(alpha), alpha
  m = int(population_size)
  assert m >= 2, population_size
  n = p.shape[0]
  if n == 1:
    return MetaSolution(np.ones(1), np.ones(1), None)
  eta = 1.0 / (n - 1)
  t = np.zeros((n, n))
  for s in range(n):
    for r in range(n):
      if r != s:
        t[s, r] = eta * np.exp(_log_fixation(p, r, s, alpha, m))
    t[s, s] = 1.0 - t[s].sum()
  # Stationary distribution: pi (T - I) = 0, sum pi = 1 (least squares, exact
  # for an irreducible chain; a reducible limit picks the minimum-norm one).
  a = np.vstack([t.T - np.eye(n), np.ones((1, n))])
  b = np.concatenate([np.zeros(n), [1.0]])
  pi, *_ = np.linalg.lstsq(a, b, rcond=None)
  pi = _normalize(pi, "alpha_rank")
  return MetaSolution(pi, pi.copy(), None)


def _project(x: np.ndarray, floor: float) -> np.ndarray:
  """Euclidean projection onto {y : y_i >= floor, sum y = 1}.

  Args:
    x: The vector.
    floor: The lower bound of every entry.

  Returns:
    The projection.
  """
  n = x.shape[0]
  assert floor * n <= 1.0 + 1e-12, (floor, n)
  z = x - floor
  budget = 1.0 - floor * n
  u = np.sort(z)[::-1]
  css = np.cumsum(u) - budget
  ks = np.arange(1, n + 1)
  rho = int(np.nonzero(u - css / ks > 0)[0][-1]) if budget > 0 else 0
  theta = css[rho] / (rho + 1) if budget > 0 else np.max(z)
  return np.maximum(z - theta, 0.0) + floor


def prd(
    payoff,
    iterations: int = 50000,
    dt: float = 1e-2,
    gamma: float = 1e-6,
    average_last: Optional[int] = None,
) -> MetaSolution:
  """Projected replicator dynamics, averaged over the last iterations.

  Args:
    payoff: the row player's payoff matrix.
    iterations: the number of steps.
    dt: the step size.
    gamma: the exploration floor of each strategy's mass.
    average_last: the number of final steps averaged; half the steps when
      None.

  Returns:
    The MetaSolution of the averaged strategies.
  """
  p = _check(payoff)
  assert iterations >= 1 and dt > 0.0 and 0.0 <= gamma < 1.0
  n, m = p.shape
  last = int(iterations) // 2 if average_last is None else int(average_last)
  assert 1 <= last <= iterations, (last, iterations)
  x, y = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
  sx, sy = np.zeros(n), np.zeros(m)
  for it in range(int(iterations)):
    ux, uy = p @ y, -(x @ p)
    vx, vy = x @ ux, y @ uy
    x = _project(x + dt * x * (ux - vx), gamma / n)
    y = _project(y + dt * y * (uy - vy), gamma / m)
    if it >= iterations - last:
      sx += x
      sy += y
  return MetaSolution(_normalize(sx, "prd"), _normalize(sy, "prd"), None)


def _max_entropy_maxmin(p: np.ndarray, temperature: float) -> np.ndarray:
  """The row player's entropy-regularized maxmin strategy.

  Args:
    p: the row player's payoff matrix.
    temperature: the entropy weight.

  Returns:
    The row strategy.

  Raises:
    RuntimeError: the optimizer fails.
  """
  n, m = p.shape
  scale = float(np.max(np.abs(p)))
  sp = p / scale if scale > 0.0 else p
  tiny = 1e-12
  x0 = np.concatenate(
      [np.full(n, 1.0 / n), [float(np.min(np.full(n, 1.0 / n) @ sp))]]
  )

  def obj(z):
    """The objective: -v + temperature * sum x log x.

    Args:
      z: the strategy x followed by the value v.

    Returns:
      The objective value.
    """
    x = np.clip(z[:n], tiny, 1.0)
    return -float(z[-1]) + temperature * float(np.sum(x * np.log(x)))

  def grad(z):
    """The gradient of `obj`.

    Args:
      z: the strategy x followed by the value v.

    Returns:
      The gradient.
    """
    x = np.clip(z[:n], tiny, 1.0)
    g = np.zeros_like(z)
    g[:n] = temperature * (np.log(x) + 1.0)
    g[-1] = -1.0
    return g

  cons = [{
      "type": "eq",
      "fun": lambda z: np.sum(z[:n]) - 1.0,
      "jac": lambda z: np.concatenate([np.ones(n), [0.0]]),
  }]
  for j in range(m):
    cons.append({
        "type": "ineq",
        "fun": lambda z, j=j: float(z[:n] @ sp[:, j] - z[-1]),
        "jac": lambda z, j=j: np.concatenate([sp[:, j], [-1.0]]),
    })
  res = optimize.minimize(
      obj,
      x0,
      jac=grad,
      bounds=[(tiny, 1.0)] * n + [(None, None)],
      constraints=cons,
      method="SLSQP",
      options={"ftol": 1e-12, "maxiter": 1000},
  )
  if not res.success:
    raise RuntimeError(f"max-entropy maxmin failed: {res.message}")
  return _normalize(res.x[:n], "max_entropy_nash")


def max_entropy_nash(payoff, temperature: float = 1.0) -> MetaSolution:
  """An entropy-regularised maxmin strategy for each player.

  Args:
    payoff: the row player's payoff matrix.
    temperature: the entropy weight.

  Returns:
    The MetaSolution, valued at the row strategy's worst case.
  """
  p = _check(payoff)
  assert temperature > 0.0, temperature
  row = _max_entropy_maxmin(p, temperature)
  col = _max_entropy_maxmin(-p.T, temperature)
  return MetaSolution(row, col, float(np.min(row @ p)))


SOLVERS: Mapping[str, Callable[..., MetaSolution]] = {
    "uniform": uniform,
    "nash": nash,
    "fictitious_play": fictitious_play,
    "regret_matching": regret_matching,
    "alpha_rank": alpha_rank,
    "prd": prd,
    "max_entropy_nash": max_entropy_nash,
}


def solve(
    name: str, payoff, num_sides: int = 2, constant_sum: bool = True, **kwargs
) -> MetaSolution:
  """Runs the meta-solver `name` and checks its distributions.

  SOLVERS[name] on the two-player zero-sum matrix `payoff`; num_sides and
  constant_sum describe the game it was reduced from (check_capability).

  Args:
    name: a key of SOLVERS.
    payoff: the row player's payoff matrix.
    num_sides: the sides of the game the matrix was reduced from.
    constant_sum: whether that game is constant-sum.
    **kwargs: passed to the solver.

  Returns:
    The MetaSolution.

  Raises:
    KeyError: an unknown solver.
  """
  if name not in SOLVERS:
    raise KeyError(f"unknown meta-solver {name!r}; known: {sorted(SOLVERS)}")
  check_capability(name, num_sides, constant_sum)
  sol = SOLVERS[name](payoff, **kwargs)
  for x in (sol.row, sol.col):
    assert x.dtype == F64 and np.all(x >= 0.0) and abs(x.sum() - 1.0) < 1e-9, (
        name,
        x,
    )
  return sol


# -- capabilities --------------------------------------------------------------
#: The solvers that compute a maxmin value (linear programs over the zero-sum
#: matrix): meaningless without a two-player zero-sum value.
LP_SOLVERS = ("nash", "max_entropy_nash")
#: Solvers of the symmetric T-player meta-game (SymmetricGame), any T >= 2.
SYMMETRIC_SOLVERS = ("uniform", "replicator", "fictitious_play", "alpha_rank")


class CapabilityError(ValueError):
  """A meta-solver asked for a game it has no meaning for (module doc)."""


def check_capability(name: str, num_sides: int, constant_sum: bool) -> None:
  """Raises unless SOLVERS[name] is valid for the game.

  SOLVERS[name] is a two-player zero-sum matrix solver; the game has
  num_sides sides, constant-sum or not (module doc, "Capabilities").

  Args:
    name: a key of SOLVERS.
    num_sides: the sides of the game the matrix was reduced from.
    constant_sum: whether that game is constant-sum.

  Raises:
    KeyError: an unknown solver.
    ValueError: fewer than two sides.
    CapabilityError: the solver has no meaning for the game.
  """
  if name not in SOLVERS:
    raise KeyError(f"unknown meta-solver {name!r}; known: {sorted(SOLVERS)}")
  t = int(num_sides)
  if t < 2:
    raise ValueError(
        f"num_sides {num_sides}: a meta-game has at least two sides"
    )
  if name == "uniform":
    return
  kind = (
      "a linear program for the maxmin value"
      if name in LP_SOLVERS
      else "a two-player zero-sum matrix solver"
  )
  if t > 2:
    raise CapabilityError(
        f"meta-solver {name!r} is {kind}; a game of {t} sides has no "
        "two-player zero-sum meta-game (no unique value, no maxmin LP). Use a "
        f"symmetric meta-solver {SYMMETRIC_SOLVERS} on the SymmetricGame "
        "(vs_field payoffs), labelled relative to its field"
    )
  if not constant_sum:
    raise CapabilityError(
        f"meta-solver {name!r} is {kind}; this two-sided game is general-sum, "
        "so the matrix is not a zero-sum game and its maxmin is no "
        f"equilibrium. Use a symmetric meta-solver {SYMMETRIC_SOLVERS} on the "
        "SymmetricGame"
    )


# -- symmetric T-player meta-games ---------------------------------------------
class NotConvergedError(RuntimeError):
  """An iterative symmetric solver found no point to report (module doc)."""


def _multisets(k: int, r: int) -> list[Any]:
  """The size-r multisets of range(k), sorted.

  Args:
    k: The strategies.
    r: The multiset size.

  Returns:
    The multisets, as sorted tuples.
  """
  return list(itertools.combinations_with_replacement(range(k), r))


def _multinomial(ms: tuple[Any, ...]) -> int:
  """The number of orderings of multiset `ms`.

  Args:
    ms: A multiset, as a tuple of strategy indices.

  Returns:
    The multinomial coefficient.
  """
  out = math.factorial(len(ms))
  for c in np.bincount(np.asarray(ms, np.int64)).tolist() if ms else ():
    out //= math.factorial(c)
  return out


class SymmetricGame:
  """The heuristic payoff table of a symmetric T-player game (module doc).

  Attributes:
    num_players: T >= 2, the sides of the underlying table.
    names: the K strategies.
    others: the multisets of T - 1 co-player strategies, the table's columns
      in `_multisets(K, T - 1)` order.
    table: u f64[K, C]: u[i, c] = i's expected side return against the
      multiset others[c].
    se: None, or f64[K, C] standard errors of a sampled table.
  """

  def __init__(self, num_players: int, names: Sequence[str], table, se=None):
    """Builds the game.

    Args:
      num_players: T >= 2, the sides of the underlying table.
      names: the K strategies, distinct.
      table: u f64[K, C]: u[i, c] = i's expected side return against the
        multiset others[c].
      se: optional f64[K, C] standard errors of a sampled table.

    Raises:
      ValueError: fewer than 2 players, empty or repeated strategies, a table
        of another shape or a non-finite entry.
    """
    self.num_players = int(num_players)
    if self.num_players < 2:
      raise ValueError(
          f"a symmetric game has at least 2 players, got {num_players}"
      )
    self.names = tuple(names)
    k = len(self.names)
    if k < 1 or len(set(self.names)) != k:
      raise ValueError(
          f"strategies must be non-empty and distinct, got {self.names}"
      )
    self.others = tuple(_multisets(k, self.num_players - 1))
    u = np.asarray(table, F64)
    if u.shape != (k, len(self.others)):
      raise ValueError(
          f"table shape {u.shape}, want ({k}, {len(self.others)}) for {k} "
          f"strategies at {self.num_players} players"
      )
    if not np.all(np.isfinite(u)):
      raise ValueError("symmetric game table has a non-finite entry")
    self.table = u
    self.se = None if se is None else np.asarray(se, F64).reshape(u.shape)
    self._coef = np.asarray([_multinomial(m) for m in self.others], F64)
    self._counts = np.zeros((len(self.others), k), np.int64)
    for c, m in enumerate(self.others):
      for j in m:
        self._counts[c, j] += 1

  @property
  def num_strategies(self) -> int:
    """K, the number of strategies."""
    return len(self.names)

  @classmethod
  def num_cells(cls, num_strategies: int, num_players: int) -> int:
    """The entries of the table: K * C(K + T - 2, T - 1).

    Args:
      num_strategies: K.
      num_players: T.

    Returns:
      The number of cells.
    """
    k, t = int(num_strategies), int(num_players)
    return k * math.comb(k + t - 2, t - 1)

  @classmethod
  def from_tensor(
      cls, u, names: Optional[Sequence[str]] = None, atol: float = 1e-12
  ) -> "SymmetricGame":
    """From seat 0's payoff tensor u f64[K]*T (u[i, j_1, ..., j_{T-1}]).

    Args:
      u: the payoff tensor.
      names: the strategy names (default "0".."K-1").
      atol: the tolerance of the symmetry check.

    Returns:
      The SymmetricGame.

    Raises:
      ValueError: u is not K x ... x K, or not invariant under permuting the
        co-players.
    """
    u = np.asarray(u, F64)
    t, k = u.ndim, u.shape[0]
    if t < 2 or any(d != k for d in u.shape):
      raise ValueError(
          f"payoff tensor must be K x ... x K with T >= 2 axes, got {u.shape}"
      )
    for perm in itertools.permutations(range(1, t)):
      if not np.allclose(u, np.transpose(u, (0,) + perm), atol=atol, rtol=0.0):
        raise ValueError("payoff tensor is not symmetric in the co-players")
    names = (
        tuple(names) if names is not None else tuple(str(i) for i in range(k))
    )
    others = _multisets(k, t - 1)
    table = np.stack([[u[(i,) + m] for m in others] for i in range(k)])
    return cls(t, names, table)

  @classmethod
  def from_function(
      cls,
      num_players: int,
      names: Sequence[str],
      payoff_fn: Callable[[int, tuple[Any, ...]], float],
  ) -> "SymmetricGame":
    """table[i, c] = payoff_fn(i, others[c]).

    Args:
      num_players: T.
      names: the K strategies.
      payoff_fn: (i, multiset of co-players) -> i's expected side return.

    Returns:
      The SymmetricGame.
    """
    k = len(tuple(names))
    others = _multisets(k, int(num_players) - 1)
    return cls(
        num_players,
        names,
        [[float(payoff_fn(i, m)) for m in others] for i in range(k)],
    )

  def payoff(self, i: int, others: Sequence[int]) -> float:
    """Strategy i's side return against the co-players `others`.

    Args:
      i: the strategy index.
      others: the T - 1 co-players' strategy indices (any order).

    Returns:
      u(i; others).
    """
    m = tuple(sorted(int(x) for x in others))
    return float(self.table[int(i), self.others.index(m)])

  def _check_sigma(self, sigma) -> np.ndarray:
    """`sigma` as a checked mixture over the strategies.

    Args:
      sigma: A mixture.

    Returns:
      f64[num_strategies].

    Raises:
      ValueError: Not a distribution over the strategies.
    """
    x = np.asarray(sigma, F64)
    if (
        x.shape != (self.num_strategies,)
        or np.any(x < -1e-12)
        or abs(x.sum() - 1.0) > 1e-9
    ):
      raise ValueError(
          f"sigma {x} is not a distribution over {self.num_strategies} "
          "strategies"
      )
    return np.maximum(x, 0.0)

  def fitness(self, sigma) -> np.ndarray:
    """f_i = vs_field(i, sigma), i against T - 1 iid draws of sigma.

    Args:
      sigma: f64[K], a distribution over the strategies.

    Returns:
      f64[K].
    """
    x = self._check_sigma(sigma)
    prob = self._coef * np.prod(x[None, :] ** self._counts, axis=1)  # f64[C]
    return self.table @ prob

  def regret(self, sigma) -> float:
    """max_i f_i(sigma) - sigma . f(sigma) >= 0.

    0 at a symmetric Nash equilibrium of the meta-game.

    Args:
      sigma: f64[K], a distribution over the strategies.

    Returns:
      The regret.
    """
    x = self._check_sigma(sigma)
    f = self.fitness(x)
    return float(np.max(f) - x @ f)

  def restrict(self, names: Sequence[str]) -> "SymmetricGame":
    """The sub-game over a subset of the strategies (any order).

    Args:
      names: the strategies kept.

    Returns:
      The SymmetricGame.
    """
    idx = [self.names.index(n) for n in names]
    sub = _multisets(len(idx), self.num_players - 1)
    cols = [self.others.index(tuple(sorted(idx[j] for j in m))) for m in sub]
    se = None if self.se is None else self.se[np.ix_(idx, cols)]
    return SymmetricGame(
        self.num_players, names, self.table[np.ix_(idx, cols)], se
    )

  def matrix(self) -> np.ndarray:
    """The payoff matrix at T = 2 (U[i, j] = u(i; (j,))).

    Returns:
      f64[K, K].

    Raises:
      ValueError: more than two players.
    """
    if self.num_players != 2:
      raise ValueError(
          f"a {self.num_players}-player symmetric game has no payoff matrix"
      )
    return self.table.copy()


class SymmetricSolution(NamedTuple):
  """A symmetric meta-solver's answer.

  Attributes:
    sigma: f64[K]: the symmetric meta-strategy (every side draws iid from it).
    regret: max_i f_i(sigma) - sigma . f(sigma).
    iterations: iterations run (0 for closed-form solvers).
  """

  sigma: np.ndarray
  regret: float
  iterations: int


def replicator(
    game: SymmetricGame,
    iterations: int = 200000,
    dt: float = 0.05,
    window: int = 2000,
    move_tol: float = 1e-7,
    regret_tol: float = 1e-4,
) -> SymmetricSolution:
  """Symmetric replicator dynamics from uniform (module doc).

  Euler steps of x_i' = x_i (f_i - x . f) with payoffs scaled by max |u|.
  Stops once the L1 change over the last `window` steps is below move_tol.

  Args:
    game: the SymmetricGame.
    iterations: the most steps.
    dt: the Euler step.
    window: steps between the iterates compared.
    move_tol: the L1 change below which the dynamics stop.
    regret_tol: the largest regret (scaled payoffs) of a reported point.

  Returns:
    The SymmetricSolution.

  Raises:
    NotConvergedError: it never stops, or stops with regret above regret_tol.
  """
  assert iterations >= window >= 1 and dt > 0.0, (iterations, window, dt)
  k = game.num_strategies
  scale = float(np.max(np.abs(game.table)))
  x = np.full(k, 1.0 / k)
  if k == 1 or scale == 0.0:
    return SymmetricSolution(x, 0.0, 0)
  hist = [x]
  for it in range(1, int(iterations) + 1):
    f = game.fitness(x) / scale
    x = x + dt * x * (f - x @ f)
    x = np.maximum(x, 0.0)
    x = x / x.sum()
    hist.append(x)
    if len(hist) > window + 1:
      hist.pop(0)
    if it >= window and float(np.abs(x - hist[0]).sum()) < move_tol:
      reg = game.regret(x) / scale
      if reg > regret_tol:
        raise NotConvergedError(
            f"replicator: stopped at {np.round(x, 6)} after {it} steps with "
            f"regret {reg:.3g} > {regret_tol} (scaled): a rest point that is "
            "not a symmetric equilibrium"
        )
      return SymmetricSolution(x, float(game.regret(x)), it)
  raise NotConvergedError(
      f"replicator: no rest point after {iterations} steps: the iterate still "
      f"moves {float(np.abs(x - hist[0]).sum()):.3g} (L1) per {window} steps "
      f"at {np.round(x, 4)} (a cyclic meta-game has no point for replicator "
      "dynamics to report; use fictitious_play, whose average is defined, or "
      "uniform)"
  )


def symmetric_fictitious_play(
    game: SymmetricGame, iterations: int = 20000
) -> SymmetricSolution:
  """Symmetric fictitious play.

  sigma_t = mean of best responses to sigma_0 .. sigma_{t-1}, sigma_0 uniform
  (ties: lowest index).

  Args:
    game: the SymmetricGame.
    iterations: T.

  Returns:
    sigma_T with its regret.
  """
  assert iterations >= 1, iterations
  k = game.num_strategies
  counts = np.zeros(k)
  avg = np.full(k, 1.0 / k)
  for _ in range(int(iterations)):
    counts[int(np.argmax(game.fitness(avg)))] += 1.0
    avg = counts / counts.sum()
  return SymmetricSolution(avg, game.regret(avg), int(iterations))


def _log_fixation_sym(
    game: SymmetricGame, r: int, s: int, alpha: float, m: int
) -> float:
  """Log of the fixation probability of one r mutant among m - 1 s residents.

  Co-players drawn without replacement (module doc).

  Args:
    game: the SymmetricGame.
    r: the mutant's strategy.
    s: the residents' strategy.
    alpha: the selection intensity.
    m: the population size.

  Returns:
    The log fixation probability.
  """
  t = game.num_players
  k = np.arange(1, m)  # mutants in the population
  fr, fs = np.zeros(m - 1), np.zeros(m - 1)
  for a in range(t):  # a mutants among the T - 1 co-players
    ms = tuple(sorted([r] * a + [s] * (t - 1 - a)))
    ur, us = game.payoff(r, ms), game.payoff(s, ms)
    pr = np.asarray(
        [
            math.comb(int(kk) - 1, a) * math.comb(m - int(kk), t - 1 - a)
            for kk in k
        ],
        F64,
    ) / math.comb(m - 1, t - 1)
    ps = np.asarray(
        [
            math.comb(int(kk), a) * math.comb(m - int(kk) - 1, t - 1 - a)
            for kk in k
        ],
        F64,
    ) / math.comb(m - 1, t - 1)
    fr += pr * ur
    fs += ps * us
  terms = np.concatenate([[0.0], np.cumsum(-alpha * (fr - fs))])
  return -float(np.logaddexp.reduce(terms))


def symmetric_alpha_rank(
    game: SymmetricGame, alpha: float = 10.0, population_size: int = 50
) -> SymmetricSolution:
  """Single-population alpha-rank with T-player fitness (module doc).

  Args:
    game: the SymmetricGame.
    alpha: the selection intensity.
    population_size: m.

  Returns:
    The SymmetricSolution (its stationary distribution).

  Raises:
    ValueError: population_size below the number of players.
  """
  assert alpha >= 0.0 and np.isfinite(alpha), alpha
  m, t, n = int(population_size), game.num_players, game.num_strategies
  if m < t:
    raise ValueError(
        f"alpha_rank: population_size {m} < {t} players (co-players are drawn "
        "without replacement from the population)"
    )
  if n == 1:
    return SymmetricSolution(np.ones(1), 0.0, 0)
  eta = 1.0 / (n - 1)
  tr = np.zeros((n, n))
  for s in range(n):
    for r in range(n):
      if r != s:
        tr[s, r] = eta * np.exp(_log_fixation_sym(game, r, s, alpha, m))
    tr[s, s] = 1.0 - tr[s].sum()
  a = np.vstack([tr.T - np.eye(n), np.ones((1, n))])
  b = np.concatenate([np.zeros(n), [1.0]])
  pi, *_ = np.linalg.lstsq(a, b, rcond=None)
  pi = _normalize(pi, "alpha_rank")
  return SymmetricSolution(pi, game.regret(pi), 0)


def _sym_uniform(game: SymmetricGame) -> SymmetricSolution:
  """The uniform mixture of `game`, with its regret.

  Args:
    game: The SymmetricGame.

  Returns:
    The SymmetricSolution.
  """
  x = np.full(game.num_strategies, 1.0 / game.num_strategies)
  return SymmetricSolution(x, game.regret(x), 0)


SYMMETRIC: Mapping[str, Callable[..., SymmetricSolution]] = {
    "uniform": _sym_uniform,
    "replicator": replicator,
    "fictitious_play": symmetric_fictitious_play,
    "alpha_rank": symmetric_alpha_rank,
}
assert tuple(SYMMETRIC) == SYMMETRIC_SOLVERS


def solve_symmetric(
    name: str, game: SymmetricGame, **kwargs
) -> SymmetricSolution:
  """SYMMETRIC[name] on a SymmetricGame.

  Args:
    name: a key of SYMMETRIC.
    game: the SymmetricGame.
    **kwargs: passed to the solver.

  Returns:
    The SymmetricSolution.

  Raises:
    TypeError: game is not a SymmetricGame.
    CapabilityError: any other solver name (a two-player matrix solver).
  """
  if not isinstance(game, SymmetricGame):
    raise TypeError(
        f"solve_symmetric takes a SymmetricGame, got {type(game).__name__}"
    )
  if name not in SYMMETRIC:
    raise CapabilityError(
        f"meta-solver {name!r} does not solve a symmetric "
        f"{game.num_players}-player meta-game; symmetric solvers: "
        f"{SYMMETRIC_SOLVERS}"
    )
  sol = SYMMETRIC[name](game, **kwargs)
  x = sol.sigma
  assert x.dtype == F64 and np.all(x >= 0.0) and abs(x.sum() - 1.0) < 1e-9, (
      name,
      x,
  )
  return sol
