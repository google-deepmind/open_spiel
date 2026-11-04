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
"""

from __future__ import annotations

from typing import Callable, Mapping, NamedTuple, Optional

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


def solve(name: str, payoff, **kwargs) -> MetaSolution:
  """Runs the meta-solver `name` and checks its distributions.

  Args:
    name: a key of SOLVERS.
    payoff: the row player's payoff matrix.
    **kwargs: passed to the solver.

  Returns:
    The MetaSolution.

  Raises:
    KeyError: an unknown solver.
  """
  if name not in SOLVERS:
    raise KeyError(f"unknown meta-solver {name!r}; known: {sorted(SOLVERS)}")
  sol = SOLVERS[name](payoff, **kwargs)
  for x in (sol.row, sol.col):
    assert x.dtype == F64 and np.all(x >= 0.0) and abs(x.sum() - 1.0) < 1e-9, (
        name,
        x,
    )
  return sol
