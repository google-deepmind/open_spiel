# Copyright 2019 DeepMind Technologies Limited
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
"""Nash averaging.

Based on https://arxiv.org/abs/1806.02643. An axiomatic strategy evaluation
metric for Agent-vs-Agent or Agent-vs-Task two-player zero-sum games.
"""

import cvxpy as cp
import numpy as np

from open_spiel.python.egt.utils import game_payoffs_array


def _max_entropy_symmetric_nash(p_mat, eps=1e-9):
  """Solves for the maxent symmetric nash for symmetric 2P zero-sum games.

    Using convex programming:
      min p^Tlog(p)
      s.t.
      p_mat.dot(p) <= 0, since game value must be 0
      p >= 0
      1^T * p = 1

  Args:
    p_mat: an N*N anti-symmetric payoff matrix for the row player
    eps: minimum probability threshold

  Returns:
    p*: a maxent symmetric nash
  """
  assert np.array_equal(p_mat, -p_mat.T) and eps >= 0 and eps <= 0.5
  n = len(p_mat)
  x = cp.Variable(shape=n)
  obj = cp.Maximize(cp.sum(cp.entr(x)))
  constraints = [p_mat @ x <= 0, x >= eps * np.ones(n)]
  constraints.append(cp.sum(x) == 1)
  prob = cp.Problem(obj, constraints)
  prob.solve()
  return x.value.reshape((-1, 1))


def _max_entropy_symmetric_nash_avt(p_mat, num_agents, num_tasks, eps=1e-9):
  """Solves for the maxent symmetric nash for symmetric 2P zero-sum games.

    This covers the agent-vs-task cases.

    Using convex programming:
      min x^Tlog(x) + y^Tlog(y)
      s.t.
      x >= 0
      1^T * x = 1
      y >= 0
      1^T * y = 1
      forall s, such that s has exactly one unit mass on an agent strategy
      and one unit mass on a task strategy,
      s^T*p_mat*z <= 0, where z = [x, y], since game-value is 0.

  Args:
    p_mat: an N*N anti-symmetric payoff matrix for the row player
    num_agents: number of agents
    num_tasks: number of tasks
    eps: minimum probability threshold

  Returns:
    (x*, y*): a maxent symmetric nash
  """
  assert np.array_equal(p_mat, -p_mat.T) and eps >= 0 and eps <= 0.5
  n = len(p_mat)
  assert n == num_agents + num_tasks
  x = cp.Variable(shape=num_agents)
  y = cp.Variable(shape=num_tasks)
  z = cp.hstack([x, y])
  obj = cp.Maximize(cp.sum(cp.entr(z)))
  constraints = [
      x >= eps * np.ones(num_agents),
      cp.sum(x) == 1,
      y >= eps * np.ones(num_tasks),
      cp.sum(y) == 1,
  ]

  dev_payoffs = p_mat @ z
  for a_idx in range(num_agents):
    for t_idx in range(num_tasks):
      pure_strategy = np.zeros(n)
      pure_strategy[a_idx] = 1
      pure_strategy[num_agents + t_idx] = 1
      pure_strategy = pure_strategy.reshape((1, -1))
      constraints.append(pure_strategy @ dev_payoffs <= 0)

  prob = cp.Problem(obj, constraints)
  prob.solve()
  return x.value.reshape((-1, 1)), y.value.reshape((-1, 1))


def nash_averaging_avt_matrix(s_mat, eps=0.0):
  """Apply the agent-vs-task Nash Averaging from Appendix D, from a matrix.

  Args:
    s_mat: The S matrix from the paper, representing m rows (agents) and n
      columns (tasks), with scores for the agent on the task. Note that the
      values need not be normalized, but will be normalized across tasks before
      being processed.
    eps: minimum probability threshold.

  Returns:
    maxent_nash: nash mixture for row player and column player
    nash_avg_score: the expected payoff under maxent_nash
  """
  m, n = s_mat.shape
  min_payoffs = np.min(s_mat, axis=0)
  max_payoffs = np.max(s_mat, axis=0)
  std_p_mat = (s_mat - min_payoffs) / (max_payoffs - min_payoffs)
  a_mat = np.block([
      [np.zeros(shape=(m, m)), std_p_mat],
      [-std_p_mat.T, np.zeros(shape=(n, n))],
  ])
  pa_sol, pe_sol = _max_entropy_symmetric_nash_avt(
      a_mat, num_agents=m, num_tasks=n, eps=eps)
  pa, pe = np.asarray(pa_sol), np.asarray(pe_sol)
  return (pa, pe), (std_p_mat.dot(pe), -std_p_mat.T.dot(pa))


def nash_averaging(game, eps=0.0, a_v_a=True):
  """Nash averaging, see https://arxiv.org/abs/1806.02643.

  Args:
    game: a pyspiel game
    eps: minimum probability mass for maxent nash
    a_v_a: whether it is Agent-vs-Agent or Agent-vs-Task

  Returns:
    maxent_nash: nash mixture for row player and column player
    nash_avg_score: the expected payoff under maxent_nash
  """

  p_mat = game_payoffs_array(game)
  if len(p_mat) != 2:
    raise ValueError("Nash Averaging works only for two players.")
  if np.max(np.abs(p_mat[0] + p_mat[1])) > 0:
    raise ValueError("Must be zero-sum")
  if a_v_a:
    if not np.array_equal(p_mat[0], -p_mat[0].T):
      raise ValueError(
          "AvA only works for symmetric two-player zero-sum games.")
    maxent_nash = np.array(_max_entropy_symmetric_nash(p_mat[0], eps=eps))
    return maxent_nash, p_mat[0].dot(maxent_nash)

  # For AvT, see appendix D of the paper.
  # Here assumes the row player represents agents and the column player
  # represents tasks.
  # game does not have to be symmetric
  return nash_averaging_avt_matrix(p_mat[0], eps=eps)
