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
  a_mat = np.ones(n).reshape((1, n))
  constraints = [p_mat @ x <= 0, a_mat @ x == 1, x >= eps * np.ones(n)]
  prob = cp.Problem(obj, constraints)
  prob.solve()
  return x.value.reshape((-1, 1))


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

  m, n = p_mat[0].shape
  a_mat = np.block([[np.zeros(shape=(m, m)), p_mat[0]],
                    [-p_mat[0].T, np.zeros(shape=(n, n))]])
  maxent_nash = np.array(_max_entropy_symmetric_nash(a_mat, eps=eps))
  pa, pe = maxent_nash[:m], maxent_nash[m:]
  return (pa, pe), (p_mat[0].dot(pe), -p_mat[0].T.dot(pa))
