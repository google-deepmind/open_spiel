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

"""Nash averaging."""

from cvxopt import solvers, matrix, spdiag, spmatrix, log
import numpy as np

from open_spiel.python.egt.utils import game_payoffs_array



def max_entropy_nash(p_mat, eps=0.0):
  '''
    Solving for maxent symmetric nash for symmetric two-player zero-sum games
    convex programming:
      max p^Tlog(p)
      s.t. 
      p_mat.dot(p) <= p^T*p_mat*p
      p >= 0
      1^T * p = 0

    Args:
      p_mat: an N*N anti-symmetric payoff matrix
      eps: minimum probability threshold

    Returns:
      p*: a maxent nash  
  '''
  assert np.array_equal(p_mat, -p_mat.T) and eps >= 0 and eps <= 0.5
  N=len(p_mat)
  p_mat = matrix(p_mat)
  solvers.options["show_progress"] = False
  def F(x=None, z=None):
    if x is None: return 2 * N, matrix(1/N, (N,1))
    if min(x) <= eps or max(x) >= 1-eps: return None
    ev = x.T * p_mat * x
    f = matrix(0.0, (2*N+1, 1))
    df = matrix(0.0, (2*N+1, N))
    f[0] = x.T * log(x)
    df[0, :] = (log(x) + 1).T
    f[1:N+1] = p_mat * x - ev
    df[1:N+1, :] = p_mat
    f[N+1:] = -x+eps
    df[N+1:, :] = -spmatrix(1.0, range(N), range(N))
    if z is None:
      return f, df
    H = spdiag(z[0] * x**(-1))
    return f, df, H
  A = matrix(1.0, (1, N))
  b = matrix(1.0, (1, 1))
  return solvers.cp(F, A=A, b=b)['x']

def nash_averaging(game, eps=0.0, AvA=True):

  '''
    Nash averaging, see https://arxiv.org/pdf/1806.02643.pdf

    Args:
      game: a pyspiel game
      eps: minimum probability mass for maxent nash
      AvA: whether it is Agent-vs-Agent or Agent-vs-Task
    Returns:
      maxent_nash: nash mixture for row player and column player
      nash_avg_score: the expected payoff under maxent_nash
  
  '''
  p_mat = game_payoffs_array(game)
  if len(p_mat) != 2:
    raise ValueError("Nash Averaging works only for two players.")
  if np.max(np.abs(p_mat[0] + p_mat[1])) > 0:
     raise ValueError("Must be zero-sum")
  if AvA:
    if not np.array_equal(p_mat[0], -p_mat[0].T):
      raise ValueError("AvA only works for symmetric two-player zero-sum games.")
    maxent_nash = np.array(max_entropy_nash(p_mat[0], eps=eps))
    return maxent_nash, p_mat[0].dot(maxent_nash)


  '''
    For AvT, see appendix D of the paper
  '''


  M, N = p_mat[0].shape
  A = np.zeros((M+N, M+N))
  A[:M, M:M+N] = p_mat[0]
  A[M:, :M] = -p_mat[0].T
  maxent_nash = np.array(max_entropy_nash(A, eps=eps))
  pa, pe = maxent_nash[:M], maxent_nash[M:]
  return (pa, pe), (p_mat[0].dot(pe), -p_mat[0].T.dot(pa))
