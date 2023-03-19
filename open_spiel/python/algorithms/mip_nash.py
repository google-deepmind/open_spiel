# Copyright 2023 DeepMind Technologies Limited
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
'''MIP-Nash.

Based on the first formulation of https://dl.acm.org/doi/10.5555/1619410.1619413.
Compute optimal Nash equilibrium of two-player general-sum games by solving a mixed-integer programming problem.
'''


import numpy as np
import cvxpy as cp
from open_spiel.python.algorithms.projected_replicator_dynamics import _simplex_projection
from open_spiel.python.egt.utils import game_payoffs_array


def mip_nash(game, objective, solver='GLPK_MI'):
  """Solves for the optimal Nash for two-player general-sum games.
    Using mixed-integer programming:
      min f(x, y, p_mat)
      s.t.
      (u0, u1 are Nash payoffs variables of player 0 and 1)
      p_mat[0] * y <= u0 
      x^T*p_mat[1] <= u1
      (if a pure strategy is in the support then its payoff is Nash payoff)
      u0 - p_mat[0] * y <= U0 * b0 
      u1 - x^T*p_mat[1] <= U1 * b1
      (if a pure strategy is not in the support its probability mass is 0)
      x <= 1 - b0
      y <= 1 - b1
      (probability constraints)
      x >= 0
      1^T * x = 1
      y >= 0
      1^T * y = 1
      for all n, b0[n] \in {0, 1},
      for all m, b1[m] \in {0, 1},
      U0, U1 are the maximum payoff differences of player 0 and 1.
    This formulation is a basic one that may only work well for simple objective function or low-dimensional inputs.
    To handle more complex cases, It is possible to extend this by using advanced internal solvers or piecewise linear approximation of the objective.
  Args:
    game: a pyspiel matrix game object
    objective: a string representing the objective (e.g., MAX_SOCIAL_WELFARE)
    solver: the mixed-integer solver used by cvxpy
  Returns:
    optimal Nash (x, y)
  """

  p_mat = game_payoffs_array(game)
  if len(p_mat) != 2:
    raise ValueError("MIP-Nash only works for two players.")

  assert len(p_mat) == 2
  assert p_mat[0].shape == p_mat[1].shape

  (M, N) = p_mat[0].shape

  U0 = np.max(p_mat[0]) - np.min(p_mat[0])
  U1 = np.max(p_mat[1]) - np.min(p_mat[1])

  x = cp.Variable(M)
  y = cp.Variable(N)
  u0 = cp.Variable(1)
  u1 = cp.Variable(1)
  b0 = cp.Variable(M, boolean=True)
  b1 = cp.Variable(N, boolean=True)

  u_m = p_mat[0] @ y
  u_n = x @ p_mat[1]

  # probabilities constraints
  constraints = [x >= 0, y >= 0, cp.sum(x) == 1, cp.sum(y) == 1]
  # support constraints
  constraints.extend([u_m <= u0, u0-u_m <= U0 * b0, x <= 1-b0])
  constraints.extend([u_n <= u1, u1-u_n <= U1 * b1, y <= 1-b1])

  variables = {'x': x, 'y': y, 'u0': u0,
               'u1': u1, 'b0': b0, 'b1': b1, 'p_mat': p_mat}

  obj = TWO_PLAYER_OBJECTIVE[objective](variables)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)

  return _simplex_projection(x.value.reshape(-1)), _simplex_projection(y.value.reshape(-1))



def max_social_welfare_two_player(variables):
  return cp.Maximize(variables['u0'] + variables['u1'])


def min_social_welfare_two_player(variables):
  return cp.Minimize(variables['u0'] + variables['u1'])


def max_support_two_player(variables):
  return cp.Minimize(cp.sum(variables['b0']) + cp.sum(variables['b1']))


def min_support_two_player(variables):
  return cp.Maximize(cp.sum(variables['b0']) + cp.sum(variables['b1']))


def max_gini_two_player(variables):
  return cp.Minimize(cp.sum(cp.square(variables['x'])) + cp.sum(cp.square(variables['y'])))


TWO_PLAYER_OBJECTIVE = {
    'MAX_SOCIAL_WELFARE': max_social_welfare_two_player,
    'MIN_SOCIAL_WELFARE': min_social_welfare_two_player,
    'MAX_SUPPORT': max_support_two_player,
    'MIN_SUPPORT': min_support_two_player,
    'MAX_GINI': max_gini_two_player,
}