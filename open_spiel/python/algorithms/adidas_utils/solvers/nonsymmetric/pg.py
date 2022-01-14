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

# Lint as: python3
"""Policy Gradient (PG)."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import updates


class Solver(updates.Solver):
  """PG Solver."""

  def __init__(self, proj_grad=True, euclidean=False, lrs=(1e-1,),
               rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    super().__init__(proj_grad, euclidean, rnd_init, seed)
    self.lrs = lrs

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients for all parameters.

    Args:
      params: tuple of params (dist,), see pg.gradients
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
          values of (2 x A x A) np.arrays, payoffs for each joint action. keys
          are sorted and arrays should be indexed in the same order
    Returns:
      tuple of gradients (grad_dist,), see pg.gradients
      unregularized exploitability (stochastic estimate)
      unregularized exploitability (stochastic estimate) *duplicate
    """
    return gradients(*params, payoff_matrices, self.num_players, self.proj_grad)

  def exploitability(self, params, payoff_matrices):
    """Policy gradient does not minimize any exploitability so return NaN.

    Args:
      params: tuple of params (dist,)
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      np.NaN
    """
    return np.NaN


def gradients(dist, payoff_matrices, num_players, proj_grad=True):
  """Computes exploitablity gradient.

  Args:
    dist: list of 1-d np.arrays, current estimate of nash distribution
    payoff_matrices: dictionary with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    num_players: int, number of players, in case payoff_matrices is abbreviated
    proj_grad: bool, if True, projects dist gradient onto simplex
  Returns:
    gradient of payoff w.r.t. (dist) as tuple
    unregularized exploitability (stochastic estimate)
    unregularized exploitability (stochastic estimate) *duplicate
  """
  # first compute best responses and payoff gradients
  grad_dist = []
  unreg_exp = []
  for i in range(num_players):

    nabla_i = np.zeros_like(dist[i])
    # TODO(imgemp): decide if averaging over nablas provides best comparison
    for j in range(num_players):
      if j == i:
        continue
      if i < j:
        hess_i_ij = payoff_matrices[(i, j)][0]
      else:
        hess_i_ij = payoff_matrices[(j, i)][1].T

      nabla_ij = hess_i_ij.dot(dist[j])
      nabla_i += nabla_ij / float(num_players - 1)

    grad_dist_i = -nabla_i
    if proj_grad:
      grad_dist_i = simplex.project_grad(grad_dist_i)
    grad_dist.append(nabla_i)

    unreg_exp.append(np.max(nabla_i) - nabla_i.dot(dist[i]))

  return (grad_dist,), np.mean(unreg_exp), np.mean(unreg_exp)
