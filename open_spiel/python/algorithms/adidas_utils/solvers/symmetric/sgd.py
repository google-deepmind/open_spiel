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

"""Stochastic Gradient Descent (SGD) Approx. Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as exp
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import updates


class Solver(updates.Solver):
  """SGD Solver."""

  def __init__(self, temperature=0., proj_grad=True, euclidean=False,
               lrs=(1e-1,), rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    super().__init__(proj_grad, euclidean, rnd_init, seed)
    if temperature < 0.:
      raise ValueError('temperature must be non-negative')
    self.temperature = temperature
    self.lrs = lrs
    self.num_estimates = 2

    if temperature > 0:
      self.eps = np.exp(-1 / temperature)  # ensure dist[i] >= eps / dim(dist)
    else:
      self.eps = 0.
    if euclidean:
      self.update = lambda *args: self.euc_descent_step(*args, eps=self.eps)
    else:
      self.update = lambda *args: self.mirror_descent_step(*args, eps=self.eps)

  def init_vars(self, num_strats, num_players):
    """Initialize solver parameters."""
    self.num_players = num_players
    if self.rnd_init:
      init_dist = self.random.rand(num_strats)
    else:
      init_dist = np.ones(num_strats)
    init_dist /= init_dist.sum()
    init_dist = simplex.project_to_interior(init_dist, self.eps)
    return (init_dist,)

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return exploitability.

    Args:
      params: tuple of params (dist,), see sgd.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability of current dist
      unregularized exploitability (stochastic estimate)
      shannon regularized exploitability (stochastic estimate)
    """
    return gradients(*params, payoff_matrices, self.num_players,
                     self.temperature, self.proj_grad)

  def exploitability(self, params, payoff_matrices):
    """Compute and return exploitability.

    Args:
      params: tuple of params (dist,), see sgd.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability as avg squared norm of projected-gradient
    """
    return exp.grad_norm_exploitability(params, payoff_matrices, eta=1.,
                                        temperature=self.temperature)


def gradients(dist, payoff_matrices, num_players, temperature=0.,
              proj_grad=True):
  """Computes exploitablity gradient.

  Assumption: eta_k = 1 for all k

  Args:
    dist: 1-d np.array, current estimate of nash distribution
    payoff_matrices: 2 (>=2 x A x A) np.arrays, payoffs for each joint action
    num_players: int, number of players, in case payoff_matrices is abbreviated
    temperature: non-negative float, default 0.
    proj_grad: bool, if True, projects dist gradient onto simplex
  Returns:
    gradient of exploitability w.r.t. (dist) as tuple
    unregularized exploitability (stochastic estimate)
    shannon regularized exploitability (stochastic estimate)
  """
  del num_players
  tau = temperature

  a, b = 0, 1  # 2 samples needed for unbiased estimation
  p_0, p_1 = 0, 1  # player 0 index, player 1 index
  hess_0_01_a = payoff_matrices[a][p_0]
  hess_1_01_a = payoff_matrices[a][p_1]
  hess_0_01_b = payoff_matrices[b][p_0]

  pg_0_a = simplex.project_grad(hess_0_01_a.dot(dist))
  pg_0_b = simplex.project_grad(hess_0_01_b.dot(dist))

  unreg_exp = np.dot(pg_0_a, pg_0_b)

  if tau > 0.:
    log_dist_safe = np.clip(np.log(dist), -40, 0)
    entr_grad_proj = simplex.project_grad(-tau * (log_dist_safe + 1))
  else:
    entr_grad_proj = 0.
  pg_0_a_entr = pg_0_a + entr_grad_proj
  pg_0_b_entr = pg_0_b + entr_grad_proj
  pg_0_entr = 0.5 * (pg_0_a_entr + pg_0_b_entr)
  pg_1_b_entr = pg_0_b_entr

  reg_exp = np.dot(pg_0_a_entr, pg_0_b_entr)

  # then construct unbiased stochastic gradient
  grad_dist = 2. * hess_1_01_a.dot(pg_1_b_entr)
  if tau > 0.:
    grad_dist += 2. * -tau * pg_0_entr / dist

  if proj_grad:
    grad_dist = simplex.project_grad(grad_dist)

  return (grad_dist,), unreg_exp, reg_exp
