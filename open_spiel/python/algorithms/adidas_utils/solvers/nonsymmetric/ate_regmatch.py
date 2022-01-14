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
"""Adaptive Tsallis Entropy (ATE) Stochastic Regret Matching Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import misc
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as exp


class Solver(object):
  """ATE Exploitability Regret Matching Solver."""

  def __init__(self, p=1., lrs=(1e-2,), optimism=True, discount=False,
               rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    if (p < 0.) or (p > 1.):
      raise ValueError('p must be in [0, 1]')
    self.num_players = None
    self.p = p
    self.rnd_init = rnd_init
    self.lrs = lrs
    self.optimism = optimism
    self.discount = discount
    self.has_aux = True
    self.aux_errors = []

    self.seed = seed
    self.random = np.random.RandomState(seed)

  def init_vars(self, num_strats, num_players):
    """Initialize solver parameters."""
    self.num_players = num_players
    if len(num_strats) != num_players:
      raise ValueError('Must specify num strategies for each player')
    init_dist = []
    for num_strats_i in num_strats:
      if self.rnd_init:
        init_dist_i = self.random.rand(num_strats_i)
      else:
        init_dist_i = np.ones(num_strats_i)
      init_dist_i /= init_dist_i.sum()
      init_dist.append(init_dist_i)
    init_y = [np.zeros_like(dist_i) for dist_i in init_dist]
    init_cumgrad = [np.zeros_like(dist_i) for dist_i in init_dist]
    return (init_dist, init_y, init_cumgrad)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    concat = []
    for grad in grads:
      concat.extend([np.ravel(g) for g in grad])
    self.aux_errors.append([np.linalg.norm(np.concatenate(concat))])

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients (and exploitabilities) for all parameters.

    Args:
      params: tuple of params (dist, y), see ate.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      tuple of gradients (grad_dist, grad_y), see ate.gradients
      unregularized exploitability (stochastic estimate)
      tsallis regularized exploitability (stochastic estimate)
    """
    return gradients(*params, payoff_matrices, self.num_players, self.p)

  def exploitability(self, dist, payoff_matrices):
    """Compute and return tsallis entropy regularized exploitability.

    Args:
      dist: tuple of list of player distributions (dist,)
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability of current dist
    """
    return exp.ate_exploitability(dist, payoff_matrices, self.p)

  def update(self, params, grads, t):
    """Projected gradient descent on exploitability using Euclidean projection.

    Args:
      params: tuple of variables to be updated (dist, y, regret)
      grads: tuple of variable gradients (grad_dist, grad_y, regret_delta)
      t: int, solver iteration
    Returns:
      new_params: tuple of update params (new_dist, new_y, new_regret)
    """
    dist, y, regret = params
    _, y_grad, regret_delta = grads

    lr_y = np.clip(1 / float(t + 1), self.lrs[0], np.inf)
    new_y = []
    for y_i, y_grad_i in zip(y, y_grad):
      new_y_i = y_i - lr_y * y_grad_i
      new_y_i = np.clip(new_y_i, 0., np.inf)
      new_y.append(new_y_i)

    if self.discount:
      gamma = t / float(t + 1)
    else:
      gamma = 1

    new_dist = []
    new_regret = []
    for dist_i, regret_i, regret_delta_i in zip(dist, regret, regret_delta):
      new_regret_i = gamma * regret_i + regret_delta_i
      new_clipped_regrets_i = np.clip(
          new_regret_i + self.optimism * regret_delta_i, 0., np.inf)
      if np.sum(new_clipped_regrets_i) > 0:
        new_dist_i = new_clipped_regrets_i / new_clipped_regrets_i.sum()
      else:
        new_dist_i = np.ones_like(dist_i) / dist_i.size
      new_dist.append(new_dist_i)
      new_regret.append(new_regret_i)

    return (new_dist, new_y, new_regret)


def gradients(dist, y, regret, payoff_matrices, num_players, p=1):
  """Computes exploitablity gradient and aux variable gradients.

  Args:
    dist: list of 1-d np.arrays, current estimate of nash distribution
    y: list 1-d np.arrays (same shape as dist), current est. of payoff gradient
    regret: list of 1-d np.arrays (same shape as dist), exploitability regrets
    payoff_matrices: dictionary with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    num_players: int, number of players, in case payoff_matrices is abbreviated
    p: float in [0, 1], Tsallis entropy-regularization --> 0 as p --> 0
  Returns:
    gradient of exploitability w.r.t. (dist, y) as tuple
    unregularized exploitability (stochastic estimate)
    tsallis regularized exploitability (stochastic estimate)
  """
  del regret

  # first compute policy gradients and player effects (fx)
  policy_gradient = []
  other_player_fx = []
  grad_y = []
  unreg_exp = []
  reg_exp = []
  for i in range(num_players):

    nabla_i = np.zeros_like(dist[i])
    for j in range(num_players):
      if j == i:
        continue
      if i < j:
        hess_i_ij = payoff_matrices[(i, j)][0]
      else:
        hess_i_ij = payoff_matrices[(j, i)][1].T

      nabla_ij = hess_i_ij.dot(dist[j])
      nabla_i += nabla_ij / float(num_players - 1)

    grad_y.append(y[i] - nabla_i)
    y[i] = nabla_i  # TODO(imgemp): overwriting temporarily to test something

    if p > 0:
      power = 1. / float(p)
      s_i = np.linalg.norm(y[i], ord=power)
      if s_i == 0:
        br_i = misc.uniform_dist(y[i])
      else:
        br_i = (y[i] / s_i)**power
    else:
      power = np.inf
      s_i = np.linalg.norm(y[i], ord=power)
      br_i = np.zeros_like(dist[i])
      maxima_i = (y[i] == s_i)
      br_i[maxima_i] = 1. / maxima_i.sum()

    policy_gradient_i = nabla_i - s_i * dist[i]**p
    policy_gradient.append(policy_gradient_i)

    unreg_exp.append(np.max(y[i]) - y[i].dot(dist[i]))

    br_i_inv_sparse = 1 - np.sum(br_i**(p + 1))
    dist_i_inv_sparse = 1 - np.sum(dist[i]**(p + 1))
    entr_br_i = s_i / (p + 1) * br_i_inv_sparse
    entr_dist_i = s_i / (p + 1) * dist_i_inv_sparse

    reg_exp.append(y[i].dot(br_i - dist[i]) + entr_br_i - entr_dist_i)

    entr_br_vec_i = br_i_inv_sparse * br_i**(1 - p)
    entr_dist_vec_i = dist_i_inv_sparse * dist[i]**(1 - p)
    other_player_fx_i = (br_i - dist[i]) + 1 / (p + 1) * (
        entr_br_vec_i - entr_dist_vec_i)
    other_player_fx.append(other_player_fx_i)

  # then construct exploitability gradient
  grad_dist = []
  regret_delta = []
  for i in range(num_players):

    grad_dist_i = -policy_gradient[i]
    for j in range(num_players):
      if j == i:
        continue
      if i < j:
        hess_j_ij = payoff_matrices[(i, j)][1]
      else:
        hess_j_ij = payoff_matrices[(j, i)][0].T

      grad_dist_i += hess_j_ij.dot(other_player_fx[j])

    regret_delta_i = -(grad_dist_i - grad_dist_i.dot(dist[i]))
    # regret_delta_i = y[i] - y[i].dot(dist[i])

    grad_dist.append(grad_dist_i)
    regret_delta.append(regret_delta_i)

  return (grad_dist, grad_y, regret_delta), np.mean(unreg_exp), np.mean(reg_exp)
