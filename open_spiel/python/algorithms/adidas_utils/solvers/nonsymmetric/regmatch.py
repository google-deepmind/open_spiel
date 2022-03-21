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
"""Regret Matching Approximate Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import numpy as np


class Solver(object):
  """Regret-matching Solver."""

  def __init__(self, optimism=True, discount=False, rnd_init=False, seed=None,
               **kwargs):
    """Ctor."""
    del kwargs
    self.num_players = None
    self.lrs = None
    self.optimism = optimism
    self.discount = discount
    self.rnd_init = rnd_init
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
    init_regret = [np.zeros_like(dist_i) for dist_i in init_dist]
    return (init_dist, init_regret)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    grad_regret = grads[1]
    grad_regret_flat = np.concatenate(grad_regret)
    self.aux_errors.append([np.linalg.norm(grad_regret_flat)])

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients (and exploitabilities) for all parameters.

    Args:
      params: tuple of params (dist, regret), see regmatch.gradients
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    Returns:
      tuple of gradients (grad_dist, grad_regret), see ate.gradients
      unregularized exploitability (stochastic estimate)
      solver exploitability (stochastic estimate) - NaN
    """
    return gradients(*params, payoff_matrices, self.num_players)

  def exploitability(self, params, payoff_matrices):
    """Regret matching does not minimize any exploitability so return NaN.

    Args:
      params: tuple of params (dist,)
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    Returns:
      np.NaN
    """
    del params
    del payoff_matrices
    return np.NaN

  def update(self, params, grads, t):
    """Update cumulative regret and strategy (dist).

    Args:
      params: tuple of variables to be updated (dist, regret)
      grads: tuple of variable gradients (grad_dist, grad_regret)
      t: int, solver iteration (not used)
    Returns:
      new_params: tuple of update params (new_dist, new_regret)
    """
    dist, regret = params
    regret_delta = grads[1]
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

    new_params = (new_dist, new_regret)
    return new_params


def gradients(dist, regret, payoff_matrices, num_players):
  """Computes regret delta to be added to regret in update.

  Args:
    dist: list of 1-d np.arrays, current estimate of nash distribution
    regret: list of 1-d np.arrays (same as dist), current estimate of regrets
    payoff_matrices: dictionary with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    num_players: int, number of players, in case payoff_matrices is abbreviated
  Returns:
    deltas w.r.t. (dist, regret) as tuple
    unregularized exploitability (stochastic estimate)
    solver exploitability (stochastic estimate) - NaN
  """
  del regret

  # first compute best responses and payoff gradients
  grad_dist = []
  grad_regret = []
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

    grad_dist_i = np.NaN * np.ones_like(nabla_i)
    grad_dist.append(grad_dist_i)

    utility_i = nabla_i.dot(dist[i])
    grad_regret_i = nabla_i - utility_i
    grad_regret.append(grad_regret_i)

    unreg_exp.append(np.max(nabla_i) - nabla_i.dot(dist[i]))

  return (grad_dist, grad_regret), np.mean(unreg_exp), np.NaN
