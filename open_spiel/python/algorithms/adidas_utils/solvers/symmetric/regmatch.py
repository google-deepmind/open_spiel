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
    if self.rnd_init:
      init_dist = self.random.rand(num_strats)
    else:
      init_dist = np.ones(num_strats)
    init_dist /= init_dist.sum()
    init_regret = np.zeros(num_strats)
    return (init_dist, init_regret)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    grad_regret = grads[1]
    self.aux_errors.append([np.linalg.norm(grad_regret)])

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients (and exploitabilities) for all parameters.

    Args:
      params: tuple of params (dist, regret), see regmatch.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      tuple of gradients (grad_dist, grad_regret), see ate.gradients
      unregularized exploitability (stochastic estimate)
      solver exploitability (stochastic estimate) - NaN
    """
    return gradients(*params, payoff_matrices)

  def exploitability(self, params, payoff_matrices):
    """Regret matching does not minimize any exploitability so return NaN.

    Args:
      params: tuple of params (dist,)
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
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
    new_regret = gamma * regret + regret_delta
    new_clipped_regrets = np.clip(new_regret + self.optimism * regret_delta,
                                  0.,
                                  np.inf)
    if np.sum(new_clipped_regrets) > 0:
      new_dist = new_clipped_regrets / new_clipped_regrets.sum()
    else:
      new_dist = np.ones_like(dist) / dist.size
    new_params = (new_dist, new_regret)
    return new_params


def gradients(dist, regret, payoff_matrices):
  """Computes regret delta to be added to regret in update.

  Args:
    dist: 1-d np.array, current estimate of nash distribution
    regret: 1-d np.array (same shape as dist), current estimate of regrets
    payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
  Returns:
    deltas w.r.t. (dist, regret) as tuple
    unregularized exploitability (stochastic estimate)
    solver exploitability (stochastic estimate) - NaN
  """
  del regret

  nabla = payoff_matrices[0].dot(dist)
  utility = nabla.dot(dist)

  grad_dist = np.NaN * np.ones_like(dist)
  grad_regret = nabla - utility

  unreg_exp = np.max(nabla) - nabla.dot(dist)

  return (grad_dist, grad_regret), unreg_exp, np.NaN
