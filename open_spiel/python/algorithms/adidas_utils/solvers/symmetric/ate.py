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
"""Adaptive Tsallis Entropy (ATE) Stochastic Approximate Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import numpy as np
from scipy import special

from open_spiel.python.algorithms.adidas_utils.helpers import misc
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as exp


class Solver(object):
  """ATE Solver."""

  def __init__(self, p=1., proj_grad=True, euclidean=False, cheap=False,
               lrs=(1e-2, 1e-1), vr=True, rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    if (p < 0.) or (p > 1.):
      raise ValueError("p must be in [0, 1]")
    self.num_players = None
    self.p = p
    self.proj_grad = proj_grad
    self.cheap = cheap
    self.vr = vr
    self.pm_vr = None
    self.rnd_init = rnd_init
    self.lrs = lrs
    self.has_aux = True
    self.aux_errors = []

    self.euclidean = euclidean
    if euclidean:
      self.update = self.euc_descent_step
    else:
      self.update = self.mirror_descent_step

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
    init_y = np.zeros(num_strats)
    if self.cheap and self.vr:
      self.pm_vr = np.zeros((num_strats, num_strats))
    return (init_dist, init_y)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    grad_y = grads[1]
    self.aux_errors.append([np.linalg.norm(grad_y)])

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
    if self.cheap and self.vr:
      grads, pm_vr, exp_sto, exp_solver_sto = cheap_gradients_vr(
          self.random, *params, payoff_matrices, self.num_players, self.pm_vr,
          self.p, self.proj_grad,)
      self.pm_vr = pm_vr
      return grads, exp_sto, exp_solver_sto
    elif self.cheap and not self.vr:
      return cheap_gradients(self.random, *params, payoff_matrices,
                             self.num_players, self.p, self.proj_grad)
    else:
      return gradients(*params, payoff_matrices, self.num_players, self.p,
                       self.proj_grad)

  def exploitability(self, params, payoff_matrices):
    """Compute and return tsallis entropy regularized exploitability.

    Args:
      params: tuple of params (dist, y), see ate.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability of current dist
    """
    return exp.ate_exploitability(params, payoff_matrices, self.p)

  def euc_descent_step(self, params, grads, t):
    """Projected gradient descent on exploitability using Euclidean projection.

    Args:
      params: tuple of variables to be updated (dist, y)
      grads: tuple of variable gradients (grad_dist, grad_y)
      t: int, solver iteration (unused)
    Returns:
      new_params: tuple of update params (new_dist, new_y)
    """
    lr_dist, lr_y = self.lrs
    new_params = [params[0] - lr_dist * grads[0]]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_params += [params[1] - lr_y * grads[1]]
    new_params = euc_project(*new_params)
    return new_params

  def mirror_descent_step(self, params, grads, t):
    """Entropic mirror descent on exploitability.

    Args:
      params: tuple of variables to be updated (dist, y)
      grads: tuple of variable gradients (grad_dist, grad_y)
      t: int, solver iteration (unused)
    Returns:
      new_params: tuple of update params (new_dist, new_y)
    """
    lr_dist, lr_y = self.lrs
    new_params = [np.log(np.clip(params[0], 0, np.inf)) - lr_dist * grads[0]]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_params += [params[1] - lr_y * grads[1]]
    new_params = mirror_project(*new_params)
    return new_params


def gradients(dist, y, payoff_matrices, num_players, p=1, proj_grad=True):
  """Computes exploitablity gradient and aux variable gradients.

  Args:
    dist: 1-d np.array, current estimate of nash distribution
    y: 1-d np.array (same shape as dist), current estimate of payoff gradient
    payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    num_players: int, number of players, in case payoff_matrices is abbreviated
    p: float in [0, 1], Tsallis entropy-regularization --> 0 as p --> 0
    proj_grad: bool, if True, projects dist gradient onto simplex
  Returns:
    gradient of exploitability w.r.t. (dist, y) as tuple
    unregularized exploitability (stochastic estimate)
    tsallis regularized exploitability (stochastic estimate)
  """
  nabla = payoff_matrices[0].dot(dist)
  if p > 0:
    power = 1. / float(p)
    s = np.linalg.norm(y, ord=power)
    if s == 0:
      br = misc.uniform_dist(y)
    else:
      br = (y / s)**power
  else:
    power = np.inf
    s = np.linalg.norm(y, ord=power)
    br = np.zeros_like(dist)
    maxima = (y == s)
    br[maxima] = 1. / maxima.sum()

  unreg_exp = np.max(y) - y.dot(dist)
  br_inv_sparse = 1 - np.sum(br**(p + 1))
  dist_inv_sparse = 1 - np.sum(dist**(p + 1))
  entr_br = s / (p + 1) * br_inv_sparse
  entr_dist = s / (p + 1) * dist_inv_sparse
  reg_exp = y.dot(br - dist) + entr_br - entr_dist

  entr_br_vec = br_inv_sparse * br**(1 - p)
  entr_dist_vec = dist_inv_sparse * dist**(1 - p)

  policy_gradient = nabla - s * dist**p
  other_player_fx = (br - dist) + 1 / (p + 1) * (entr_br_vec - entr_dist_vec)

  other_player_fx_translated = payoff_matrices[1].dot(other_player_fx)
  grad_dist = -policy_gradient + (num_players - 1) * other_player_fx_translated
  if proj_grad:
    grad_dist = simplex.project_grad(grad_dist)
  grad_y = y - nabla

  return (grad_dist, grad_y), unreg_exp, reg_exp


def cheap_gradients(random, dist, y, payoff_matrices, num_players, p=1,
                    proj_grad=True):
  """Computes exploitablity gradient and aux variable gradients with samples.

  This implementation takes payoff_matrices as input so technically uses O(d^2)
  compute but only a single column of payoff_matrices is used to perform the
  update so can be re-implemented in O(d) if needed.

  Args:
    random: random number generator, np.random.RandomState(seed)
    dist: 1-d np.array, current estimate of nash distribution
    y: 1-d np.array (same shape as dist), current estimate of payoff gradient
    payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    num_players: int, number of players, in case payoff_matrices is abbreviated
    p: float in [0, 1], Tsallis entropy-regularization --> 0 as p --> 0
    proj_grad: bool, if True, projects dist gradient onto simplex
  Returns:
    gradient of exploitability w.r.t. (dist, y) as tuple
    unregularized exploitability (stochastic estimate)
    tsallis regularized exploitability (stochastic estimate)
  """
  action_1 = random.choice(dist.size, p=dist)
  nabla = payoff_matrices[0][:, action_1]
  if p > 0:
    power = 1. / float(p)
    s = np.linalg.norm(y, ord=power)
    if s == 0:
      br = misc.uniform_dist(y)
    else:
      br = (y / s)**power
  else:
    power = np.inf
    s = np.linalg.norm(y, ord=power)
    br = np.zeros_like(dist)
    maxima = (y == s)
    br[maxima] = 1. / maxima.sum()

  unreg_exp = np.max(y) - y.dot(dist)
  br_inv_sparse = 1 - np.sum(br**(p + 1))
  dist_inv_sparse = 1 - np.sum(dist**(p + 1))
  entr_br = s / (p + 1) * br_inv_sparse
  entr_dist = s / (p + 1) * dist_inv_sparse
  reg_exp = y.dot(br - dist) + entr_br - entr_dist

  entr_br_vec = br_inv_sparse * br**(1 - p)
  entr_dist_vec = dist_inv_sparse * dist**(1 - p)

  policy_gradient = nabla - s * dist**p
  other_player_fx = (br - dist) + 1 / (p + 1) * (entr_br_vec - entr_dist_vec)

  action_u = random.choice(dist.size)  # uniform, ~importance sampling
  other_player_fx = dist.size * other_player_fx[action_u]
  other_player_fx_translated = payoff_matrices[1, :, action_u] * other_player_fx
  grad_dist = -policy_gradient + (num_players - 1) * other_player_fx_translated
  if proj_grad:
    grad_dist = simplex.project_grad(grad_dist)
  grad_y = y - nabla

  return (grad_dist, grad_y), unreg_exp, reg_exp


def cheap_gradients_vr(random, dist, y, payoff_matrices, num_players, pm_vr,
                       p=1, proj_grad=True, version=0):
  """Computes exploitablity gradient and aux variable gradients with samples.

  This implementation takes payoff_matrices as input so technically uses O(d^2)
  compute but only a single column of payoff_matrices is used to perform the
  update so can be re-implemented in O(d) if needed.

  Args:
    random: random number generator, np.random.RandomState(seed)
    dist: 1-d np.array, current estimate of nash distribution
    y: 1-d np.array (same shape as dist), current estimate of payoff gradient
    payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    num_players: int, number of players, in case payoff_matrices is abbreviated
    pm_vr: approximate payoff_matrix for variance reduction
    p: float in [0, 1], Tsallis entropy-regularization --> 0 as p --> 0
    proj_grad: bool, if True, projects dist gradient onto simplex
    version: int, default 0, two options for variance reduction
  Returns:
    gradient of exploitability w.r.t. (dist, y) as tuple
    unregularized exploitability (stochastic estimate)
    tsallis regularized exploitability (stochastic estimate)
  """
  if pm_vr is None:
    raise ValueError("pm_vr must be np.array of shape (num_strats, num_strats)")
  if (not isinstance(version, int)) or (version < 0) or (version > 1):
    raise ValueError("version must be non-negative int < 2")

  action_1 = random.choice(dist.size, p=dist)
  nabla = payoff_matrices[0][:, action_1]
  if p > 0:
    power = 1. / float(p)
    s = np.linalg.norm(y, ord=power)
    if s == 0:
      br = misc.uniform_dist(y)
    else:
      br = (y / s)**power
  else:
    power = np.inf
    s = np.linalg.norm(y, ord=power)
    br = np.zeros_like(dist)
    maxima = (y == s)
    br[maxima] = 1. / maxima.sum()

  unreg_exp = np.max(y) - y.dot(dist)
  br_inv_sparse = 1 - np.sum(br**(p + 1))
  dist_inv_sparse = 1 - np.sum(dist**(p + 1))
  entr_br = s / (p + 1) * br_inv_sparse
  entr_dist = s / (p + 1) * dist_inv_sparse
  reg_exp = y.dot(br - dist) + entr_br - entr_dist

  entr_br_vec = br_inv_sparse * br**(1 - p)
  entr_dist_vec = dist_inv_sparse * dist**(1 - p)

  policy_gradient = nabla - s * dist**p
  other_player_fx = (br - dist) + 1 / (p + 1) * (entr_br_vec - entr_dist_vec)

  if version == 0:
    other_player_fx_translated = pm_vr.dot(other_player_fx)
    action_u = random.choice(dist.size)  # uniform, ~importance sampling
    other_player_fx = other_player_fx[action_u]
    pm_mod = dist.size * (payoff_matrices[1, :, action_u] - pm_vr[:, action_u])
    other_player_fx_translated += pm_mod * other_player_fx
  elif version == 1:
    other_player_fx_translated = np.sum(pm_vr, axis=1)
    action_u = random.choice(dist.size)  # uniform, ~importance sampling
    other_player_fx = other_player_fx[action_u]
    pm_mod = dist.size * payoff_matrices[1, :, action_u]
    r = dist.size * pm_vr[:, action_u]
    other_player_fx_translated += pm_mod * other_player_fx - r

  grad_dist = -policy_gradient + (num_players - 1) * other_player_fx_translated
  if proj_grad:
    grad_dist = simplex.project_grad(grad_dist)
  grad_y = y - nabla

  if version == 0:
    pm_vr[:, action_u] = payoff_matrices[1, :, action_u]
  elif version == 1:
    pm_vr[:, action_u] = payoff_matrices[1, :, action_u] * other_player_fx

  return (grad_dist, grad_y), pm_vr, unreg_exp, reg_exp


def euc_project(dist, y):
  """Project variables onto their feasible sets (euclidean proj for dist).

  Args:
    dist: 1-d np.array, current estimate of nash distribution
    y: 1-d np.array (same shape as dist), current estimate of payoff gradient
  Returns:
    projected variables (dist, y) as tuple
  """
  dist = simplex.euclidean_projection_onto_simplex(dist)
  y = np.clip(y, 0., np.inf)

  return dist, y


def mirror_project(dist, y):
  """Project variables onto their feasible sets (softmax for dist).

  Args:
    dist: 1-d np.array, current estimate of nash distribution
    y: 1-d np.array (same shape as dist), current estimate of payoff gradient
  Returns:
    projected variables (dist, y) as tuple
  """
  dist = special.softmax(dist)
  y = np.clip(y, 0., np.inf)

  return dist, y
