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
"""Quantal Response Equilibrium (QRE) Stochastic Approximate Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import numpy as np
from scipy import special

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as exp


class Solver(object):
  """QRE Solver."""

  def __init__(self, temperature=1., proj_grad=True, euclidean=False,
               cheap=False, lrs=(1e-2, 1e-1), exp_thresh=-1., vr=True,
               rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    if temperature < 0.:
      raise ValueError("temperature must be non-negative")
    self.num_players = None
    self.temperature = temperature
    self.proj_grad = proj_grad
    self.cheap = cheap
    self.vr = vr
    self.pm_vr = None
    self.rnd_init = rnd_init
    self.lrs = lrs
    self.exp_thresh = exp_thresh
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
    init_anneal_steps = 0
    if self.cheap and self.vr:
      self.pm_vr = np.zeros((num_strats, num_strats))
    return (init_dist, init_y, init_anneal_steps)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    grad_y = grads[1]
    self.aux_errors.append([np.linalg.norm(grad_y)])

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients (and exploitabilities) for all parameters.

    Args:
      params: tuple of params (dist, y, anneal_steps), see gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      tuple of gradients (grad_dist, grad_y, grad_anneal_steps), see gradients
      unregularized exploitability (stochastic estimate)
      tsallis regularized exploitability (stochastic estimate)
    """
    if self.cheap and self.vr:
      grads, pm_vr, exp_sto, exp_solver_sto = self.cheap_gradients_vr(
          self.random, *params, payoff_matrices, self.num_players, self.pm_vr,
          self.temperature, self.proj_grad,)
      self.pm_vr = pm_vr
      return grads, exp_sto, exp_solver_sto
    elif self.cheap and not self.vr:
      return self.cheap_gradients(self.random, *params, payoff_matrices,
                                  self.num_players, self.temperature,
                                  self.proj_grad)
    else:
      return self.gradients(*params, payoff_matrices, self.num_players,
                            self.temperature, self.proj_grad)

  def exploitability(self, params, payoff_matrices):
    """Compute and return shannon entropy regularized exploitability.

    Args:
      params: tuple of params (dist, y), see qre.gradients
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability of current dist
    """
    return exp.qre_exploitability(params, payoff_matrices, self.temperature)

  def gradients(self, dist, y, anneal_steps, payoff_matrices, num_players,
                temperature=0., proj_grad=True):
    """Computes exploitablity gradient and aux variable gradients.

    Args:
      dist: 1-d np.array, current estimate of nash distribution
      y: 1-d np.array (same shape as dist), current estimate of payoff gradient
      anneal_steps: int, elapsed num steps since last anneal
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
      num_players: int, number of players, in case payoff_matrices is
        abbreviated
      temperature: non-negative float, default 0.
      proj_grad: bool, if True, projects dist gradient onto simplex
    Returns:
      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple
      unregularized exploitability (stochastic estimate)
      tsallis regularized exploitability (stochastic estimate)
    """
    nabla = payoff_matrices[0].dot(dist)
    if temperature >= 1e-3:
      br = special.softmax(y / temperature)
      br_mat = (np.diag(br) - np.outer(br, br)) / temperature
      log_br_safe = np.clip(np.log(br), -1e5, 0)
      br_policy_gradient = nabla - temperature * (log_br_safe + 1)
    else:
      power = np.inf
      s = np.linalg.norm(y, ord=power)
      br = np.zeros_like(dist)
      maxima = (y >= s)
      br[maxima] = 1. / maxima.sum()
      br_mat = np.zeros((br.size, br.size))
      br_policy_gradient = np.zeros_like(br)

    unreg_exp = np.max(y) - y.dot(dist)
    entr_br = temperature * special.entr(br).sum()
    entr_dist = temperature * special.entr(dist).sum()
    reg_exp = y.dot(br - dist) + entr_br - entr_dist

    policy_gradient = np.array(nabla)
    if temperature > 0:
      log_dist_safe = np.clip(np.log(dist), -1e5, 0)
      policy_gradient -= temperature * (log_dist_safe + 1)
    other_player_fx = (br - dist) + br_mat.dot(br_policy_gradient)

    other_player_fx_translated = payoff_matrices[1].dot(other_player_fx)
    grad_dist = -policy_gradient
    grad_dist += (num_players - 1) * other_player_fx_translated
    if proj_grad:
      grad_dist = simplex.project_grad(grad_dist)
    grad_y = y - nabla

    _, lr_y = self.lrs
    if (reg_exp < self.exp_thresh) and (anneal_steps >= 1 / lr_y):
      self.temperature = np.clip(temperature / 2., 0., 1.)
      if self.temperature < 1e-3:
        self.temperature = 0.
      grad_anneal_steps = -anneal_steps
    else:
      grad_anneal_steps = 1

    return (grad_dist, grad_y, grad_anneal_steps), unreg_exp, reg_exp

  def cheap_gradients(self, random, dist, y, anneal_steps, payoff_matrices,
                      num_players, temperature=0., proj_grad=True):
    """Computes exploitablity gradient and aux variable gradients with samples.

    This implementation takes payoff_matrices as input so technically uses
    O(d^2) compute but only a single column of payoff_matrices is used to
    perform the update so can be re-implemented in O(d) if needed.

    Args:
      random: random number generator, np.random.RandomState(seed)
      dist: 1-d np.array, current estimate of nash distribution
      y: 1-d np.array (same shape as dist), current estimate of payoff gradient
      anneal_steps: int, elapsed num steps since last anneal
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
      num_players: int, number of players, in case payoff_matrices is
        abbreviated
      temperature: non-negative float, default 0.
      proj_grad: bool, if True, projects dist gradient onto simplex
    Returns:
      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple
      unregularized exploitability (stochastic estimate)
      tsallis regularized exploitability (stochastic estimate)
    """
    del anneal_steps
    action_1 = random.choice(dist.size, p=dist)
    nabla = payoff_matrices[0][:, action_1]
    if temperature >= 1e-3:
      br = special.softmax(y / temperature)
      br_mat = (np.diag(br) - np.outer(br, br)) / temperature
      br_policy_gradient = nabla - temperature * (np.log(br) + 1)
    else:
      power = np.inf
      s = np.linalg.norm(y, ord=power)
      br = np.zeros_like(dist)
      maxima = (y == s)
      br[maxima] = 1. / maxima.sum()
      br_mat = np.zeros((br.size, br.size))
      br_policy_gradient = np.zeros_like(br)

    unreg_exp = np.max(y) - y.dot(dist)
    entr_br = temperature * special.entr(br).sum()
    entr_dist = temperature * special.entr(dist).sum()
    reg_exp = y.dot(br - dist) + entr_br - entr_dist

    policy_gradient = nabla - temperature * (np.log(dist) + 1)
    other_player_fx = (br - dist) + br_mat.dot(br_policy_gradient)

    action_u = random.choice(dist.size)  # uniform, ~importance sampling
    other_player_fx = dist.size * other_player_fx[action_u]
    other_player_fx_translat = payoff_matrices[1, :, action_u] * other_player_fx
    grad_dist = -policy_gradient + (num_players - 1) * other_player_fx_translat
    if proj_grad:
      grad_dist = simplex.project_grad(grad_dist)
    grad_y = y - nabla

    return (grad_dist, grad_y, None), unreg_exp, reg_exp

  def cheap_gradients_vr(self, random, dist, y, anneal_steps, payoff_matrices,
                         num_players, pm_vr, temperature=0., proj_grad=True,
                         version=0):
    """Computes exploitablity gradient and aux variable gradients with samples.

    This implementation takes payoff_matrices as input so technically uses
    O(d^2) compute but only a single column of payoff_matrices is used to
    perform the update so can be re-implemented in O(d) if needed.

    Args:
      random: random number generator, np.random.RandomState(seed)
      dist: 1-d np.array, current estimate of nash distribution
      y: 1-d np.array (same shape as dist), current estimate of payoff gradient
      anneal_steps: int, elapsed num steps since last anneal
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
      num_players: int, number of players, in case payoff_matrices is
        abbreviated
      pm_vr: approximate payoff_matrix for variance reduction
      temperature: non-negative float, default 0.
      proj_grad: bool, if True, projects dist gradient onto simplex
      version: int, default 0, two options for variance reduction
    Returns:
      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple
      unregularized exploitability (stochastic estimate)
      tsallis regularized exploitability (stochastic estimate)
    """
    del anneal_steps
    if pm_vr is None:
      raise ValueError("pm_vr must be np.array of shape (num_strats,) * 2")
    if (not isinstance(version, int)) or (version < 0) or (version > 1):
      raise ValueError("version must be non-negative int < 2")

    action_1 = random.choice(dist.size, p=dist)
    nabla = payoff_matrices[0][:, action_1]
    if temperature >= 1e-3:
      br = special.softmax(y / temperature)
      br_mat = (np.diag(br) - np.outer(br, br)) / temperature
      br_policy_gradient = nabla - temperature * (np.log(br) + 1)
    else:
      power = np.inf
      s = np.linalg.norm(y, ord=power)
      br = np.zeros_like(dist)
      maxima = (y == s)
      br[maxima] = 1. / maxima.sum()
      br_mat = np.zeros((br.size, br.size))
      br_policy_gradient = np.zeros_like(br)

    unreg_exp = np.max(y) - y.dot(dist)
    entr_br = temperature * special.entr(br).sum()
    entr_dist = temperature * special.entr(dist).sum()
    reg_exp = y.dot(br - dist) + entr_br - entr_dist

    policy_gradient = nabla - temperature * (np.log(dist) + 1)
    other_player_fx = (br - dist) + br_mat.dot(br_policy_gradient)

    if version == 0:
      other_player_fx_translated = pm_vr.dot(other_player_fx)
      action_u = random.choice(dist.size)  # uniform, ~importance sampling
      other_player_fx = other_player_fx[action_u]
      m = dist.size
      pm_mod = m * (payoff_matrices[1, :, action_u] - pm_vr[:, action_u])
      other_player_fx_translated += pm_mod * other_player_fx
    elif version == 1:
      other_player_fx_translated = np.sum(pm_vr, axis=1)
      action_u = random.choice(dist.size)  # uniform, ~importance sampling
      other_player_fx = other_player_fx[action_u]
      pm_mod = dist.size * payoff_matrices[1, :, action_u]
      r = dist.size * pm_vr[:, action_u]
      other_player_fx_translated += pm_mod * other_player_fx - r

    grad_dist = -policy_gradient
    grad_dist += (num_players - 1) * other_player_fx_translated
    if proj_grad:
      grad_dist = simplex.project_grad(grad_dist)
    grad_y = y - nabla

    if version == 0:
      pm_vr[:, action_u] = payoff_matrices[1, :, action_u]
    elif version == 1:
      pm_vr[:, action_u] = payoff_matrices[1, :, action_u] * other_player_fx

    return (grad_dist, grad_y, None), pm_vr, unreg_exp, reg_exp

  def euc_descent_step(self, params, grads, t):
    """Projected gradient descent on exploitability using Euclidean projection.

    Args:
      params: tuple of variables to be updated (dist, y, anneal_steps)
      grads: tuple of variable gradients (grad_dist, grad_y, grad_anneal_steps)
      t: int, solver iteration
    Returns:
      new_params: tuple of update params (new_dist, new_y, new_anneal_steps)
    """
    lr_dist, lr_y = self.lrs
    new_params = [params[0] - lr_dist * grads[0]]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_params += [params[1] - lr_y * grads[1]]
    new_params = euc_project(*new_params)
    new_params += (params[2] + grads[2],)
    return new_params

  def mirror_descent_step(self, params, grads, t):
    """Entropic mirror descent on exploitability.

    Args:
      params: tuple of variables to be updated (dist, y, anneal_steps)
      grads: tuple of variable gradients (grad_dist, grad_y, grad_anneal_steps)
      t: int, solver iteration
    Returns:
      new_params: tuple of update params (new_dist, new_y, new_anneal_steps)
    """
    lr_dist, lr_y = self.lrs
    new_params = [np.log(np.clip(params[0], 0, np.inf)) - lr_dist * grads[0]]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_params += [params[1] - lr_y * grads[1]]
    new_params = mirror_project(*new_params)
    new_params += (params[2] + grads[2],)
    return new_params


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
