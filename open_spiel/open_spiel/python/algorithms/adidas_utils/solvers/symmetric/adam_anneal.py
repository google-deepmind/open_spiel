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

"""Stochastic Gradient Descent (Adam) Approx. Nash Solver w/ Annealing."""

from absl import logging  # pylint:disable=unused-import

import jax
import jax.numpy as jnp

import numpy as np

import optax

from scipy import special

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as exp


class Solver(object):
  """Adam Solver with temperature annealing."""

  def __init__(self, temperature=1., proj_grad=True, lrs=(1e-2, 1e-1),
               exp_thresh=-1., rnd_init=False, seed=None, **kwargs):
    """Ctor."""
    del kwargs
    if temperature < 0.:
      raise ValueError("temperature must be non-negative")
    self.num_players = None
    self.temperature = temperature
    self.proj_grad = proj_grad
    self.rnd_init = rnd_init
    self.lrs = lrs
    self.num_estimates = 2
    self.exp_thresh = exp_thresh
    self.has_aux = True
    self.aux_errors = []

    self.update = self.descent_step

    self.opt = optax.adam(learning_rate=lrs[0])
    self.opt_state = self.opt.init(jnp.zeros(1))

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

    init_params = jnp.array(dist_to_logits(init_dist))

    self.opt_state = self.opt.init(init_params)

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
      shannon entropy regularized exploitability (stochastic estimate)
    """
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

  def gradients(self, dist: np.ndarray, y: np.ndarray, anneal_steps: int,
                payoff_matrices, num_players,
                temperature=0., proj_grad=True
                ) -> tuple[tuple[np.ndarray, np.ndarray, int], float, float]:
    """Computes exploitablity gradient and aux variable gradients.

    Args:
      dist: 1-d np.array, current estimate of nash distribution
      y: 1-d np.array (same shape as dist), current estimate of payoff gradient
      anneal_steps: int, elapsed num steps since last anneal
      payoff_matrices: 2 (>=2 x A x A) np.arrays, payoffs for each joint action
      num_players: int, number of players, in case payoff_matrices is
        abbreviated
      temperature: non-negative float, default 0.
      proj_grad: bool, if True, projects dist gradient onto simplex
    Returns:
      gradient of exploitability w.r.t. (dist, anneal_steps) as tuple
      unregularized exploitability (stochastic estimate)
      shannon entropy regularized exploitability (stochastic estimate)
    """

    grad_dist = loss_gradients(dist, payoff_matrices, num_players, temperature,
                               proj_grad)[0][0]

    a = 0  # 2 samples (a, b) needed for unbiased estimation
    p_0 = 0  # player 0 index
    nabla = payoff_matrices[a][p_0].dot(dist)
    grad_y = y - nabla

    if temperature >= 1e-3:
      br = special.softmax(y / temperature)
    else:
      power = np.inf
      s = np.linalg.norm(y, ord=power)
      br = np.zeros_like(dist)
      maxima = (y == s)
      br[maxima] = 1. / maxima.sum()

    unreg_exp = np.max(y) - y.dot(dist)
    entr_br = temperature * special.entr(br).sum()
    entr_dist = temperature * special.entr(dist).sum()
    reg_exp = y.dot(br - dist) + entr_br - entr_dist

    if reg_exp < self.exp_thresh:
      self.temperature = np.clip(temperature / 2., 0., np.inf)
      grad_anneal_steps = -anneal_steps
    else:
      grad_anneal_steps = 1

    return (grad_dist, grad_y, grad_anneal_steps), unreg_exp, reg_exp

  def descent_step(self, params, grads, t, eps=0.):
    """Gradient descent on exploitability wrt logits.

    Args:
      params: tuple of variables to be updated (dist, y, anneal_steps)
      grads: tuple of variable gradients (grad_dist, grad_y, grad_anneal_steps)
      t: int, solver iteration
      eps: float > 0, force all probabilities >= eps / dim(dist) (unused)
    Returns:
      new_params: tuple of update params (new_dist, new_y, new_anneal_steps)
    """
    del eps

    dist = params[0]
    grads_dist = grads[0]

    dist_jnp = jnp.array(dist)
    grads_dist_jnp = jnp.array(grads_dist)

    # map dist to logits and grads to grad_logits using jacobian
    logits = dist_to_logits(dist)
    grads_logits = jax.jvp(dist_to_logits, [dist_jnp], [grads_dist_jnp])[1]

    opt_updates, self.opt_state = self.opt.update(grads_logits,
                                                  self.opt_state,
                                                  logits)

    new_logits = optax.apply_updates(logits, opt_updates)

    new_dist = logits_to_dist(new_logits)
    new_dist = np.array(new_dist)

    lr_y = self.lrs[1]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_y = params[1] - lr_y * grads[1]

    new_anneal_steps = params[2] + grads[2]

    return (new_dist, new_y, new_anneal_steps)


def logits_to_dist(logits):
  logits_ext = jnp.append(logits, 0.)
  payoff = jax.nn.softmax(logits_ext)
  return payoff


def dist_to_logits(dist, eps=1e-8):
  # dist[-1] = exp(logits[-1]) / Z = exp(0) / Z
  z = 1 / jnp.clip(dist[-1], eps, 1.)
  logits = jnp.log(jnp.clip(dist[:-1] * z, eps, np.inf))
  return logits


def loss_gradients(dist, payoff_matrices, num_players, temperature=0.,
                   proj_grad=True):
  """Computes exploitablity gradient.

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
  # if consulting paper https://arxiv.org/abs/2310.06689, code assumes eta = 1
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
