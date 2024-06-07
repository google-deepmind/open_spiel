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
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as exp


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
    if len(num_strats) != num_players:
      raise ValueError("Must specify num strategies for each player")

    init_dist = []
    for num_strats_i in num_strats:
      if self.rnd_init:
        init_dist_i = self.random.rand(num_strats_i)
      else:
        init_dist_i = np.ones(num_strats_i)
      init_dist_i /= init_dist_i.sum()
      init_dist.append(init_dist_i)

    init_params = [
        jnp.array(dist_to_logits(init_dist_i)) for init_dist_i in init_dist
    ]

    self.opt_state = self.opt.init(init_params)

    init_y = [np.zeros_like(dist_i) for dist_i in init_dist]
    init_anneal_steps = 0

    return (init_dist, init_y, init_anneal_steps)

  def record_aux_errors(self, grads):
    """Record errors for the auxiliary variables."""
    grad_y = grads[1]
    # call ravel in case use y to track entire payoff matrices in future
    grad_y_flat = np.concatenate([np.ravel(g) for g in grad_y])
    self.aux_errors.append([np.linalg.norm(grad_y_flat)])

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients (and exploitabilities) for all parameters.

    Args:
      params: tuple of params (dist, y, anneal_steps), see gradients
      payoff_matrices: 2 dictionaries with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
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
      payoff_matrices: 2 dictionaries with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    Returns:
      float, exploitability of current dist
    """
    return exp.qre_exploitability(params, payoff_matrices, self.temperature)

  def gradients(self, dist: np.ndarray, y: np.ndarray, anneal_steps: int,
                payoff_matrices, num_players,
                temperature=0., proj_grad=True
                ) -> tuple[tuple[list[np.ndarray], list[np.ndarray], int],
                           float,
                           float]:
    """Computes exploitablity gradient and aux variable gradients.

    Args:
      dist: list of 1-d np.arrays, current estimate of nash distribution
      y: list 1-d np.arrays (same shape as dist), current est. of payoff
        gradient
      anneal_steps: int, elapsed num steps since last anneal
      payoff_matrices: 2 dictionaries with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
      num_players: int, number of players, in case payoff_matrices is
        abbreviated
      temperature: non-negative float, default 0.
      proj_grad: bool, if True, projects dist gradient onto simplex
    Returns:
      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple
      unregularized exploitability (stochastic estimate)
      shannon entropy regularized exploitability (stochastic estimate)
    """

    grad_dist = loss_gradients(dist, payoff_matrices, num_players, temperature,
                               proj_grad)[0][0]

    grad_y = []
    unreg_exp = []
    reg_exp = []
    for i in range(num_players):

      nabla_i = np.zeros_like(dist[i])
      for j in range(num_players):
        if j == i:
          continue
        if i < j:
          hess_i_ij = 0.5 * payoff_matrices[0][(i, j)][0]
          hess_i_ij += 0.5 * payoff_matrices[1][(i, j)][0]
        else:
          hess_i_ij = 0.5 * payoff_matrices[0][(j, i)][1].T
          hess_i_ij += 0.5 * payoff_matrices[1][(j, i)][1].T

        nabla_ij = hess_i_ij.dot(dist[j])
        nabla_i += nabla_ij / float(num_players - 1)

      grad_y.append(y[i] - nabla_i)

      if temperature >= 1e-3:
        br_i = special.softmax(y[i] / temperature)
      else:
        power = np.inf
        s_i = np.linalg.norm(y[i], ord=power)
        br_i = np.zeros_like(dist[i])
        maxima_i = (y[i] == s_i)
        br_i[maxima_i] = 1. / maxima_i.sum()

      unreg_exp.append(np.max(y[i]) - y[i].dot(dist[i]))

      entr_br_i = temperature * special.entr(br_i).sum()
      entr_dist_i = temperature * special.entr(dist[i]).sum()

      reg_exp.append(y[i].dot(br_i - dist[i]) + entr_br_i - entr_dist_i)

    unreg_exp_mean = np.mean(unreg_exp)
    reg_exp_mean = np.mean(reg_exp)

    _, lr_y = self.lrs
    if (reg_exp_mean < self.exp_thresh) and (anneal_steps >= 1 / lr_y):
      self.temperature = np.clip(temperature / 2., 0., np.inf)
      grad_anneal_steps = -anneal_steps
    else:
      grad_anneal_steps = 1

    return (grad_dist, grad_y, grad_anneal_steps), unreg_exp_mean, reg_exp_mean

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

    dist_jnp = [jnp.array(dist_i) for dist_i in dist]
    grads_dist_jnp = [jnp.array(grad_i) for grad_i in grads_dist]

    # map dist to logits and grads to grad_logits using jacobian
    logits = [dist_to_logits(dist_i) for dist_i in params[0]]
    grads_logits = [
        jax.jvp(dist_to_logits, [dist_i], [grads_i])[1]
        for dist_i, grads_i in zip(dist_jnp, grads_dist_jnp)
    ]

    opt_updates, self.opt_state = self.opt.update(grads_logits,
                                                  self.opt_state,
                                                  logits)

    new_logits = optax.apply_updates(logits, opt_updates)

    new_dist = [logits_to_dist(logits) for logits in new_logits]
    new_dist = [np.array(dist_i) for dist_i in new_dist]

    lr_y = self.lrs[1]
    lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
    new_y = []
    for y_i, y_grad_i in zip(params[1], grads[1]):
      new_y_i = y_i - lr_y * y_grad_i
      new_y_i = np.clip(new_y_i, 0., np.inf)
      new_y.append(new_y_i)

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
    dist: list of 1-d np.arrays, current estimate of nash distribution
    payoff_matrices: 2 dictionaries with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
    num_players: int, number of players, in case payoff_matrices is abbreviated
    temperature: non-negative float, default 0.
    proj_grad: bool, if True, projects dist gradient onto simplex
  Returns:
    gradient of exploitability w.r.t. (dist) as tuple
    unregularized exploitability (stochastic estimate)
    shannon regularized exploitability (stochastic estimate)
  """
  # first compute projected gradients (for every player, for each sample a & b)
  # if consulting paper https://arxiv.org/abs/2310.06689, code assumes eta_k = 1
  tau = temperature

  pgs = []
  for i in range(num_players):

    pg_i_a = np.zeros_like(dist[i])
    pg_i_b = np.zeros_like(dist[i])

    for j in range(num_players):
      if j == i:
        continue
      if i < j:
        hess_i_ij_a = payoff_matrices[0][(i, j)][0]
        hess_i_ij_b = payoff_matrices[1][(i, j)][0]
      else:
        hess_i_ij_a = payoff_matrices[0][(j, i)][1].T
        hess_i_ij_b = payoff_matrices[1][(j, i)][1].T

      pg_i_a_est = simplex.project_grad(hess_i_ij_a.dot(dist[j]))
      pg_i_b_est = simplex.project_grad(hess_i_ij_b.dot(dist[j]))

      pg_i_a += pg_i_a_est / float(num_players - 1)
      pg_i_b += pg_i_b_est / float(num_players - 1)

    pgs.append((pg_i_a, pg_i_b))

  # then construct unbiased stochastic gradient
  grad_dist = []
  unreg_exp = []
  reg_exp = []

  for i in range(num_players):

    grad_dist_i = np.zeros_like(dist[i])

    for j in range(num_players):
      pg_j_a = pgs[j][0]
      pg_j_b = pgs[j][1]
      if tau > 0.:
        log_dist_safe = np.clip(np.log(dist[j]), -40, 0)
        entr_grad_proj = simplex.project_grad(-tau * (log_dist_safe + 1))
      else:
        entr_grad_proj = 0.
      pg_j_a_entr = pg_j_a + entr_grad_proj
      pg_j_b_entr = pg_j_b + entr_grad_proj

      if j == i:
        if tau > 0.:
          hess_j_ij_a = -tau * np.diag(1. / dist[j])
        else:
          hess_j_ij_a = np.diag(np.zeros_like(dist[j]))
        unreg_exp_i = np.dot(pg_j_a, pg_j_b)
        reg_exp_i = np.dot(pg_j_a_entr, pg_j_b_entr)
        unreg_exp.append(unreg_exp_i)
        reg_exp.append(reg_exp_i)
      elif i < j:
        hess_j_ij_a = payoff_matrices[0][(i, j)][1]
      else:
        hess_j_ij_a = payoff_matrices[0][(j, i)][0].T

      grad_dist_i += 2. * hess_j_ij_a.dot(pg_j_b_entr)

    if proj_grad:
      grad_dist_i = simplex.project_grad(grad_dist_i)

    grad_dist.append(grad_dist_i)

  return (grad_dist,), np.mean(unreg_exp), np.mean(reg_exp)
