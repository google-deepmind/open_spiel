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

"""Stochastic Gradient Descent (Adam) Approx. Nash Solver."""

from absl import logging  # pylint:disable=unused-import

import jax
import jax.numpy as jnp

import numpy as np

import optax

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as exp
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import updates


class Solver(updates.Solver):
  """Adam Solver."""

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
    self.update = lambda *args: self.descent_step(*args, eps=self.eps)

    self.opt = optax.adam(learning_rate=lrs[0])
    self.opt_state = self.opt.init(jnp.zeros(1))

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
      init_dist_i = simplex.project_to_interior(init_dist_i, self.eps)
      init_dist.append(init_dist_i)

    init_params = [
        jnp.array(dist_to_logits(init_dist_i)) for init_dist_i in init_dist
    ]

    self.opt_state = self.opt.init(init_params)

    return (init_dist,)

  def descent_step(self, params, grads, t, eps=0.):
    """Projected gradient descent on exploitability using Euclidean projection.

    Args:
      params: tuple of variables to be updated (dist,)
      grads: tuple of variable gradients (grad_dist,)
      t: int, solver iteration (unused)
      eps: float > 0, force all probabilities >= eps / dim(dist)
    Returns:
      new_params: tuple of update params (new_dist,)
    """
    del t
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

    return (new_dist,)

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return exploitability.

    Args:
      params: tuple of params (dist,), see sgd.gradients
      payoff_matrices: 2 dictionaries with keys as tuples of agents (i, j) and
        values of (2 x A x A) np.arrays, payoffs for each joint action. keys
        are sorted and arrays should be indexed in the same order
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


def logits_to_dist(logits):
  logits_ext = jnp.append(logits, 0.)
  payoff = jax.nn.softmax(logits_ext)
  return payoff


def dist_to_logits(dist):
  # dist[-1] = exp(logits[-1]) / Z = exp(0) / Z
  z = 1 / dist[-1]
  logits = jnp.log(dist[:-1] * z)
  return logits


def gradients(dist, payoff_matrices, num_players, temperature=0.,
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
