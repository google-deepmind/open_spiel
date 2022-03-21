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
"""Exploitability measurement utils."""

from absl import logging  # pylint:disable=unused-import

import numpy as np
from scipy import special

from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability


class Solver(object):
  """Generic Solver."""

  def __init__(self, proj_grad=True, euclidean=False, rnd_init=False,
               seed=None):
    """Ctor."""
    self.num_players = None
    self.proj_grad = proj_grad
    self.rnd_init = rnd_init
    self.lrs = (None, None, None)
    self.has_aux = False

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
    return (init_dist,)

  def compute_gradients(self, params, payoff_matrices):
    """Compute and return gradients for all parameters.

    Args:
      params: e.g., tuple of params (dist,)
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      eg., tuple of gradients (grad_dist,)
    """
    raise NotImplementedError("Should be implemented by specific solver.")

  def exploitability(self, params, payoff_matrices):
    """Compute and return exploitability that solver is minimizing.

    Args:
      params: e.g., tuple of params (dist,)
      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action
    Returns:
      float, exploitability of current dist
    """
    return exploitability.unreg_exploitability(params, payoff_matrices)

  def euc_descent_step(self, params, grads, t):
    """Projected gradient descent on exploitability using Euclidean projection.

    Args:
      params: tuple of variables to be updated (dist,)
      grads: tuple of variable gradients (grad_dist,)
      t: int, solver iteration
    Returns:
      new_params: tuple of update params (new_dist,)
    """
    del t
    new_params = params[0] - self.lrs[0] * grads[0]
    new_params = simplex.euclidean_projection_onto_simplex(new_params)
    return (new_params,)

  def mirror_descent_step(self, params, grads, t):
    """Entropic mirror descent on exploitability.

    Args:
      params: tuple of variables to be updated (dist)
      grads: tuple of variable gradients (grad_dist)
      t: int, solver iteration
    Returns:
      new_params: tuple of update params (new_dist)
    """
    del t
    dist = np.clip(params[0], 0, np.inf)
    return (special.softmax(np.log(dist) - self.lrs[0] * grads[0]),)
