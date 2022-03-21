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

"""Treatment of iterates and gradients over the simplex."""

from absl import logging  # pylint:disable=unused-import

import numpy as np


def grad_norm(dist, grad, eps=1e-8, simplex_tol=1e-9):
  """Compute norm of gradient projected onto the tangent space of simplex.

  *assumes context is gradient descent (not ascent)

  Args:
    dist: np.array, distribution
    grad: np.array, gradient (same shape as distribution)
    eps: float, elements of dist in [eps, 1 - eps] are considered to be in the
      interior of the simplex. gradients on border of simplex
    simplex_tol: float, tolerance for checking if a point lies on the simplex,
      sum(vec) <= 1 + simplex_tol and all(vec > -simplex_tol). should be smaller
      than eps descent steps or points that are "leaving" simplex will be
      mislabeled
  Returns:
    float, norm of projected gradient
  """
  if simplex_tol >= eps:
    raise ValueError("simplex_tol should be less than eps")
  grad_proj = project_grad(grad)
  g_norm = np.linalg.norm(grad_proj)
  if g_norm > 0:
    # take a gradient descent step in the direction grad_proj with len eps
    # to determine if the update is "leaving" the simplex
    dist -= eps * grad_proj / g_norm
    if not ((np.sum(dist) <= 1 + simplex_tol) and np.all(dist >= -simplex_tol)):
      g_norm = 0.
  return g_norm


def project_grad(g):
  """Project gradient onto tangent space of simplex."""
  return g - g.sum() / g.size


# Project to probability simplex
# Based on this paper:
# Projection onto the probability simplex: An efficient algorithm with a
# simple proof, and an application
# https://arxiv.org/pdf/1309.1541.pdf
def euclidean_projection_onto_simplex(y, eps=1e-3, subset=True):
  """O(n log n) Euclidean projection of y onto the simplex.

  Args:
    y: np.array
    eps: float, ensure x remains at least eps / dim away from facets of simplex
    subset: bool, whether to project onto a subset of the simplex defined by eps
  Returns:
    np.array, y projected onto the simplex
  """
  if np.all(y >= 0.) and np.abs(np.sum(y) - 1.) < 1e-8:
    return y
  d = len(y)
  u = sorted(y, reverse=True)
  sum_uj = 0.
  for j in range(d):
    sum_uj += u[j]
    tj = (1. - sum_uj) / (j + 1.)
    if u[j] + tj <= 0:
      rho = j - 1
      sum_uj = sum_uj - u[j]
      break
    else:
      rho = j
  lam = (1. - sum_uj) / (rho + 1.)
  x = np.array([max(y[i] + lam, 0.) for i in range(d)])
  if subset:
    scale = 1. - eps * float(d + 1) / d
    offset = eps / float(d)
    x = scale * x + offset
    x /= x.sum()
  return x
