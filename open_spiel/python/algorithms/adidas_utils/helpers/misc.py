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

"""Miscellaneous utils."""

from absl import logging  # pylint:disable=unused-import

import numpy as np


def uniform_dist(x):
  """Returns a uniform distribution with same shape as the given numpy array.

  Args:
    x: numpy array
  Returns:
    constant numpy array of same shape as input x, sums to 1
  """
  return np.ones_like(x) / float(x.size)


def argmax(random, z):
  """Returns argmax of flattened z with ties split randomly.

  Args:
    random: Random number generator, e.g., np.random.RandomState()
    z: np.array
  Returns:
    integer representing index of argmax
  """
  inds = np.arange(z.size)
  random.shuffle(inds)
  z_shuffled = z[inds]
  ind_max = np.argmax(z_shuffled)
  return inds[ind_max]


def pt_reduce(payoff_tensor, strats, remove_players):
  """Computes possible payoffs for remove_players with others' strats fixed.

  This is equivalent to the Jacobian of the payoff w.r.t. remove_players:
  sum_{a...z} A_k * x_1a * ... * x_nz for player k.
  Args:
    payoff_tensor: a single player k's payoff tensor, i.e.,
      a num action x ... x num action (num player) np.array
    strats: list of distributions over strategies for each player
    remove_players: players to NOT sum over in expectation
  Returns:
    payoff tensor of shape: num_action x ... x num_action,
      num_action for each player in remove_players
  """
  result = np.copy(payoff_tensor)
  result_dims = list(range(len(result.shape)))
  other_player_idxs = list(result_dims)
  for remove_player in remove_players:
    other_player_idxs.remove(remove_player)
  for other_player_idx in other_player_idxs:
    new_result_dims = list(result_dims)
    new_result_dims.remove(other_player_idx)
    result = np.einsum(result, result_dims, strats[other_player_idx],
                       [other_player_idx], new_result_dims)
    result_dims = new_result_dims
  return result


def isnan(x):
  """Checks for NaN's in nested objects."""
  if isinstance(x, float):
    return np.isnan(x)
  elif isinstance(x, int):
    return np.isnan(x)
  elif isinstance(x, np.ndarray):
    return np.any(np.isnan(x))
  elif isinstance(x, list):
    return np.any([isnan(xi) for xi in x])
  elif isinstance(x, tuple):
    return np.any([isnan(xi) for xi in x])
  elif isinstance(x, dict):
    return np.any([isnan(xi) for xi in x.values()])
  else:
    typ = repr(type(x))
    err_string = 'type(x)={:s} not recognized when checking for NaN'.format(typ)
    raise NotImplementedError(err_string)
