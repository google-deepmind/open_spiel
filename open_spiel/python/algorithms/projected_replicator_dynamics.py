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

"""Projected Replicator Dynamics Algorithm.

This is an N-player implementation of the Projected Replicator Dynamics
algorithm described in Lanctot et al., 2017: https://arxiv.org/abs/1711.00832.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _partial_multi_dot(player_payoff_tensor, strategies, index_avoided):
  """Computes a generalized dot product avoiding one dimension.

  This is used to directly get the expected return of a given action, given
  other players' strategies, for the player indexed by index_avoided.
  Note that the numpy.dot function is used to compute this product, as it ended
  up being (Slightly) faster in performance tests than np.tensordot. Using the
  reduce function proved slower for both np.dot and np.tensordot.

  Args:
    player_payoff_tensor: payoff tensor for player[index_avoided], of dimension
      (dim(vector[0]), dim(vector[1]), ..., dim(vector[-1])).
    strategies: Meta strategy probabilities for each player.
    index_avoided: Player for which we do not compute the dot product.

  Returns:
    Vector of expected returns for each action of player [the player indexed by
      index_avoided].
  """
  new_axis_order = [index_avoided] + [
      i for i in range(len(strategies)) if (i != index_avoided)
  ]
  accumulator = np.transpose(player_payoff_tensor, new_axis_order)
  for i in range(len(strategies) - 1, -1, -1):
    if i != index_avoided:
      accumulator = np.dot(accumulator, strategies[i])
  return accumulator


def _project_distribution(updated_strategy, gamma):
  """Projects the distribution in updated_x to have minimal probabilities.

  Minimal probabilities are set as gamma, and the probabilities are then
  renormalized to sum to 1.

  Args:
    updated_strategy: New distribution value after being updated by update rule.
    gamma: minimal probability value when divided by number of actions.

  Returns:
    Projected distribution.
  """
  # Epsilon approximation of L2-norm projection onto the Delta_gamma space.
  updated_strategy[updated_strategy < gamma] = gamma
  updated_strategy = updated_strategy / np.sum(updated_strategy)
  return updated_strategy


def _approx_simplex_projection(updated_strategy, gamma=0.0):
  """Approximately projects the distribution in updated_x to have minimal probabilities.

  Minimal probabilities are set as gamma, and the probabilities are then
  renormalized to sum to 1.

  Args:
    updated_strategy: New distribution value after being updated by update rule.
    gamma: minimal probability value when divided by number of actions.

  Returns:
    Projected distribution.
  """
  # Epsilon approximation of L2-norm projection onto the Delta_gamma space.
  updated_strategy[updated_strategy < gamma] = gamma
  updated_strategy = updated_strategy / np.sum(updated_strategy)
  return updated_strategy


def _simplex_projection(updated_strategy, gamma=0.0):
  """Project updated_strategy on the closest point in L2-norm on gamma-simplex.

  Based on: https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf

  Args:
    updated_strategy: New distribution value after being updated by update rule.
    gamma: minimal probability value when divided by number of actions.

  Returns:
    Projected distribution

  Algorithm description:
  It aims to find a scalar lam to be substracted by each dimension of v
  with the restriction that the resulted quantity should lie in [gamma, 1]
  until the resulted vector summed up to 1
  Example: [0.4, 0.7, 0.6], 0.2 -- > find lam=0.25
            --> [max(0.4-0.25, 0.2), max(0.7-0.25, 0.2), max(0.6-0.25, 0.2)]
            --> [0.2,  0.45, 0.35]
  """

  n = len(updated_strategy)
  idx = np.arange(1, n + 1)
  u = np.sort(updated_strategy)[::-1]
  u_tmp = (1 - np.cumsum(u) - (n - idx) * gamma) / idx
  rho = np.searchsorted(u + u_tmp <= gamma, True)
  return np.maximum(updated_strategy + u_tmp[rho - 1], gamma)


def _projected_replicator_dynamics_step(payoff_tensors, strategies, dt, gamma,
                                        use_approx=False):
  """Does one step of the projected replicator dynamics algorithm.

  Args:
    payoff_tensors: List of payoff tensors for each player.
    strategies: List of the strategies used by each player.
    dt: Update amplitude term.
    gamma: Minimum exploratory probability term.
    use_approx: use approximate simplex projection.

  Returns:
    A list of updated strategies for each player.
  """

  # TODO(author4): Investigate whether this update could be fully vectorized.
  new_strategies = []
  for player in range(len(payoff_tensors)):
    current_payoff_tensor = payoff_tensors[player]
    current_strategy = strategies[player]

    values_per_strategy = _partial_multi_dot(current_payoff_tensor, strategies,
                                             player)
    average_return = np.dot(values_per_strategy, current_strategy)
    delta = current_strategy * (values_per_strategy - average_return)

    updated_strategy = current_strategy + dt * delta
    updated_strategy = (
        _approx_simplex_projection(updated_strategy, gamma) if use_approx
        else _simplex_projection(updated_strategy, gamma))
    new_strategies.append(updated_strategy)
  return new_strategies


def projected_replicator_dynamics(payoff_tensors,
                                  prd_initial_strategies=None,
                                  prd_iterations=int(1e5),
                                  prd_dt=1e-3,
                                  prd_gamma=1e-6,
                                  average_over_last_n_strategies=None,
                                  use_approx=False,
                                  **unused_kwargs):
  """The Projected Replicator Dynamics algorithm.

  Args:
    payoff_tensors: List of payoff tensors for each player.
    prd_initial_strategies: Initial list of the strategies used by each player,
      if any. Could be used to speed up the search by providing a good initial
      solution.
    prd_iterations: Number of algorithmic steps to take before returning an
      answer.
    prd_dt: Update amplitude term.
    prd_gamma: Minimum exploratory probability term.
    average_over_last_n_strategies: Running average window size for average
      policy computation. If None, use the whole trajectory.
    use_approx: use the approximate simplex projection.
    **unused_kwargs: Convenient way of exposing an API compatible with other
      methods with possibly different arguments.

  Returns:
    PRD-computed strategies.
  """
  number_players = len(payoff_tensors)
  # Number of actions available to each player.
  action_space_shapes = payoff_tensors[0].shape

  # If no initial starting position is given, start with uniform probabilities.
  new_strategies = prd_initial_strategies or [
      np.ones(action_space_shapes[k]) / action_space_shapes[k]
      for k in range(number_players)
  ]

  average_over_last_n_strategies = average_over_last_n_strategies or prd_iterations

  meta_strategy_window = []
  for i in range(prd_iterations):
    new_strategies = _projected_replicator_dynamics_step(
        payoff_tensors, new_strategies, prd_dt, prd_gamma, use_approx)
    if i >= prd_iterations - average_over_last_n_strategies:
      meta_strategy_window.append(new_strategies)
  average_new_strategies = np.mean(meta_strategy_window, axis=0)
  return average_new_strategies
