# Copyright 2022 DeepMind Technologies Limited
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
"""Regret-Matching Algorithm.

This is an N-player implementation of the regret-matching algorithm described in
Hart & Mas-Colell 2000:
https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00153
"""

import numpy as np
from open_spiel.python.algorithms import nfg_utils


# Start with initial regrets of 1 / denom
INITIAL_REGRET_DENOM = 1e6


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


def _regret_matching_step(payoff_tensors, strategies, regrets, gamma):
  """Does one step of the projected replicator dynamics algorithm.

  Args:
    payoff_tensors: List of payoff tensors for each player.
    strategies: List of the strategies used by each player.
    regrets: List of cumulative regrets used by each player.
    gamma: Minimum exploratory probability term.

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
    regrets[player] += values_per_strategy - average_return

    updated_strategy = regrets[player].copy()
    updated_strategy[updated_strategy < 0] = 0.0
    sum_regret = updated_strategy.sum()
    uniform_strategy = np.ones(len(updated_strategy)) / len(updated_strategy)

    if sum_regret > 0:
      updated_strategy /= sum_regret
      updated_strategy = gamma * uniform_strategy + (1 -
                                                     gamma) * updated_strategy
    else:
      updated_strategy = uniform_strategy

    new_strategies.append(updated_strategy)
  return new_strategies


def regret_matching(payoff_tensors,
                    initial_strategies=None,
                    iterations=int(1e5),
                    gamma=1e-6,
                    average_over_last_n_strategies=None,
                    **unused_kwargs):
  """Runs regret-matching for the stated number of iterations.

  Args:
    payoff_tensors: List of payoff tensors for each player.
    initial_strategies: Initial list of the strategies used by each player, if
      any. Could be used to speed up the search by providing a good initial
      solution.
    iterations: Number of algorithmic steps to take before returning an answer.
    gamma: Minimum exploratory probability term.
    average_over_last_n_strategies: Running average window size for average
      policy computation. If None, use the whole trajectory.
    **unused_kwargs: Convenient way of exposing an API compatible with other
      methods with possibly different arguments.

  Returns:
    RM-computed strategies.
  """
  number_players = len(payoff_tensors)
  # Number of actions available to each player.
  action_space_shapes = payoff_tensors[0].shape

  # If no initial starting position is given, start with uniform probabilities.
  new_strategies = initial_strategies or [
      np.ones(action_space_shapes[k]) / action_space_shapes[k]
      for k in range(number_players)
  ]

  regrets = [
      np.ones(action_space_shapes[k]) / INITIAL_REGRET_DENOM
      for k in range(number_players)
  ]

  averager = nfg_utils.StrategyAverager(number_players, action_space_shapes,
                                        average_over_last_n_strategies)
  averager.append(new_strategies)

  for _ in range(iterations):
    new_strategies = _regret_matching_step(payoff_tensors, new_strategies,
                                           regrets, gamma)
    averager.append(new_strategies)
  return averager.average_strategies()
