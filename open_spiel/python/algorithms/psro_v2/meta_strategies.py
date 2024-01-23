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

"""Meta-strategy solvers for PSRO."""

import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.algorithms import regret_matching
import pyspiel


EPSILON_MIN_POSITIVE_PROBA = 1e-8


def uniform_strategy(solver, return_joint=False):
  """Returns a Random Uniform distribution on policies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  policy_lengths = [len(pol) for pol in policies]
  result = [np.ones(pol_len) / pol_len for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def softmax_on_range(number_policies):
  x = np.array(list(range(number_policies)))
  x = np.exp(x-x.max())
  x /= np.sum(x)
  return x


def uniform_biased_strategy(solver, return_joint=False):
  """Returns a Biased Random Uniform distribution on policies.

  The uniform distribution is biased to prioritize playing against more recent
  policies (Policies that were appended to the policy list later in training)
  instead of older ones.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  if not isinstance(policies[0], list):
    policies = [policies]
  policy_lengths = [len(pol) for pol in policies]
  result = [softmax_on_range(pol_len) for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def renormalize(probabilities):
  """Replaces all negative entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities


def get_joint_strategy_from_marginals(probabilities):
  """Returns a joint strategy tensor from a list of marginals.

  Args:
    probabilities: list of probabilities.

  Returns:
    A flat joint strategy from a list of marginals.
  """
  res = np.ones((1,), dtype=np.float64)
  for prob in probabilities:
    res = res[..., None] @ np.asarray(prob).reshape((1,) * res.ndim + (-1,))
  return res.reshape(-1)


def nash_strategy(solver, return_joint=False):
  """Returns nash distribution on meta game matrix.

  This method only works for two player zero-sum games.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    Nash distribution on strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  meta_games = [x.tolist() for x in meta_games]
  if len(meta_games) != 2:
    raise NotImplementedError(
        "nash_strategy solver works only for 2p zero-sum"
        "games, but was invoked for a {} player game".format(len(meta_games)))
  nash_prob_1, nash_prob_2, _, _ = (
      lp_solver.solve_zero_sum_matrix_game(
          pyspiel.create_matrix_game(*meta_games)))
  result = [
      renormalize(np.array(nash_prob_1).reshape(-1)),
      renormalize(np.array(nash_prob_2).reshape(-1))
  ]

  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def prd_strategy(solver, return_joint=False):
  """Computes Projected Replicator Dynamics strategies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  result = projected_replicator_dynamics.projected_replicator_dynamics(
      meta_games, **kwargs)
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def rm_strategy(solver, return_joint=False):
  """Computes regret-matching strategies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  result = regret_matching.regret_matching(meta_games, **kwargs)
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


META_STRATEGY_METHODS = {
    "uniform_biased": uniform_biased_strategy,
    "uniform": uniform_strategy,
    "nash": nash_strategy,
    "prd": prd_strategy,
    "rm": rm_strategy,
}
