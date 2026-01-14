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

"""Optimization algorithms to compute (C)CE weights."""

import numpy as np
import scipy.optimize
import scipy.sparse.linalg


# pylint: disable=invalid-name
def get_proba_constraints_positivity(nus):
  A = np.zeros((nus.shape[0], 1 + nus.shape[0]))
  A[:, 1:] = -np.eye(nus.shape[0])
  return A, np.zeros(A.shape[0])


def get_proba_constraint_sum_eq(nus):
  A = np.ones((1, 1 + nus.shape[0]))
  A[0, 0] = 0.0
  return A, np.array([1.0])


def compress_internal_weights(nus, regrets, rewards, lbd=0.0):
  """Computes distribution over `nus` while minimizing internal regret.

  Args:
    nus: [T, P] array, T the number of different population distributions, P the
      number of different policies.
    regrets: [T, P, P] array, regrets[t, i, j] = payoff for switching from
      policy i to j at time t.
    rewards: [T, P] array, T the number of different population distributions, P
      the number of different policies
    lbd: Sparsity argument.

  Returns:
    Computed distribution over `nus`.
  """

  def get_c(nus):
    return np.concatenate(
        (np.array([1.0]), -lbd * np.sum(rewards * nus, axis=1))
    )

  def get_max_constraint(regrets):
    regrets = np.transpose(np.array(regrets), axes=[0, 2, 1])
    regrets = regrets.reshape(-1, regrets.shape[-1])
    A = np.zeros((regrets.shape[0], 1 + regrets.shape[1]))
    A[:, 1:] = regrets
    A[:, 0] = -1.0

    b = np.zeros(A.shape[0])
    return A, b

  def get_a_ub(nus, regrets):
    Amax, bmax = get_max_constraint(regrets)
    Apos, bpos = get_proba_constraints_positivity(nus)
    return np.concatenate((Amax, Apos), axis=0), np.concatenate(
        (bmax, bpos), axis=0
    )

  c = get_c(nus)

  A_ub, b_ub = get_a_ub(nus, regrets)
  A_eq, b_eq = get_proba_constraint_sum_eq(nus)

  res = scipy.optimize.linprog(
      c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options={'tol': 1e-10}
  )
  new_weights = res.x
  return new_weights[1:]


def compress_external_weights(nus, regrets, rewards, lbd=0.0):
  """Computes distribution over `nus` while minimizing external regret.

  Args:
    nus: [T, P] array, T the number of different population distributions, P the
      number of different policies.
    regrets: [T, P] array, regrets[t, i] = payoff for switching from current
      policy to i at time t.
    rewards: [T, P] array, reward for playing policy P at time T.
    lbd: Sparsity argument.

  Returns:
    Computed distribution over `nus`.
  """

  def get_c(nus):
    return np.concatenate(
        (np.array([1.0]), -lbd * np.sum(rewards * nus, axis=1))
    )

  def get_max_constraints(nus, regrets, lbd):
    A = np.zeros((regrets.shape[1], 1 + nus.shape[0]))
    A[:, 0] = -1.0
    A[:, 1:] = np.transpose(
        regrets
        - np.sum(regrets * nus, axis=1).reshape(-1, 1)
        - lbd * np.abs(regrets)
    )
    return A, np.zeros(A.shape[0])

  def get_a_ub(nus, regrets, lbd):
    Amax, bmax = get_max_constraints(nus, regrets, lbd)
    Apos, bpos = get_proba_constraints_positivity(nus)
    return np.concatenate((Amax, Apos), axis=0), np.concatenate(
        (bmax, bpos), axis=0
    )

  c = get_c(nus)

  A_ub, b_ub = get_a_ub(nus, regrets, lbd)
  A_eq, b_eq = get_proba_constraint_sum_eq(nus)

  res = scipy.optimize.linprog(
      c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options={'tol': 1e-10}
  )
  new_weights = res.x
  return new_weights[1:]
