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

"""Mean-Field Correlated Equilibrium Gap & Best Response Computation Library.

"""

import numpy as np
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import joint_best_response_value as jbr
from open_spiel.python.mfg.algorithms import utils


def get_joint_br(game, weights, mus):
  br_value = jbr.JointBestResponse(game, mus, weights)
  greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
  return greedy_pi, br_value


def compute_rewards(game, policies, mus):
  return np.array([
      [utils.get_exact_value(pi, mu, game) for pi in policies] for mu in mus
  ])


def compute_average_welfare(game, policies, mus, rhos, nus):
  """Computes average welfare.

  Args:
    game: Pyspiel game.
    policies: List of policies, length P
    mus: List of State Distributions of length T
    rhos: Temporal weights, length T
    nus: Policy distribution per time, shape [T, P]

  Returns:
    Average welfare.
  """
  assert len(mus) == len(rhos)
  assert len(rhos) == nus.shape[0]
  assert len(policies) == nus.shape[1]

  rewards = compute_rewards(game, policies, mus)
  return np.sum(rewards * nus * rhos.reshape(-1, 1))


def cce_br(game, policies, weights, mus, nus, rewards=None):
  """Computes CCE-BR.

  Args:
    game: Pyspiel MFG Game.
    policies: List of pyspiel policies, length P.
    weights: Array of temporal weights on each distribution in `nu`, length T.
    mus: List of state distributions, length T.
    nus: Array of policy distribution per timestep, shape (T, P)
    rewards: Optional array of policy reward per timestep, shape (T, P)

  Returns:
    Best-response, computed exploitability from `rewards`.
  """
  assert len(mus) == len(nus)
  assert len(mus) == len(weights)

  del policies
  pol, val = get_joint_br(game, weights, mus)
  cce_gap_value = None
  if len(rewards) > 0:  # pylint: disable=g-explicit-length-test
    deviation_value = val.value(game.new_initial_states()[0])
    on_policy_value = np.sum(weights * np.sum(rewards * nus, axis=1))
    cce_gap_value = deviation_value - on_policy_value
  return [pol], cce_gap_value


def ce_br(game, policies, weights, mus, nus, rewards=None):
  """Computes CE-BR.

  Args:
    game: Pyspiel MFG Game.
    policies: List of pyspiel policies, length P.
    weights: Array of temporal weights on each distribution in `nu`, length T.
    mus: List of state distributions, length T.
    nus: Array of policy distribution per timestep, shape (T, P)
    rewards: Optional array of policy reward per timestep, shape (T, P)

  Returns:
    Best-responses, computed exploitability from `rewards`.
  """
  assert len(mus) == len(nus)
  assert len(mus) == len(weights)

  policy_probability = np.sum(nus, axis=0)
  new_policies = []
  ce_gap_value = 0.0
  nus = np.array(nus)
  weights = np.array(weights)
  for policy_index in range(len(policies)):
    if policy_probability[policy_index] > 0:
      # Take conditional distribution
      pol_weights = nus[:, policy_index] * weights
      pol_proba = np.sum(pol_weights)
      pol_weights = pol_weights / pol_proba

      # Prune state distribution and weights from 0.0-weightred values
      new_mus = [mu for ind, mu in enumerate(mus) if pol_weights[ind] > 0]
      new_weights = np.array([
          weight for ind, weight in enumerate(pol_weights)
          if pol_weights[ind] > 0
      ])

      # Compute best-response.
      new_pol, new_val = get_joint_br(game, new_weights, new_mus)
      new_br_val = new_val.value(game.new_initial_states()[0])

      # Evaluate CE-Gap
      if len(rewards) > 0:  # pylint: disable=g-explicit-length-test
        on_policy_value = np.sum(
            np.array(rewards)[:, policy_index] * pol_weights)
        ce_gap_value += pol_proba * (new_br_val - on_policy_value)
      new_policies.append(new_pol)
  return new_policies, ce_gap_value


def partial_ce_br(game, policies, weights, mus, nus, rewards=None):
  """Computes CE-BR for a single sampled policy.

  Args:
    game: Pyspiel MFG Game.
    policies: List of pyspiel policies, length P.
    weights: Array of temporal weights on each distribution in `nu`, length T.
    mus: List of state distributions, length T.
    nus: Array of policy distribution per timestep, shape (T, P)
    rewards: Optional array of policy reward per timestep, shape (T, P)

  Returns:
    Best-response, noisy exploitability estimation.
  """
  policy_probability = np.sum(nus, axis=0)
  new_policies = []

  ce_gap_value = None
  policy_index = np.random.choice(list(range(len(policies))))
  if policy_probability[policy_index] > 0:
    # Take conditional distribution
    pol_weights = [nu[policy_index] * weight for nu, weight in zip(
        nus, weights)]
    pol_proba = np.sum(pol_weights)
    pol_weights = np.array(pol_weights) / pol_proba

    # Prune state distribution and weights from 0.0-weightred values
    new_mus = [mu for ind, mu in enumerate(mus) if pol_weights[ind] > 0]
    new_weights = [
        weight for ind, weight in enumerate(pol_weights)
        if pol_weights[ind] > 0
    ]

    # Compute best-response.
    new_pol, new_val = get_joint_br(game, new_weights, new_mus)
    new_br_val = new_val.value(game.new_initial_states()[0])

    # Evaluate CE-Gap
    if len(rewards) > 0:  # pylint: disable=g-explicit-length-test
      on_policy_value = np.sum(np.array(rewards)[:, policy_index] * pol_weights)
      ce_gap_value = (new_br_val - on_policy_value)
    new_policies.append(new_pol)
  return new_policies, ce_gap_value


def cce_gap(game, policies, weights, mus, nus, rewards=None,
            compute_true_rewards=False):
  if compute_true_rewards:
    rewards = compute_rewards(game, policies, mus)
  assert rewards is not None, ("Must provide rewards matrix when computing CCE "
                               "Gap.")
  _, gap = cce_br(game, policies, weights, mus, nus, rewards=rewards)
  return gap


def ce_gap(game, policies, weights, mus, nus, rewards=None,
           compute_true_rewards=False):
  if compute_true_rewards:
    rewards = compute_rewards(game, policies, mus)
  assert rewards is not None, ("Must provide rewards matrix when computing CE "
                               "Gap.")
  _, gap = ce_br(game, policies, weights, mus, nus, rewards=rewards)
  return gap
