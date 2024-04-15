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

"""Collection of useful functions and classes."""

from typing import List, Optional

import numpy as np

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value
import pyspiel


class MergedPolicy(policy_std.Policy):
  """Merge several policies."""

  def __init__(
      self,
      game,
      player_ids,
      policies: List[policy_std.Policy],
      weights: List[float],
      distributions: Optional[List[distribution_std.Distribution]] = None,
  ):
    """Initializes the merged policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
      policies: A `List[policy_std.Policy]` object.
      weights: A `List[float]` object. They should sum to 1.
      distributions: A `List[distribution_std.Distribution]` object.
    """
    super(MergedPolicy, self).__init__(game, player_ids)
    self._policies = policies
    self._distributions = distributions
    self._weights = weights
    if distributions is None:
      distributions = [
          distribution.DistributionPolicy(game, policy) for policy in policies
      ]
    else:
      assert len(policies) == len(
          distributions
      ), f'Length mismatch {len(policies)} != {len(distributions)}'
    assert len(policies) == len(
        weights
    ), f'Length mismatch {len(policies)} != {len(weights)}'

  def action_probabilities(self, state, player_id=None):
    action_prob = []
    legal = state.legal_actions()
    num_legal = len(legal)
    for a in legal:
      merged_pi = 0.0
      norm_merged_pi = 0.0
      for p, d, w in zip(self._policies, self._distributions, self._weights):
        merged_pi += w * d(state) * p(state)[a]
        norm_merged_pi += w * d(state)
      if norm_merged_pi > 0.0:
        action_prob.append((a, merged_pi / norm_merged_pi))
      else:
        action_prob.append((a, 1.0 / num_legal))
    return dict(action_prob)


class MixedDistribution:
  """Mixes a list of distributions wrt. a list of weights.

  The mixed distribution remains a probability distribution over states.

  Attributes:
    mus: The state distributions being mixed.
    weights: The list of weights of each `mus` member.
    _mus: The state distributions being mixed, post-pruning.
    _weights: The list of weights of each `mus` member, post-pruning.
    _tol: Tolerance (`mus` members with weights below tolerance are ignored)
    _value_str_cache: Cache for value_str calls.
  """

  def __init__(self, mus, weights, tol=1e-4):
    """Mixes the distribution.

    Args:
      mus: List of distributions to mix.
      weights: List of weights to mix `mus` over.
      tol: Tolerance (`mus` members with weights below tolerance are ignored)
    """
    self.mus = mus
    self.weights = weights
    self._tol = tol
    self._prune()
    self._value_str_cache = {}

  def _prune(self):
    self._mus = [mu for w, mu in zip(self.weights, self.mus) if w > self._tol]
    self._weights = [w for w in self.weights if w > self._tol]
    self._weights = [w / sum(self._weights) for w in self._weights]

  def value(self, state):
    """Returns the probability of the distribution on the state.

    Args:
      state: A `pyspiel.State` object.

    Returns:
      A `float`.
    """
    return sum([
        weight * mu.value(state) for weight, mu in zip(self._weights, self._mus)
    ])

  def value_str(self, state_str, default_value=None):
    """Returns the probability of the distribution on the given state string.

    Args:
      state_str: A string.
      default_value: If not None, return this value if the state is not in the
        support of the distribution.

    Returns:
      A `float`.
    """
    if state_str not in self._value_str_cache:
      self._value_str_cache[state_str] = sum([
          weight * mu.value_str(state_str, default_value)
          for weight, mu in zip(self._weights, self._mus)
      ])
    return self._value_str_cache[state_str]

  def __call__(self, state):
    """Turns the distribution into a callable.

    Args:
      state: The current state of the game.

    Returns:
      Float: probability.
    """
    return self.value(state)


def get_exact_value(
    pi: policy_std.Policy, mu: distribution_std.Distribution, game
):
  """Computes the exact value of playing `pi` against distribution `mu`.

  Args:
    pi: A policy object whose value is evaluated against `mu`.
    mu: A distribution object against which `pi` is evaluated.
    game: A pyspiel.Game object, the evaluation game.

  Returns:
    Exact value of `pi` in `game` against `mu`.
  """
  root_state = game.new_initial_states()[0]
  return policy_value.PolicyValue(game, mu, pi).value(root_state)


def sample_value(
    pi: policy_std.Policy, mu: distribution_std.Distribution, game
):
  """Samples the value of playing `pi` against distribution `mu`.

  Args:
    pi: A policy object whose value is evaluated against `mu`.
    mu: A distribution object against which `pi` is evaluated.
    game: A pyspiel.Game object, the evaluation game.

  Returns:
    Sampled value of `pi` in `game` against `mu`.
  """
  mfg_state = game.new_initial_states()[0]
  total_reward = 0.0
  while not mfg_state.is_terminal():
    if mfg_state.current_player() == pyspiel.PlayerId.CHANCE:
      action_list, prob_list = zip(*mfg_state.chance_outcomes())
      action = np.random.choice(action_list, p=prob_list)
      mfg_state.apply_action(action)
    elif mfg_state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = mfg_state.distribution_support()
      dist = [mu.value_str(str_state, 0.0) for str_state in dist_to_register]
      mfg_state.update_distribution(dist)
    else:
      total_reward += mfg_state.rewards()[0]
      action_prob = pi(mfg_state)
      action = np.random.choice(
          list(action_prob.keys()), p=list(action_prob.values())
      )
      mfg_state.apply_action(action)

  return total_reward


def get_nu_values(policies, nu, game):
  rewards = np.zeros(len(policies))
  mu = distribution.DistributionPolicy(
      game, MergedPolicy(game, None, policies, nu)
  )
  for index, policy in enumerate(policies):
    rewards[index] = sample_value(policy, mu, game)
  return rewards
