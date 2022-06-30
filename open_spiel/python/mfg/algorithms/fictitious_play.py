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
"""Implementation of Fictitious Play from Perrin & al.

Reference: https://arxiv.org/abs/2007.03458.
As presented, the Fictitious Play algorithm provides a robust approximation
scheme for Nash equilibrium by iteratively computing the best response
against the distribution induced by the average of the past best responses.
The provided formulation of Deep Fictitious Play mirrors this procedure,
but substitutes out the exact best reponse computation with an approximation
of best response values through a Reinforcement Learning approach (where
the RL method in question is a user-determined parameter for each iteration).

Policy is initialized to uniform policy.
Each iteration:
 1. Compute best response against policy
 2. Update policy as weighted average of best response and current policy
    (default learning rate is 1 / num_iterations + 1).

To use fictitious play one should initialize it and run multiple iterations:
fp = FictitiousPlay(game)
for _ in range(num_iterations):
  fp.iteration()
policy = fp.get_policy()
"""

import math

from typing import List

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import policy_value


class MergedPolicy(policy_std.Policy):
  """Merge several policies."""

  def __init__(self, game, player_ids, policies: List[policy_std.Policy],
               distributions: List[distribution_std.Distribution],
               weights: List[float]):
    """Initializes the merged policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
      policies: A `List[policy_std.Policy]` object.
      distributions: A `List[distribution_std.Distribution]` object.
      weights: A `List[float]` object. The elements should sum to 1.
    """
    super().__init__(game, player_ids)
    self._policies = policies
    self._distributions = distributions
    self._weights = weights
    assert len(policies) == len(distributions), (
        f'Length mismatch {len(policies)} != {len(distributions)}')
    assert len(policies) == len(weights), (
        f'Length mismatch {len(policies)} != {len(weights)}')
    assert math.isclose(
        sum(weights),
        1.0), (f'Weights should sum to 1, but instead sum to {sum(weights)}')

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


class FictitiousPlay(object):
  """Computes the value of a specified strategy."""

  def __init__(self, game):
    """Initializes the greedy policy.

    Args:
      game: The game to analyze.
    """
    self._game = game
    self._states = None  # Required to avoid attribute-error.
    self._policy = policy_std.UniformRandomPolicy(self._game)
    self._fp_step = 0
    self._states = policy_std.get_tabular_policy_states(self._game)

  def get_policy(self):
    return self._policy

  def iteration(self, br_policy=None, learning_rate=None):
    """Returns a new `TabularPolicy` equivalent to this policy.

    Args:
      br_policy: Policy to compute the best response value for each iteration.
        If none provided, the exact value is computed.
      learning_rate: The learning rate.
    """
    self._fp_step += 1

    distrib = distribution.DistributionPolicy(self._game, self._policy)

    if br_policy:
      br_value = policy_value.PolicyValue(self._game, distrib, br_policy)
    else:
      br_value = best_response_value.BestResponse(
          self._game, distrib, value.TabularValueFunction(self._game))

    greedy_pi = greedy_policy.GreedyPolicy(self._game, None, br_value)
    greedy_pi = greedy_pi.to_tabular(states=self._states)
    distrib_greedy = distribution.DistributionPolicy(self._game, greedy_pi)

    weight = learning_rate if learning_rate else 1.0 / (self._fp_step + 1)

    self._policy = MergedPolicy(
        self._game, list(range(self._game.num_players())),
        [self._policy, greedy_pi], [distrib, distrib_greedy],
        [1.0 - weight, weight]).to_tabular(states=self._states)
