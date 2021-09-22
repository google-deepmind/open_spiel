# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Fictitious Play from Perrin & al.

Refference : https://arxiv.org/abs/2007.03458.
"""
from typing import List

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy


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
      weights: A `List[float]` object. They should sum to 1.
    """
    super(MergedPolicy, self).__init__(game, player_ids)
    self._policies = policies
    self._distributions = distributions
    self._weights = weights
    assert len(policies) == len(distributions), (
        f'Length mismatch {len(policies)} != {len(distributions)}')
    assert len(policies) == len(weights), (
        f'Length mismatch {len(policies)} != {len(weights)}')

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

  def iteration(self):
    """Returns a new `TabularPolicy` equivalent to this policy."""
    self._fp_step += 1

    distrib = distribution.DistributionPolicy(self._game, self._policy)
    br_value = best_response_value.BestResponse(
        self._game, distrib, value.TabularValueFunction(self._game))

    greedy_pi = greedy_policy.GreedyPolicy(self._game, None, br_value)
    greedy_pi = greedy_pi.to_tabular(states=self._states)
    distrib_greedy = distribution.DistributionPolicy(self._game, greedy_pi)

    self._policy = MergedPolicy(
        self._game, list(range(self._game.num_players())),
        [self._policy, greedy_pi], [distrib, distrib_greedy],
        [1.0 * self._fp_step / (self._fp_step + 1), 1.0 /
         (self._fp_step + 1)]).to_tabular(states=self._states)
