# Copyright 2023 DeepMind Technologies Limited
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
"""Copeland's method.

Based on https://en.wikipedia.org/wiki/Copeland%27s_method.
"""

from open_spiel.python.voting import base


class CopelandVoting(base.AbstractVotingMethod):
  """Implements Copeland's method."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "copeland"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    copeland_scores = {}
    alternatives = profile.alternatives
    m = len(alternatives)
    margin_matrix = profile.margin_matrix()
    for r in range(m):
      alternative = alternatives[r]
      num_majority = (margin_matrix[r] > 0).sum()
      # Subtract one because we don't include the diagonal.
      num_ties = (margin_matrix[r] == 0).sum() - 1
      copeland_scores[alternative] = num_majority + 0.5 * num_ties
    sorted_scores = sorted(copeland_scores.items(), key=lambda item: item[1],
                           reverse=True)
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_scores)
    return outcome
