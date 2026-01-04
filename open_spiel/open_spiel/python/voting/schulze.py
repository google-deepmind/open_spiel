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
"""Schulze method.

Based on https://en.wikipedia.org/wiki/Schulze_method.
"""

import functools
import numpy as np
from open_spiel.python.voting import base


class SchulzeVoting(base.AbstractVotingMethod):
  """Implements Schulze's method."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "schulze"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    alternatives = profile.alternatives
    num_alternatives = profile.num_alternatives()
    pref_mat = profile.pref_matrix()
    strongest_paths = np.zeros(shape=(num_alternatives, num_alternatives),
                               dtype=np.float32)
    # calculating the direct paths
    for i in range(num_alternatives):
      for j in range(num_alternatives):
        if i != j:
          if pref_mat[i, j] > pref_mat[j, i]:
            strongest_paths[i, j] = pref_mat[i, j]
          else:
            strongest_paths[i, j] = 0
    # checking if any indirect paths are better
    for i in range(num_alternatives):
      for j in range(num_alternatives):
        if i != j and strongest_paths[j, i] > 0:
          for k in range(num_alternatives):
            if i != k and j != k:
              # if the path from j to k through i is better, replace
              strongest_paths[j, k] = max(strongest_paths[j, k],
                                          min(strongest_paths[j, i],
                                              strongest_paths[i, k]))

    def compare(x, y):
      return strongest_paths[x, y] - strongest_paths[y, x]
    ranking_idx = np.arange(num_alternatives)
    sorted_ranking_idx = sorted(ranking_idx, key=functools.cmp_to_key(compare),
                                reverse=True)
    # Define the scores as the sum of preferences for everything it beats in
    # the order.
    cumul_score = 0
    # start at the end and work backwards
    ranking_alts = [alternatives[sorted_ranking_idx[-1]]]
    scores = [0]
    i = num_alternatives - 2
    while i >= 0:
      alt_idx_i = sorted_ranking_idx[i]
      alt_idx_j = sorted_ranking_idx[i+1]
      ranking_alts.insert(0, alternatives[alt_idx_i])
      cumul_score += pref_mat[alt_idx_i, alt_idx_j]
      scores.insert(0, cumul_score)
      i -= 1
    return base.RankOutcome(rankings=ranking_alts, scores=scores)
