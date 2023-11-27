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
"""Kemeny-Young method.

Based on https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method.
"""

import itertools
import numpy as np
from open_spiel.python.voting import base


class KemenyYoungVoting(base.AbstractVotingMethod):
  """Implements Kemeny-Young's method."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "kemeny_young"

  def _score(self,
             alternatives: list[base.AlternativeId],
             pref_mat: np.ndarray,
             perm: tuple[int, ...]) -> tuple[list[base.AlternativeId], int,
                                             np.ndarray]:
    # The score of alternative a_i in a ranking R is defined to be:
    #      KemenyScore(a_i) = sum_{a_j s.t. R(a_i) >= R(a_j)} N(a_i, a_j)
    # The score of ranking R is then sum_i KemenyScore(a_i).
    num_alts = len(perm)
    scores = np.zeros(num_alts, dtype=np.int32)
    ranking = []
    for i in range(num_alts):
      alt_idx_i = perm[i]
      for j in range(i+1, num_alts):
        alt_idx_j = perm[j]
        value = pref_mat[alt_idx_i, alt_idx_j]
        scores[i] += value
      ranking.append(alternatives[alt_idx_i])
    return (ranking, scores.sum(), scores)

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    pref_mat = profile.pref_matrix()
    alternatives = profile.alternatives
    m = profile.num_alternatives()
    # ranking info is tuples of (ranking, total_score, scores list)
    best_ranking_info = (None, 0, [])
    for perm in itertools.permutations(range(m)):
      # perm is a permutation of alternative indices
      ranking_info = self._score(alternatives, pref_mat, perm)
      if ranking_info[1] > best_ranking_info[1]:
        best_ranking_info = ranking_info
    outcome = base.RankOutcome(rankings=best_ranking_info[0],
                               scores=list(best_ranking_info[2]))
    return outcome
