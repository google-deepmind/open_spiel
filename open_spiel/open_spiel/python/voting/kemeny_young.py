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
from typing import List, Tuple
import numpy as np
from open_spiel.python.voting import base


class KemenyYoungVoting(base.AbstractVotingMethod):
  """Implements Kemeny-Young's method."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "kemeny_young"

  def _score(
      self,
      pref_mat: np.ndarray,
      perm: Tuple[int, ...],
  ) -> np.ndarray:
    # The score of alternative a_i in a ranking R is defined to be:
    #      KemenyScore(a_i) = sum_{a_j s.t. R(a_i) >= R(a_j)} N(a_i, a_j)
    # The score of ranking R is then sum_i KemenyScore(a_i).
    num_alts = len(perm)
    scores = np.zeros(num_alts, dtype=np.int32)
    for i in range(num_alts):
      for j in range(i+1, num_alts):
        scores[i] += pref_mat[perm[i], perm[j]]
    return scores

  def _permutation_to_ranking(
      self,
      alternatives: List[base.AlternativeId],
      permutation: Tuple[base.AlternativeId, ...]) -> List[base.AlternativeId]:
    assert len(permutation) == len(alternatives)
    return [alternatives[permutation[i]] for i in range(len(alternatives))]

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    pref_mat = profile.pref_matrix()
    alternatives = profile.alternatives
    m = profile.num_alternatives()
    best_permutation = None
    best_score = -1
    best_score_array = None
    for permutation in itertools.permutations(range(m)):
      scores = self._score(pref_mat, permutation)
      total_score = scores.sum()
      if total_score > best_score:
        best_score = total_score
        best_score_array = scores
        best_permutation = permutation
    best_ranking = self._permutation_to_ranking(alternatives, best_permutation)
    outcome = base.RankOutcome(rankings=best_ranking,
                               scores=list(best_score_array))
    return outcome

