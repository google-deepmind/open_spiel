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
"""Fishburn's Maximal lotteries method.

Based on https://en.wikipedia.org/wiki/Maximal_lotteries.
"""

from typing import List
import numpy as np
from open_spiel.python.algorithms import lp_solver
import pyspiel
from open_spiel.python.voting import base


class MaximalLotteriesVoting(base.AbstractVotingMethod):
  """Implements Copeland's method."""

  def __init__(self,
               iterative: bool = False,
               verbose: bool = False,
               zero_tolerance: float = 1e-6):
    self._iterative = iterative
    self._verbose = verbose
    self._zero_tolerance = zero_tolerance

  def name(self) -> str:
    return f"maximal_lotteries(iterative={self._iterative})"

  def _create_matrix_game(self, matrix: np.ndarray):
    return pyspiel.create_tensor_game([matrix, -matrix]).as_matrix_game()

  def _solve_game(
      self, margin_matrix: np.ndarray
    ) -> np.ndarray:
    matrix_game = self._create_matrix_game(margin_matrix)
    p0_sol, _, _, _ = lp_solver.solve_zero_sum_matrix_game(matrix_game)
    return p0_sol

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    margin_matrix = profile.margin_matrix()
    alternatives = profile.alternatives
    m = profile.num_alternatives()
    if self._verbose:
      print(f"Margin matrix: \n{margin_matrix}")
      print(f"Alternatives: {alternatives}")
    p0_sol = self._solve_game(margin_matrix)

    # For now define scores as the probabilities.
    scores = {}
    if not self._iterative:
      # and negligible noise to break ties
      noise = 1e-10 * np.random.uniform(size=m)
      for i in range(m):
        scores[alternatives[i]] = p0_sol[i] + noise[i]
      sorted_scores = sorted(scores.items(), key=lambda item: item[1])
      sorted_scores.reverse()
      outcome = base.RankOutcome()
      outcome.unpack_from(sorted_scores)
      return outcome
    else:
      # Continue to iteratively solve all the remaining subgames.
      return self._iterate(alternatives, margin_matrix, p0_sol)

  def _iterate(
      self,
      alternatives: List[base.AlternativeId],
      margin_matrix: np.ndarray,
      p0_sol: np.ndarray,
  ):
    remaining_alternatives = alternatives[:]
    leveled_ranking = []
    leveled_scores = []
    while remaining_alternatives:
      # Pull out the nonzero entries and make them winners of this level.
      m = len(remaining_alternatives)
      if self._verbose:
        print(f"\nRemaining alternatives: {remaining_alternatives}")
        cur_level = len(leveled_ranking)
        print(f"IML Level {cur_level}")
        print(f"Remaining alternatives: {remaining_alternatives}")
        print(f"Margin matrix: \n{margin_matrix}\n")
      if m == 1:
        leveled_ranking.append(remaining_alternatives[:])
        leveled_scores.append([1])
        break
      noise = 1e-10 * np.random.uniform(size=m)
      for i in range(m):
        p0_sol[i] += noise[i]
      values = -1 * np.ones(m, dtype=np.float64)
      level_winners_idxs = []
      for i in range(m):
        if p0_sol[i] > self._zero_tolerance:
          # print(f"p0_sol[{i}] = {p0_sol[i]}")
          level_winners_idxs.append(i)
          values[i] = p0_sol[i]
      num_level_winners = len(level_winners_idxs)
      assert num_level_winners >= 1
      indices = np.argsort(-values)
      level_winners_ranked = []
      level_winners_scores = []
      for j in range(num_level_winners):
        idx = int(indices[j])
        level_winners_ranked.append(remaining_alternatives[idx])
        level_winners_scores.append(p0_sol[idx])
      leveled_ranking.append(level_winners_ranked)
      leveled_scores.append(level_winners_scores)
      if self._verbose:
        print(f"Level winners: {level_winners_ranked}")
        print(f"Level scores: {level_winners_scores}")
      # Now, take them out of the margin matrix and remaining alternatives
      # Delete in reverse order.
      for j in range(num_level_winners):
        idx = level_winners_idxs[num_level_winners - 1 - j]
        del remaining_alternatives[idx]
        margin_matrix = np.delete(margin_matrix, (idx), axis=0)
        margin_matrix = np.delete(margin_matrix, (idx), axis=1)
      if len(remaining_alternatives) > 1:
        p0_sol = self._solve_game(margin_matrix)
    # Now bump up the scores by level, and put them in the outcome.
    scores = {}
    num_levels = len(leveled_ranking)
    if self._verbose:
      print(f"Num levels: {num_levels}")
    level_base_points = num_levels - 1
    for level in range(num_levels):
      for j in range(len(leveled_ranking[level])):
        alternative = leveled_ranking[level][j]
        score = level_base_points + leveled_scores[level][j]
        scores[alternative] = score
      level_base_points -= 1
    sorted_scores = sorted(scores.items(), key=lambda item: item[1])
    sorted_scores.reverse()
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_scores)
    return outcome
