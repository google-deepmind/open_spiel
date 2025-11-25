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

"""A helpful wrapper for the Elo rating system."""

import numpy as np
from open_spiel.python.algorithms import elo
import pyspiel
from open_spiel.python.voting import base


def compute_ratings_from_preference_profile(
    profile: base.PreferenceProfile,
    smoothing_factor: float = pyspiel.elo.DEFAULT_SMOOTHING_FACTOR,
    max_iterations: int = pyspiel.elo.DEFAULT_MAX_ITERATIONS,
    convergence_delta: float = pyspiel.elo.DEFAULT_CONVERGENCE_DELTA,
) -> dict[base.AlternativeId, float]:
  """Compute Elo ratings from a win matrix and a draw matrix."""
  options = pyspiel.elo.default_elo_options()
  options.smoothing_factor = smoothing_factor
  options.max_iterations = max_iterations
  options.convergence_delta = convergence_delta
  num_agents = profile.num_alternatives()
  alt_idx = profile.alternatives_dict
  win_matrix = np.zeros((num_agents, num_agents), dtype=int)
  for vote in profile.votes:
    for i in range(len(vote.vote)):
      for j in range(i + 1, len(vote.vote)):
        for _ in range(vote.weight):
          agent_i_idx = alt_idx[vote.vote[i]]
          agent_j_idx = alt_idx[vote.vote[j]]
          win_matrix[agent_i_idx, agent_j_idx] += 1
  ratings_array = elo.compute_ratings_from_matrices(
      win_matrix, smoothing_factor=smoothing_factor,
      max_iterations=max_iterations,
      convergence_delta=convergence_delta,
  )
  alternatives = profile.alternatives
  return {alternatives[i]: ratings_array[i] for i in range(num_agents)}


class EloVoting(base.AbstractVotingMethod):
  """Implements Elo as a voting method.

  Note: if there is no data on an alternatives, it is assigned a rating of 0.
  """

  def __init__(
      self,
      smoothing_factor: float = pyspiel.elo.DEFAULT_SMOOTHING_FACTOR,
      max_iterations: int = pyspiel.elo.DEFAULT_MAX_ITERATIONS,
      convergence_delta: float = pyspiel.elo.DEFAULT_CONVERGENCE_DELTA):
    self._smoothing_factor = smoothing_factor
    self._max_iterations = max_iterations
    self._convergence_delta = convergence_delta

  def name(self) -> str:
    return "elo"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    ratings_dict = compute_ratings_from_preference_profile(
        profile, self._smoothing_factor, self._max_iterations,
        self._convergence_delta,
    )
    sorted_ratings = sorted(ratings_dict.items(), key=lambda item: item[1],
                            reverse=True)
    if len(sorted_ratings) < profile.num_alternatives():
      # If missing alternatives, fill them in with zero scores.
      for alt in profile.alternatives:
        if alt not in ratings_dict:
          sorted_ratings.append((alt, 0.0))
    assert len(sorted_ratings) == profile.num_alternatives()
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_ratings)
    return outcome
