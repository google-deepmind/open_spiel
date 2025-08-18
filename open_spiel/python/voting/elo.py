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

import pyspiel
from open_spiel.python.voting import base


def compute_ratings_from_preference_profile(
    profile: base.PreferenceProfile,
    smoothing_factor: float = pyspiel.elo.DEFAULT_SMOOTHING_FACTOR,
    max_iterations: int = pyspiel.elo.DEFAULT_MAX_ITERATIONS,
    convergence_delta: float = pyspiel.elo.DEFAULT_CONVERGENCE_DELTA,
) -> dict[str, float]:
  """Compute Elo ratings from a win matrix and a draw matrix."""
  options = pyspiel.elo.default_elo_options()
  options.smoothing_factor = smoothing_factor
  options.max_iterations = max_iterations
  options.convergence_delta = convergence_delta
  match_records = []
  for vote in profile.votes:
    for i in range(len(vote.vote)):
      for j in range(i + 1, len(vote.vote)):
        for _ in range(vote.weight):
          match_records.append(
              pyspiel.elo.MatchRecord(
                  str(vote.vote[i]),
                  str(vote.vote[j]),
                  pyspiel.elo.MatchOutcome.FIRST_PLAYER_WIN,
              )
          )
  return pyspiel.elo.compute_ratings_from_match_records(match_records, options)
