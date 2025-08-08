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

"""Helpers for pairwise data sets."""

import numpy as np
from open_spiel.python.voting import base


def balance_profile(
    profile: base.PreferenceProfile,
    num_votes_per_matchup: int = None,
) -> base.PreferenceProfile:
  """Balances an uneven preference profile.

  Args:
    profile: the preference profile to balance.
    num_votes_per_matchup: the number of votes to cast for each matchup. If
      None, then the function will take the maximum count of any matchup in the
      profile and use that.

  Returns:
    A new preference profile with balanced pairwise counts.

  An uneven profile is one where the distribution over number of matchups
  between (i, j) is non-uniform. This function creates a new profile
  (of pairwise preferences) where the number of votes per (i, j) matchup is
  uniform, while (in expectation) maintaining the true data's win rates per
  matchup.
  """
  m = profile.num_alternatives()
  new_profile = base.PreferenceProfile(alternatives=profile.alternatives)

  profile_counts = profile.pairwise_count_matrix()
  pref_matrix = profile.pref_matrix().astype(float)

  if num_votes_per_matchup is None:
    num_votes_per_matchup = profile.pairwise_count_matrix().max()

  print(f"num_votes_per_matchup: {num_votes_per_matchup}")

  # Add I to the counts to prevent the divide-by-zero on the diagonal.
  winrate_mat = pref_matrix.astype(float) / (profile_counts + np.eye(m))
  assert profile.alternatives == new_profile.alternatives

  for i in range(m):
    for j in range(i+1, m):
      alt_i = new_profile.alternatives[i]
      alt_j = new_profile.alternatives[j]
      num_wins = int(round(winrate_mat[i, j] * num_votes_per_matchup))
      num_losses = max(0, num_votes_per_matchup - num_wins)
      for _ in range(num_wins):
        new_profile.add_vote([alt_i, alt_j])
      for _ in range(num_losses):
        new_profile.add_vote([alt_j, alt_i])

  return new_profile

