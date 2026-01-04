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

"""Implements Borda's method.

Based on: https://en.wikipedia.org/wiki/Borda_count.
"""

from open_spiel.python.voting import base


class BordaVoting(base.AbstractVotingMethod):
  """Implements Borda's method of voting."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "borda"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    scores = {}
    for alternative in profile.alternatives:
      scores[alternative] = 0
    for vote in profile.votes:
      # Do we need a check here for the length of the vote?
      points = len(vote.vote) - 1
      for alternative in vote.vote:
        scores[alternative] += (points * vote.weight)
        points -= 1
      assert points == -1
    sorted_scores = sorted(scores.items(), key=lambda item: item[1],
                           reverse=True)
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_scores)
    return outcome
