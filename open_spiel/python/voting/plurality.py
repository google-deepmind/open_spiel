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

"""Plurality voting method.

Based on https://en.wikipedia.org/wiki/Plurality_voting.
"""

from open_spiel.python.voting import base


class PluralityVoting(base.AbstractVotingMethod):
  """Implements the plurality (first past the post) voting rule."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "plurality"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    tally = {}
    for alternative in profile.alternatives:
      tally[alternative] = 0
    for vote in profile.votes:
      tally[vote.vote[0]] += vote.weight
    sorted_tally = sorted(tally.items(), key=lambda item: item[1], reverse=True)
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_tally)
    return outcome
