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

"""Implements approval voting method.

Based on: https://en.wikipedia.org/wiki/Approval_voting.
"""

from open_spiel.python.voting import base


# This seems arbitrary.. is there something sensible we should default to?
DEFAULT_K = 3


class ApprovalVoting(base.AbstractVotingMethod):
  """Implements approval voting."""

  def __init__(self, k: int = 1):
    """Construct an k-Approval voting scheme.

    Note: there are no checks on the length of the votes and how they relate to 
    the value of k. So, the user is responsible for appropriately balancing the
    lengths of the votes appropriately.

    Arguments:
      k: the number of top positions to count in each vote.
    """
    self._k = k

  def name(self) -> str:
    return f"approval(k={self._k})"

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    scores = {alternative: 0 for alternative in profile.alternatives}
    for vote in profile.votes:
      vote_len = len(vote.vote)
      for i in range(self._k):
        if i >= vote_len: break
        alternative = vote.vote[i]
        scores[alternative] += vote.weight
    sorted_scores = sorted(scores.items(), key=lambda item: item[1],
                           reverse=True)
    outcome = base.RankOutcome()
    outcome.unpack_from(sorted_scores)
    return outcome
