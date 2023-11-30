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
"""Single Transferrable Vote (STV) method.

Based on https://en.wikipedia.org/wiki/Single_transferable_vote.
"""

from typing import Dict, List, Union
from open_spiel.python.voting import base


class MutableVote(object):
  """A mutable vote annotated with the current preferred alternative.
 
  This is used to keep track of votes and which index (into the preference list)
  is currently active, i.e. the most preferred. When votes get used to determine
  winners or elimintations, some of these votes get "transfered" down to the
  next alternative. To transfer the vote, the index here is incremented to
  indicate that this vote is now representing a vote for the next highest
  alternative.
  """

  def __init__(self, idx: int, weight: int, vote: List[base.AlternativeId]):
    self.idx = idx
    self.weight = weight
    self.vote = vote


class STVVoting(base.AbstractVotingMethod):
  """Implements STV method."""

  def __init__(
      self, num_winners: Union[int, None] = None, verbose: bool = False
  ):
    """Construct an instance of STV with the specified number of winners.

    Args:
      num_winners: number of winners. Should be less than number of
          alternatives (m). If not specified, defaults to int(m/2).
      verbose: whether or not to print debug information as STV is running.
    """
    self._num_winners = num_winners
    self._verbose = verbose

  def name(self) -> str:
    return f"single_transferable_vote(num_winners={self._num_winners})"

  def _is_still_active(
      self,
      alternative: base.AlternativeId,
      winners: List[base.AlternativeId],
      losers: List[base.AlternativeId],
  ) -> bool:
    """Returns whether the alternative is still in the running."""
    return alternative not in winners and alternative not in losers

  def _next_idx_in_the_running(
      self,
      mutable_vote: MutableVote,
      winners: List[base.AlternativeId],
      losers: List[base.AlternativeId],
  ) -> int:
    """"Returns the next index in the list that is still in the running."""
    new_idx = mutable_vote.idx + 1
    while (new_idx < len(mutable_vote.vote) and
           not self._is_still_active(mutable_vote.vote[new_idx], winners,
                                     losers)):
      new_idx += 1
    return new_idx

  def _initial_scores_for_round(
      self,
      profile: base.PreferenceProfile,
      winners: List[base.AlternativeId],
      losers: List[base.AlternativeId],
  ) -> Dict[base.AlternativeId, float]:
    """Returns round's initial scores for alternatives still in the running."""
    alt_scores = {}
    for alt in profile.alternatives:
      if self._is_still_active(alt, winners, losers):
        alt_scores[alt] = 0
    return alt_scores

  def _remove_winning_votes(
      self,
      winning_alt: base.AlternativeId,
      num_to_remove: int,
      all_votes: List[MutableVote],
  ):
    while num_to_remove > 0:
      for mutable_vote in all_votes:
        if (mutable_vote.idx < len(mutable_vote.vote) and
            mutable_vote.vote[mutable_vote.idx] == winning_alt):
          removing_now = min(mutable_vote.weight, num_to_remove)
          mutable_vote.weight -= removing_now
          num_to_remove -= removing_now
        if num_to_remove == 0:
          break

  def run_election(self, profile: base.PreferenceProfile) -> base.RankOutcome:
    assert self.is_valid_profile(profile)
    winners = []
    losers = []
    winner_scores = []
    loser_scores = []
    votes = profile.votes
    total_votes = profile.total_weight()
    m = profile.num_alternatives()
    num_winners = self._num_winners
    if num_winners is None:
      num_winners = int(m/2)
      if self._verbose:
        print("Warning: number of winners not specified." +
              f"Choosing {num_winners}")
    assert num_winners < m
    quota = int(total_votes / float(num_winners + 1) + 1)
    # Table holds a list of the IndexAndWeightedVote. The index corresponds to
    # the current alternative that this vote is representing. They all start at
    # 0 at the start, corresponding to their highest preference, and they get
    # incremented as they become used up.
    all_votes: List[MutableVote] = []
    for vote in votes:
      all_votes.append(MutableVote(idx=0, weight=vote.weight, vote=vote.vote))
    while len(winners) + len(losers) < m:
      scores = self._initial_scores_for_round(profile, winners, losers)
      for mutable_vote in all_votes:
        if (mutable_vote.idx < len(mutable_vote.vote) and
            mutable_vote.weight > 0):
          alt = mutable_vote.vote[mutable_vote.idx]
          scores[alt] += mutable_vote.weight
      sorted_scores = sorted(scores.items(), key=lambda item: item[1],
                             reverse=True)
      best_score = sorted_scores[0][1]
      if best_score >= quota:
        # Quota reached. A candidate wins!
        if self._verbose:
          print(f"Quota {quota} reached. Candidate {sorted_scores[0][0]} wins!")
        winning_alt = sorted_scores[0][0]
        winners.append(winning_alt)
        winner_scores.append(best_score)
        surplus = sorted_scores[0][1] - quota
        # Remove votes that contributed to the winner, up to the quota.
        self._remove_winning_votes(winning_alt, quota, all_votes)
        # Then, convert all the rest.
        num_converted = 0
        for mutable_vote in all_votes:
          if (mutable_vote.idx < len(mutable_vote.vote) and
              mutable_vote.vote[mutable_vote.idx] == winning_alt and
              mutable_vote.weight > 0):
            # find the next one in the list still in the running.
            new_idx = self._next_idx_in_the_running(mutable_vote, winners,
                                                    losers)
            mutable_vote.idx = new_idx
            num_converted += mutable_vote.weight
        assert num_converted == surplus
      else:
        # No winner, eliminate the bottom candidate.
        eliminated_alt = sorted_scores[-1][0]
        eliminated_score = sorted_scores[-1][1]
        if self._verbose:
          print(f"No winner. Quota = {quota}. Eliminating candidate: " +
                f"{eliminated_alt} with score: {eliminated_score}")
        elim_count = sorted_scores[-1][1]
        losers.insert(0, eliminated_alt)
        loser_scores.insert(0, eliminated_score)
        # All of the votes with this alternative as the top is converted.
        votes_counted = 0
        for mutable_vote in all_votes:
          if (mutable_vote.idx < len(mutable_vote.vote) and
              mutable_vote.vote[mutable_vote.idx] == eliminated_alt and
              mutable_vote.weight > 0):
            # find the next one in the list still in the running.
            new_idx = self._next_idx_in_the_running(mutable_vote, winners,
                                                    losers)
            mutable_vote.idx = new_idx
            votes_counted += mutable_vote.weight
        assert votes_counted == elim_count
    ranking = winners + losers
    scores = []
    win_score_base = profile.num_alternatives() * 2
    lose_score_base = profile.num_alternatives()
    for winner_score in winner_scores:
      scores.append(float(str(win_score_base) + "." + str(winner_score)))
      win_score_base -= 1
    for loser_score in loser_scores:
      scores.append(float(str(lose_score_base) + "." + str(loser_score)))
      lose_score_base -= 1
    outcome = base.RankOutcome(rankings=ranking, scores=scores)
    return outcome
