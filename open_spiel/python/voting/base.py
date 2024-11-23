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

"""Base classes for voting methods."""

import abc
from typing import Dict, List, NamedTuple, Tuple, Union
import numpy as np


# The id of an alternative can be a string or an integer.
AlternativeId = Union[str, int]

# List of alternative ids.
PreferenceList = List[AlternativeId]


# Basic type to represent a vote.
#    - The weight is an integer representing the number of voters
#    - The vote is a list of alternative ids, e.g. ["a", "b", "c"],
#      corresponding to a preference a > b > c.
class WeightedVote(NamedTuple):
  weight: int
  vote: PreferenceList


class PreferenceProfile(object):
  """Base class for preference profiles.

  IMPORTANT NOTE: see the assumptions below about indexing of alternatives.
  """
  _votes: List[WeightedVote]  # Tracks cast votes along with their count
  _alternatives_dict: Dict[AlternativeId, int]  # Maps ID to index
  # Identifiers for all possible alternatives
  _alternatives_ids: List[AlternativeId]

  def __init__(
      self,
      votes: Union[List[PreferenceList], List[WeightedVote], None] = None,
      alternatives: Union[List[AlternativeId], None] = None,
  ):
    """Initialize the preference profile.

    Args:
      votes: Either (i) a list of lists, each containing ids of alternatives,
        e.g. ["a", "b", "c"] signifiying a > b > c, or None for no votes, or
        (ii) a list of Vote tuples containing the weight and vote.
      alternatives: a list of alternatives ids.

    Note regarding how alternatives are indexed: if the second argument is
    passed, then the index of each alternative (e.g. when calling functions
    like margin_matrix etc.) will be assigned 0 up to the (number of
    alternatives) - 1 in the order of the list. If this argument is omitted,
    then alternatives will be indexed depending on when they are first seen
    (i.e. via a add_vote method) and so (only) in the latter case the indexing
    could depend on the order votes are added. Hence it is advised to pass in
    the list of alternatives to this function whenever they are known ahead of
    time.

    The alternatives_dict property below will return a dictionary of alternative
    IDs to index.
    """
    # List of Vote named tuples from above.
    self._votes: List[WeightedVote] = []
    # alternative id -> index (used for registering alternatives)
    self._alternatives_dict: Dict[AlternativeId, int] = {}
    # IDs (labels) of each alternative (usually strings). The alternative's
    # index is then the index of this array.
    self._alternatives_ids: List[AlternativeId] = []

    # Register the alternatives and add the votes, if any are provided.
    if alternatives is not None:
      for alternative in alternatives:
        self._register_alternative(alternative)
    if votes is not None:
      for vote in votes:
        self.add_vote(vote)
    if self._votes and not self._alternatives_ids:
      self._register_alternatives_from_votes()

  def _register_index_based_alternatives(self, num: int):
    """Register indices up to num-1 as possible alternatives."""
    for idx in range(num):
      self._register_alternative(idx)

  def _register_alternative(self, alternative: AlternativeId):
    """Add this alternative to interal recors if not already there."""
    idx = self._alternatives_dict.get(alternative)
    if idx is None:
      self._alternatives_ids.append(alternative)
      self._alternatives_dict[alternative] = len(self._alternatives_ids) - 1
    assert (self._alternatives_ids[self._alternatives_dict[alternative]]
            == alternative)

  def _register_alternatives_from_votes(self):
    for vote in self._votes:
      for alternative in vote:
        self._register_alternative(alternative)

  def add_vote(
      self, vote: Union[PreferenceList, WeightedVote], weight: int = 1
  ):
    """Add a vote to this preference profile.

    Args:
      vote: Either (i) a list of ids, e.g. ["a", "b", "c"] signifying a > b > c,
          or, (ii) a Vote tuple containing both the weight and the vote of the
          form in (i).
      weight: the count, i.e. how many people have submitted this vote. Only
          used when the first argument is a list.
    """
    # For now support only integral weights (counts). Makes some things easier,
    # like N(x,y) and the margin matrices can be integers. Should be easy to
    # extend if we need to.
    assert isinstance(weight, int)
    assert weight > 0
    if isinstance(vote, WeightedVote):
      self._votes.append(vote)
      for alternative in vote.vote:
        self._register_alternative(alternative)
    else:
      weighted_vote = WeightedVote(weight, vote)
      self._votes.append(weighted_vote)
      for alternative in vote:
        self._register_alternative(alternative)

  def add_vote_from_values(
      self,
      values: Union[List[float], List[int]],
      tie_tolerance: float = 1e-10,
      weight: int = 1,
  ):
    """Adds a vote from a list of values.

    Note: this list is expected to cover all of the alternatives.

    WARNING: to ensure that ties are broken randomly, small random values are
    added to the values (within [0, tie_tolarance]). If the values are smaller
    than the tie_tolerance, this can be disabled by setting the tie_tolerance to
    0.

    Does not add the vote if the values are all within tie_tolerance of each
    other. For all others, adds a uniform * tie_tolerance to break ties.

    If the alternatives ids are not registered for this profile yet, then this
    method uses the indices of these values as the alternative IDs. Otherwise,
    the length of the array must be equal to the number of alternatives.

    Args:
      values: a list or numpy array of values for the alternative labeled by
          the index.
      tie_tolerance: a numerical threshold for determining ties.
      weight: the weight for the resulting vote.
    """
    # Check if any alternatives are registered for this profile. If not, then
    # first register ids for them all first.
    if not self._alternatives_ids:
      self._register_index_based_alternatives(len(values))
    else:
      assert len(values) == len(self._alternatives_ids)
    vals_copy = np.copy(np.asarray(values))
    max_val = vals_copy.max()
    min_val = vals_copy.min()
    if (max_val - min_val) < tie_tolerance:
      print(f"Warning: not casting vote from values: {vals_copy}")
      return
    # Add noise for tie_breaking
    vals_copy += tie_tolerance * np.random.uniform(size=len(vals_copy))
    vote = np.argsort(-vals_copy)
    # The vote is currently based on indices. Now convert to names.
    alternatives = self.alternatives
    assert alternatives
    assert len(alternatives) == len(vote)
    named_vote = []
    for idx in vote:
      assert 0 <= idx < len(alternatives)
      named_vote.append(alternatives[idx])
    self.add_vote(named_vote, weight=weight)

  @property
  def votes(self) -> List[WeightedVote]:
    """Returns a list of votes."""
    return self._votes

  @property
  def alternatives(self) -> List[AlternativeId]:
    """Returns a list of alternatives."""
    return self._alternatives_ids

  @property
  def alternatives_dict(self) -> Dict[AlternativeId, int]:
    """Returns a dict of alternative id -> index for each alternative."""
    return self._alternatives_dict

  def num_alternatives(self) -> int:
    return len(self._alternatives_ids)

  def num_votes(self) -> int:
    """Returns the number of votes."""
    total = 0
    for vote in self._votes:
      total += vote.weight
    return total

  def pref_matrix(self) -> np.ndarray:
    """Returns the candidate preference matrix for this profile.

    Define N(x,y) as number of voters that prefer x > y. The candidate
    preference matrix is one whose entries are N(x,y) for row x and column y.
    """
    # First map the alternatives to indices.
    m = self.num_alternatives()
    mat = np.zeros(shape=(m, m), dtype=np.int32)
    for vote in self._votes:
      vote_len = len(vote.vote)
      for i in range(vote_len):
        for j in range(i + 1, vote_len):
          # vote.vote[i] > vote.vote[j]
          idx_i = self._alternatives_dict[vote.vote[i]]
          idx_j = self._alternatives_dict[vote.vote[j]]
          mat[idx_i, idx_j] += vote.weight
    return mat

  def margin_matrix(self) -> np.ndarray:
    """Returns the margin matrix for this profile.

    Define N(x,y) = number of voters that prefer x > y. The margin matrix
    is a num_alternatives x num_alternatives whose entry at (r,c) is:
    delta(r,c) = N(r, c) - N(c, r). The r and c refer to columns, which
    correspond to the indices in the list returned by self.alternatives.
    """
    pref_matrix = self.pref_matrix()
    return pref_matrix - pref_matrix.T

  def condorcet_winner(
      self, strong: bool = True, margin_matrix: Union[np.ndarray, None] = None
  ):
    """Returns the Condorcet winner(s).

    Args:
      strong: whether it's a strong Condorcet winner (see below).
      margin_matrix: the margin matrix (optional: only used to to avoid
          recomputing).

    Returns:
      A list containing the Condorcet winners. There may be multiple weak
      Condorcet winners, but there is at most one strong winner.

    A strong Condorcet winner is an alternative a* in A such that for all
    a' in A: N(a*, a') > N(a', a*). A weak Condorcet winner is a similar
    definition using great-than-or-equal-to >=.
    """
    condorcet_winners = []
    if margin_matrix is None:
      margin_matrix = self.margin_matrix()
    for alt_idx in range(self.num_alternatives()):
      if strong and np.all(np.delete(margin_matrix[alt_idx] > 0, alt_idx)):
        # Don't count the diagonal 0 in the checking of > 0.
        condorcet_winners.append(self._alternatives_ids[alt_idx])
      elif not strong and np.all(margin_matrix[alt_idx] >= 0):
        condorcet_winners.append(self._alternatives_ids[alt_idx])
    if strong:
      assert len(condorcet_winners) <= 1
    return condorcet_winners

  def group(self):
    """Group up the votes.

    This will combine multiple identical votes into the smallest set of unique
    weighted votes.
    """
    old_votes = self._votes
    self._votes = []
    while old_votes:
      vote = old_votes[0].vote
      total_weight = old_votes[0].weight
      del old_votes[0]
      i = 0
      while i < len(old_votes):
        if old_votes[i].vote == vote:
          total_weight += old_votes[i].weight
          del old_votes[i]
        else:
          i += 1
      self._votes.append(WeightedVote(total_weight, vote))

  def ungroup(self):
    """Splits the votes into individual votes (each with weight of 1)."""
    old_votes = self._votes
    self._votes = []
    for vote in old_votes:
      for _ in range(vote.weight):
        self._votes.append(WeightedVote(1, vote.vote))

  def __str__(self) -> str:
    """Get a string representation of this profile."""
    string = ""
    for vote in self._votes:
      string += str(vote) + "\n"
    return string

  def total_weight(self) -> int:
    w = 0
    for vote in self._votes:
      w += vote.weight
    return w

  def get_weight(self, vote: PreferenceList) -> int:
    total_weight = 0
    for v in self._votes:
      if v.vote == vote:
        total_weight += v.weight
    return total_weight

  def set_weight(self, index: int, value: int):
    self._votes[index] = self._votes[index]._replace(weight=value)

  def set_all_weights(self, value: int):
    """Sets the weight of all the votes to the specified value."""
    for i in range(len(self._votes)):
      self.set_weight(i, value)


class RankOutcome(object):
  """Basic object for outcomes of the voting methods."""

  def __init__(self, rankings=None, scores=None):
    self._rankings: List[AlternativeId] = rankings
    self._scores: List[float] = scores
    self._rank_dict: Dict[AlternativeId, int] = None
    if self._rankings is not None:
      self.make_rank_dict()

  def unpack_from(
      self, ranked_alternatives_and_scores: List[Tuple[AlternativeId, float]]
  ):
    """A rank outcome that comes packed as (alternative id, score) tuples."""
    self._rankings, self._scores = zip(*ranked_alternatives_and_scores)
    self._rankings = list(self._rankings)
    self._scores = list(self._scores)
    self.make_rank_dict()

  @property
  def ranking(self) -> List[AlternativeId]:
    """Returns an ordered list W of alternatives' ids (winner is first)."""
    return self._rankings

  @property
  def scores(self) -> List[float]:
    """Returns a alternative's scores S (in the same order as the ranking)."""
    return self._scores

  def ranking_with_scores(self) -> Tuple[List[AlternativeId], List[float]]:
    """Returns an ordered list of alternative ids and dict of scores W, S."""
    return self._rankings, self._scores

  def make_rank_dict(self):
    """Makes the rank dictionary from the rankings and scores."""
    self._rank_dict = {}
    for r, alt in enumerate(self._rankings):
      self._rank_dict[alt] = r

  def get_rank(self, alternative: AlternativeId) -> int:
    """Returns the rank of a specific alternative."""
    return self._rank_dict[alternative]

  def get_score(self, alternative: AlternativeId) -> float:
    """Returns the score of a specific alternative."""
    return self._scores[self.get_index(alternative)]

  def get_index(self, alternative: AlternativeId) -> int:
    """Returns the index of a specific alternative."""
    return self._rankings.index(alternative)

  def __str__(self) -> str:
    str_rep = "Rank: " + str(self._rankings) + "\n"
    if self._scores is not None:
      str_rep += "Scores: " + str(self._scores)
    return str_rep

  def pretty_table_string(self, top: Union[int, None] = None):
    """Return an easier-to-read table for the rankings and scores.

    Args:
      top: (optional) if specified, only returns the top `top` alternatives.

    Returns:
      An easier-to-read table string.
    """
    if top is None:
      top = len(self._rankings)
    max_len = -1
    for i, alt in enumerate(self._rankings):
      if i == top:
        break
      max_len = max(max_len, len(str(alt)))
    table_string = ""
    max_len += 1
    for i, alt in enumerate(self._rankings):
      if i == top:
        break
      score = self._scores[i]
      prefix = f"    Rank {i+1}: "
      while len(prefix) < 14:
        prefix += " "
      prefix += str(alt)
      while len(prefix) < (14 + max_len):
        prefix += " "
      table_string += f"{prefix} ({score})\n"
    return table_string

  def pretty_latex_table(
      self, header: Union[str, None] = None, top: Union[int, None] = None
  ):
    """Return an easier-to-read table string for the rankings and scores.

    The string returned include LaTeX formatting for putting the tables into
    papers.

    Args:
      header: (optional) if specified, uses this as the header of the table.
      top: (optional) if specified, only returns the top `top` alternatives.

    Returns:
      An easier-to-read table string (with LaTeX formattinf)
    """

    if top is None:
      top = len(self._rankings)
    table_string = "\\begin{center}\n\\begin{tabular}{|c|ll|}\n"
    if header is not None:
      table_string += "\\multicolumn{3}{c}{\\bf " + header + "}\\\\\n\\hline\n"
    table_string += "Rank & Agent & Score\\\\\n\\hline\n"
    for i, alt in enumerate(self._rankings):
      if i == top:
        break
      score = self._scores[i]
      # table_string += f"{i+1} & \\textsc" + "{"
      table_string += f"{i+1} & " + "{\\tt "
      table_string += f"{alt}" + "} & " + f"{score}\\\\\n"
    table_string += "\\hline\n"
    table_string += "\\end{tabular}\n\\end{center}"
    return table_string


class AbstractVotingMethod(metaclass=abc.ABCMeta):
  """Abstract base class for voting methods."""

  @abc.abstractmethod
  def __init__(self, **method_specific_kwargs):
    """Initializes the voting method.

    Args:
      **method_specific_kwargs: optional extra args.
    """

  @abc.abstractmethod
  def name(self) -> str:
    """Returns the name of the voting method."""

  @abc.abstractmethod
  def run_election(self, profile: PreferenceProfile) -> RankOutcome:
    """Runs the election and returns the result.

    Args:
        profile: a preference profile.

    Returns:
        a RankOutcome object that can be queried for the results.
    """

  def is_valid_profile(self, profile: PreferenceProfile) -> bool:
    """Returns true if a profile is valid.
 
    A valid profile is valid if it contains at least one vote and one
    alternative. Most voting schemes can't run unless the profile is valid.

    Args:
      profile: the profile to check.
    """
    return profile.num_votes() > 0 and profile.num_alternatives() > 0

