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

"""Tests for open_spiel.python.voting.plurality."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.voting import base
from open_spiel.python.voting import plurality

_SIMPLE_VOTE = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
_SIMPLE_WINNER = (_SIMPLE_VOTE, "a")
_WEIGHTED_WINNER = (_SIMPLE_VOTE, [1, 2, 3], [3, 3, 0], ["a", "b"])


class PluralityVotingTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.method = plurality.PluralityVoting()

  @parameterized.parameters(_SIMPLE_WINNER)
  def test_plurality_with_votes_in_profile_constructor(self, votes, winner):
    profile = base.PreferenceProfile(votes=votes)
    outcome = self.method.run_election(profile)
    self.assertEqual(outcome.ranking[0], winner)

  @parameterized.parameters(_SIMPLE_WINNER)
  def test_plurality_with_alternatives_specified(self, votes, winner):
    profile = base.PreferenceProfile(alternatives=["c", "b", "a"])
    for vote in votes:
      profile.add_vote(vote)
    outcome = self.method.run_election(profile)
    self.assertEqual(outcome.ranking[0], winner)

  @parameterized.parameters(_SIMPLE_WINNER)
  def test_plurality_with_no_default_votes(self, votes, winner):
    profile = base.PreferenceProfile()
    for vote in votes:
      profile.add_vote(vote)
    outcome = self.method.run_election(profile)
    self.assertEqual(outcome.ranking[0], winner)

  @parameterized.parameters(_WEIGHTED_WINNER)
  def test_plurality_with_weighted_votes(self, votes, weights,
                                         correct_scores, winner):
    profile = base.PreferenceProfile()
    for i, vote in enumerate(votes):
      profile.add_vote(vote, weight=weights[i])
    outcome = self.method.run_election(profile)

    with self.subTest("Weighted score correctly calculated."):
      self.assertListEqual(correct_scores, outcome.scores)
    with self.subTest("Winners take the top spots in the ranking."):
      self.assertCountEqual(outcome.ranking[:len(winner)], winner)


if __name__ == "__main__":
  absltest.main()
