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

"""Tests for open_spiel.python.voting.base."""

from absl.testing import absltest

import numpy as np

from open_spiel.python.voting import base


class BaseTest(absltest.TestCase):

  def test_basic_preference_profile(self):
    # Create a preference profile from preferences:
    #
    #    a > b > c > d
    #    b > d > a > c
    #    a > c > d > b
    #    d > b > c > a
    #
    # Each has a weight of 1 by default. E.g. each corresponds to one voter.
    votes = [
        ["a", "b", "c", "d"],
        ["b", "d", "a", "c"],
        ["a", "c", "d", "b"],
        ["d", "b", "c", "a"]
    ]
    profile = base.PreferenceProfile(votes=votes)
    self.assertLen(profile.votes, 4)
    self.assertEqual(profile.total_weight(), 4)

  def test_basic_preference_profile_weighted(self):
    # Create a weighted preference profile from preferences:
    #
    #    1:  a > b > c
    #    2:  a > c > b
    #    3:  b > a > c
    #
    # Each vote has a weight of 1, 2, and 3 respectively.
    votes = [
        base.WeightedVote(1, ["a", "b", "c"]),
        base.WeightedVote(2, ["a", "c", "b"]),
        base.WeightedVote(3, ["b", "a", "c"])
    ]
    profile = base.PreferenceProfile(votes=votes)
    self.assertLen(profile.votes, 3)
    self.assertEqual(profile.total_weight(), 6)

  def test_preference_profile_incremental_group(self):
    # Create a weighted preference profile from preferences:
    #
    #    1:  a > b > c
    #    2:  a > c > b
    #    3:  b > a > c
    #
    # by incrementally adding individual groups and then grouping them.
    profile = base.PreferenceProfile()
    for _ in range(1):
      profile.add_vote(["a", "b", "c"])
    for _ in range(2):
      profile.add_vote(["a", "c", "b"])
    for _ in range(3):
      profile.add_vote(["b", "a", "c"])

    # Assure there are 6 votes, each with weight 1.
    with self.subTest("All votes added correctly"):
      self.assertLen(profile.votes, 6)
      self.assertEqual(profile.total_weight(), 6)
    with self.subTest("Vote weight defaults to 1"):
      for vote in profile.votes:
        self.assertEqual(vote.weight, 1)

    # Group up the votes. Check that there are 3 but with total weight
    # unchanged (6).
    profile.group()
    with self.subTest("Grouping votes reduced to correct number"):
      self.assertLen(profile.votes, 3)
    with self.subTest("Grouping votes did not change total weight"):
      self.assertEqual(profile.total_weight(), 6)
    with self.subTest("Grouping votes computed weights correctly"):
      self.assertEqual(profile.get_weight(["a", "b", "c"]), 1)
      self.assertEqual(profile.get_weight(["a", "c", "b"]), 2)
      self.assertEqual(profile.get_weight(["b", "a", "c"]), 3)

  def test_pref_margin_matrices_strong_condorcet(self):
    votes = [
        base.WeightedVote(1, ["a", "b", "c"]),
        base.WeightedVote(1, ["a", "c", "b"]),
        base.WeightedVote(2, ["c", "a", "b"]),
        base.WeightedVote(1, ["b", "c", "a"]),
    ]
    profile = base.PreferenceProfile(votes=votes)

    pref_matrix = profile.pref_matrix()
    expected_pref_matrix = np.array(
        [[0, 4, 2],
         [1, 0, 2],
         [3, 3, 0]]
    )
    with self.subTest("Preference matrix calculated correctly."):
      self.assertTrue(np.array_equal(pref_matrix, expected_pref_matrix))

    margin_matrix = profile.margin_matrix()
    expected_margin_matrix = np.array(
        [[0, 3, -1],
         [-3, 0, -1],
         [1, 1, 0]]   # <-- all positive, except diagonal:
    )                 #     "c" is a strong Condorcet winner.
    with self.subTest("Expected margin matrix calculated correctly."):
      self.assertTrue(np.array_equal(margin_matrix, expected_margin_matrix))

    # Check that there is exactly one strong Condorcet winner.
    condorcet_winners = profile.condorcet_winner(strong=True,
                                                 margin_matrix=margin_matrix)
    with self.subTest("Exactly one strong Condorcet winner found."):
      self.assertListEqual(condorcet_winners, ["c"])

    # A strong Condorcet winner is also a weak Condorcet winner, by definition.
    condorcet_winners = profile.condorcet_winner(strong=False,
                                                 margin_matrix=margin_matrix)
    with self.subTest("A strong Cond. winner is also a weak Cond. winner."):
      self.assertListEqual(condorcet_winners, ["c"])

  def test_weak_condorcet(self):
    votes = [
        base.WeightedVote(1, ["a", "b", "c"]),
        base.WeightedVote(1, ["a", "c", "b"]),
        base.WeightedVote(1, ["c", "a", "b"]),
        base.WeightedVote(1, ["b", "c", "a"]),
    ]
    profile = base.PreferenceProfile(votes=votes)

    # Leads to margin matrix:
    # [[ 0  2  0]
    #  [-2  0  0]
    #  [ 0  0  0]]
    # ==> no strong Condorcet winners, and two weak Condorcet winners
    margin_matrix = profile.margin_matrix()

    strong_condorcet_winners = profile.condorcet_winner(
        strong=True, margin_matrix=margin_matrix)
    with self.subTest("No strong Condorect winner found."):
      self.assertListEqual(strong_condorcet_winners, [])

    # A strong Condorcet winner is also a weak Condorcet winner, by definition.
    weak_condorcet_winners = profile.condorcet_winner(
        strong=False, margin_matrix=margin_matrix)
    self.assertLen(weak_condorcet_winners, 2)
    with self.subTest("Found all weak Condorcet winners."):
      self.assertCountEqual(["a", "c"], weak_condorcet_winners)


if __name__ == "__main__":
  absltest.main()
