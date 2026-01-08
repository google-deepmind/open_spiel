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

"""Tests for open_spiel.python.voting.stv."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.voting import base
from open_spiel.python.voting import stv


class STVTest(parameterized.TestCase):
  @parameterized.named_parameters(("four", 4), ("one", 1))
  def test_stv_records_number(self, num):
    method = stv.STVVoting(num_winners=num)
    self.assertEqual(
        method.name(), f"single_transferable_vote(num_winners={num})"
    )

  def test_ranked_pairs_wikipedia_example(self):
    alternatives = ["Orange", "Pear", "Strawberry", "Cake", "Chocolate",
                    "Hamburger", "Chicken"]
    votes = [
        base.WeightedVote(4, ["Orange", "Pear"]),
        base.WeightedVote(7, ["Pear", "Strawberry", "Cake"]),
        base.WeightedVote(1, ["Strawberry", "Cake", "Pear"]),
        base.WeightedVote(3, ["Cake", "Chocolate", "Strawberry"]),
        base.WeightedVote(1, ["Cake", "Chocolate", "Hamburger"]),
        base.WeightedVote(4, ["Hamburger"]),
        base.WeightedVote(3, ["Chicken", "Hamburger"]),
    ]
    profile = base.PreferenceProfile(votes=votes,
                                     alternatives=alternatives)
    method = stv.STVVoting(num_winners=3)
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking,
                         ["Pear", "Cake", "Hamburger", "Orange", "Chicken",
                          "Strawberry", "Chocolate"])
    self.assertListEqual(outcome.scores, [14.7, 13.6, 12.7, 7.4, 6.3, 5.2, 4.0])

  def test_meeple_pentathlon(self):
    alternatives = ["A", "B", "C"]
    votes = [
        base.WeightedVote(1, ["A", "B", "C"]),
        base.WeightedVote(1, ["A", "C", "B"]),
        base.WeightedVote(2, ["C", "A", "B"]),
        base.WeightedVote(1, ["B", "C", "A"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = stv.STVVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking, ["C", "A", "B"])
    self.assertListEqual(outcome.scores, [6.3, 3.2, 2.1])


if __name__ == "__main__":
  absltest.main()
