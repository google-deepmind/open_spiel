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

"""Tests for open_spiel.python.voting.schulze."""

from absl.testing import absltest

from open_spiel.python.voting import base
from open_spiel.python.voting import schulze


class SchulzeTest(absltest.TestCase):
  def test_shulze_construction(self):
    method = schulze.SchulzeVoting()
    self.assertEqual(method.name(), "schulze")

  def test_shulze_wikipedia_example(self):
    votes = [
        base.WeightedVote(5, ["A", "C", "B", "E", "D"]),
        base.WeightedVote(5, ["A", "D", "E", "C", "B"]),
        base.WeightedVote(8, ["B", "E", "D", "A", "C"]),
        base.WeightedVote(3, ["C", "A", "B", "E", "D"]),
        base.WeightedVote(7, ["C", "A", "E", "B", "D"]),
        base.WeightedVote(2, ["C", "B", "A", "D", "E"]),
        base.WeightedVote(7, ["D", "C", "E", "B", "A"]),
        base.WeightedVote(8, ["E", "B", "A", "D", "C"])
    ]
    profile = base.PreferenceProfile(votes=votes,
                                     alternatives=["A", "B", "C", "D", "E"])
    method = schulze.SchulzeVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking, ["E", "A", "C", "B", "D"])
    self.assertListEqual(outcome.scores, [111, 88, 62, 33, 0])

  def test_meeple_pentathlon(self):
    alternatives = ["A", "B", "C"]
    votes = [
        base.WeightedVote(1, ["A", "B", "C"]),
        base.WeightedVote(1, ["A", "C", "B"]),
        base.WeightedVote(2, ["C", "A", "B"]),
        base.WeightedVote(1, ["B", "C", "A"])
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = schulze.SchulzeVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking, ["C", "A", "B"])
    self.assertListEqual(outcome.scores, [7, 4, 0])


if __name__ == "__main__":
  absltest.main()
