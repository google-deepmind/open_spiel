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

"""Tests for open_spiel.python.voting.kemeny_young."""

from absl.testing import absltest

from open_spiel.python.voting import base
from open_spiel.python.voting import kemeny_young


class KemenyYoungTest(absltest.TestCase):

  def test_ranked_pairs_wikipedia_example(self):
    alternatives = ["Memphis", "Nashville", "Chattanooga", "Knoxville"]
    votes = [
        base.WeightedVote(42,
                          ["Memphis", "Nashville", "Chattanooga", "Knoxville"]),
        base.WeightedVote(26,
                          ["Nashville", "Chattanooga", "Knoxville", "Memphis"]),
        base.WeightedVote(15,
                          ["Chattanooga", "Knoxville", "Nashville", "Memphis"]),
        base.WeightedVote(17,
                          ["Knoxville", "Chattanooga", "Nashville", "Memphis"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = kemeny_young.KemenyYoungVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking,
                         ["Nashville", "Chattanooga", "Knoxville", "Memphis"])
    self.assertListEqual(outcome.scores, [194, 141, 58, 0])

  def test_meeple_pentathlon(self):
    alternatives = ["A", "B", "C"]
    votes = [
        base.WeightedVote(1, ["A", "B", "C"]),
        base.WeightedVote(1, ["A", "C", "B"]),
        base.WeightedVote(2, ["C", "A", "B"]),
        base.WeightedVote(1, ["B", "C", "A"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = kemeny_young.KemenyYoungVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking, ["C", "A", "B"])
    self.assertListEqual(outcome.scores, [6, 4, 0])


if __name__ == "__main__":
  absltest.main()
