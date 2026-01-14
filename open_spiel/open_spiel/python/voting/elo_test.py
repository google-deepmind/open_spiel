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

"""Tests for Elo rating system wrapper."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.voting import base
from open_spiel.python.voting import elo


class EloTest(parameterized.TestCase):

  def test_meeple_pentathlon(self):
    alternatives = ["A", "B", "C"]
    votes = [
        base.WeightedVote(1, ["A", "B", "C"]),
        base.WeightedVote(1, ["A", "C", "B"]),
        base.WeightedVote(2, ["C", "A", "B"]),
        base.WeightedVote(1, ["B", "C", "A"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    ratings_map = elo.compute_ratings_from_preference_profile(profile)
    self.assertAlmostEqual(ratings_map["C"], ratings_map["A"])
    self.assertLess(ratings_map["B"], ratings_map["C"])

    # Now run it as a voting method using the same profile, and ensure that the
    # ratings are the same.
    method = elo.EloVoting()
    outcome = method.run_election(profile)
    self.assertAlmostEqual(outcome.get_score("A"), ratings_map["A"])
    self.assertAlmostEqual(outcome.get_score("B"), ratings_map["B"])
    self.assertAlmostEqual(outcome.get_score("C"), ratings_map["C"])

  def test_meeple_pentathlon_with_integer_alternatives(self):
    alternatives = [0, 1, 2]
    votes = [
        base.WeightedVote(1, [0, 1, 2]),
        base.WeightedVote(1, [0, 2, 1]),
        base.WeightedVote(2, [2, 0, 1]),
        base.WeightedVote(1, [1, 2, 0]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    ratings_map = elo.compute_ratings_from_preference_profile(profile)
    # Make sure alteratives with integer keys are in the ratings map.
    for alt in alternatives:
      self.assertIn(alt, ratings_map)
    # Make sure alteratives with string keys are NOT in the ratings map.
    for string_alt in [str(alt) for alt in alternatives]:
      self.assertNotIn(string_alt, ratings_map)
    self.assertAlmostEqual(ratings_map[2], ratings_map[0])
    self.assertLess(ratings_map[1], ratings_map[2])


if __name__ == "__main__":
  absltest.main()
