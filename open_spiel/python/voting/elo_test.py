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


if __name__ == "__main__":
  absltest.main()
