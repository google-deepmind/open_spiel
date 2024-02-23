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

from open_spiel.python.voting import base
from open_spiel.python.voting import copeland


class CopelandVotingTest(absltest.TestCase):
  def test_copeland_construction(self):
    method = copeland.CopelandVoting()
    self.assertEqual(method.name(), "copeland")

  def test_copeland_basic_run(self):
    votes = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
    profile = base.PreferenceProfile(votes=votes)
    method = copeland.CopelandVoting()
    outcome = method.run_election(profile)
    self.assertListEqual(outcome.ranking, ["a", "b", "c"])
    self.assertListEqual(outcome.scores, [2.0, 1.0, 0.0])

  def test_copeland_basic_run2(self):
    votes = [
        base.WeightedVote(1, ["a", "b", "c"]),
        base.WeightedVote(2, ["a", "c", "b"]),
        base.WeightedVote(3, ["b", "a", "c"]),
    ]
    profile = base.PreferenceProfile(votes=votes)
    method = copeland.CopelandVoting()
    outcome = method.run_election(profile)
    self.assertTrue(outcome.ranking == ["a", "b", "c"] or
                    outcome.ranking == ["b", "a", "c"])
    self.assertListEqual(outcome.scores, [1.5, 1.5, 0.0])


if __name__ == "__main__":
  absltest.main()
