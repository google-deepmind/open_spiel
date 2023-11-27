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

"""Tests for open_spiel.python.voting.borda."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.voting import base
from open_spiel.python.voting import borda


class BordaVotingTest(parameterized.TestCase):

  def test_borda_setup(self):
    method = borda.BordaVoting()
    self.assertEqual(method.name(), "borda")

  @parameterized.named_parameters(
      dict(testcase_name="uniform votes",
           votes=[["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]],
           ranking=["a", "b", "c"],
           scores=[5, 3, 1]),
      dict(testcase_name="weighted votes",
           votes=[
               base.WeightedVote(1, ["a", "b", "c"]),
               base.WeightedVote(2, ["a", "c", "b"]),
               base.WeightedVote(3, ["b", "a", "c"])
           ],
           ranking=["a", "b", "c"],
           scores=[9, 7, 2]))
  def test_borda_basic_run(self, votes, ranking, scores):
    profile = base.PreferenceProfile(votes=votes)
    method = borda.BordaVoting()
    outcome = method.run_election(profile)
    with self.subTest("ranking correct"):
      self.assertListEqual(outcome.ranking, ranking)
    with self.subTest("scores correct"):
      self.assertListEqual(outcome.scores, scores)


if __name__ == "__main__":
  absltest.main()
