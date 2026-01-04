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

"""Tests for open_spiel.python.voting.approval."""

from absl.testing import absltest

from open_spiel.python.voting import approval
from open_spiel.python.voting import base


class ApprovalVotingTest(absltest.TestCase):

  def test_approval_name_correct(self):
    method = approval.ApprovalVoting(k=7)
    self.assertEqual(method.name(), "approval(k=7)")

  def test_approval_basic_run(self):
    votes = [
        ["a", "b", "c", "d"],
        ["b", "d", "a", "c"],
        ["a", "c", "d", "b"],
        ["d", "b", "c", "a"]
    ]
    profile = base.PreferenceProfile(votes=votes)
    method = approval.ApprovalVoting(k=2)
    outcome = method.run_election(profile)
    with self.subTest("Approval voting gets basic ranking correct"):
      self.assertTrue(outcome.ranking == ["b", "d", "a", "c"] or
                      outcome.ranking == ["b", "a", "d", "c"])
    with self.subTest("Approval voting gets basic scores correct"):
      self.assertListEqual(outcome.scores, [3, 2, 2, 1])

  def test_approval_basic_run_with_weights(self):
    votes = [
        base.WeightedVote(1, ["a", "b", "c", "d"]),
        base.WeightedVote(2, ["b", "d", "a", "c"]),
        base.WeightedVote(3, ["a", "c", "d", "b"]),
        base.WeightedVote(4, ["d", "b", "c", "a"])
    ]
    profile = base.PreferenceProfile(votes=votes)
    method = approval.ApprovalVoting(k=2)
    outcome = method.run_election(profile)
    with self.subTest("Approval voting gets weighted ranking correct"):
      self.assertListEqual(outcome.ranking, ["b", "d", "a", "c"])
    with self.subTest("Approval voting gets weighted scores correct"):
      self.assertListEqual(outcome.scores, [7, 6, 4, 3])


if __name__ == "__main__":
  absltest.main()
