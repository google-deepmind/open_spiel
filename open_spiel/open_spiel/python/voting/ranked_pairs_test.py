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

"""Tests for open_spiel.python.voting.ranked_pairs."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.voting import base
from open_spiel.python.voting import ranked_pairs


class RankedPairsTest(absltest.TestCase):

  def test_ranked_pairs_wikipedia_example1(self):
    alternatives = ["w", "x", "y", "z"]
    votes = [
        base.WeightedVote(7, ["w", "x", "z", "y"]),
        base.WeightedVote(2, ["w", "y", "x", "z"]),
        base.WeightedVote(4, ["x", "y", "z", "w"]),
        base.WeightedVote(5, ["x", "z", "w", "y"]),
        base.WeightedVote(1, ["y", "w", "x", "z"]),
        base.WeightedVote(8, ["y", "z", "w", "x"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = ranked_pairs.RankedPairsVoting()
    outcome = method.run_election(profile)
    with self.subTest("Ranking and scores"):
      self.assertListEqual(outcome.ranking, ["w", "x", "y", "z"])
      self.assertListEqual(outcome.scores, [29, 19, 3, 0])
    with self.subTest("Check the graph"):
      expected_graph = np.array(
          [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
      )
      self.assertTrue(np.array_equal(outcome.graph, expected_graph))

  def test_ranked_pairs_wikipedia_example2(self):
    alternatives = ["Memphis", "Nashville", "Chattanooga", "Knoxville"]
    votes = [
        base.WeightedVote(
            42, ["Memphis", "Nashville", "Chattanooga", "Knoxville"]
        ),
        base.WeightedVote(
            26, ["Nashville", "Chattanooga", "Knoxville", "Memphis"]
        ),
        base.WeightedVote(
            15, ["Chattanooga", "Knoxville", "Nashville", "Memphis"]
        ),
        base.WeightedVote(
            17, ["Knoxville", "Chattanooga", "Nashville", "Memphis"]
        ),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = ranked_pairs.RankedPairsVoting()
    outcome = method.run_election(profile)
    with self.subTest("Ranking and scores"):
      self.assertListEqual(
          outcome.ranking, ["Nashville", "Chattanooga", "Knoxville", "Memphis"]
      )
      self.assertListEqual(outcome.scores, [186, 98, 16, 0])
    with self.subTest("Check the graph"):
      expected_graph = np.array(
          [[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]]
      )
      self.assertTrue(np.array_equal(outcome.graph, expected_graph))

  def test_meeple_pentathlon(self):
    alternatives = ["A", "B", "C"]
    votes = [
        base.WeightedVote(1, ["A", "B", "C"]),
        base.WeightedVote(1, ["A", "C", "B"]),
        base.WeightedVote(2, ["C", "A", "B"]),
        base.WeightedVote(1, ["B", "C", "A"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = ranked_pairs.RankedPairsVoting()
    outcome = method.run_election(profile)
    with self.subTest("Ranking and scores"):
      self.assertListEqual(outcome.ranking, ["C", "A", "B"])
      self.assertListEqual(outcome.scores, [5, 3, 0])
    with self.subTest("Check the graph"):
      # A -> B, C -> A, C -> B
      expected_graph = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 0]])
      self.assertTrue(np.array_equal(outcome.graph, expected_graph))

  def test_ranked_pairs_simple_cycle(self):
    alternatives = ["A", "B"]
    votes = [
        base.WeightedVote(1, ["A", "B"]),
        base.WeightedVote(1, ["B", "A"]),
    ]
    profile = base.PreferenceProfile(votes=votes, alternatives=alternatives)
    method = ranked_pairs.RankedPairsVoting()
    outcome = method.run_election(profile)
    with self.subTest("Check the graph is empty"):
      expected_graph = np.array(
          [[0, 0], [0, 0]]
      )
      self.assertTrue(np.array_equal(outcome.graph, expected_graph))
    with self.subTest("Rankings and scores"):
      self.assertTrue(outcome.ranking == ["A", "B"] or
                      outcome.ranking == ["B", "A"])
      self.assertListEqual(outcome.scores, [0, 0])

if __name__ == "__main__":
  absltest.main()
