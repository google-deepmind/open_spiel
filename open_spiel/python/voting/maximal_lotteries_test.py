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

"""Tests for open_spiel.python.voting.maximal_lotteries."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from open_spiel.python.voting import base
from open_spiel.python.voting import maximal_lotteries


class MaximalLotteriesTest(parameterized.TestCase):
  @parameterized.named_parameters(("iterative", True), ("non-iterative", False))
  def test_stv_records_number(self, iterate):
    method = maximal_lotteries.MaximalLotteriesVoting(iterative=iterate)
    self.assertEqual(
        method.name(), f"maximal_lotteries(iterative={iterate})"
    )

  def test_maximal_lotteries_basic_run(self):
    # "a" is a dominant strategy of the margin game, so it should be chosen with
    # probablity 1.
    votes = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
    profile = base.PreferenceProfile(votes=votes)
    method = maximal_lotteries.MaximalLotteriesVoting()
    outcome = method.run_election(profile)
    with self.subTest("Top-rank the condorcet winner"):
      self.assertEqual(outcome.ranking[0], "a")
    with self.subTest("Check extreme scores"):
      self.assertAlmostEqual(outcome.scores[0], 1.0)
      self.assertAlmostEqual(outcome.scores[1], 0.0)
      self.assertAlmostEqual(outcome.scores[2], 0.0)

  def test_maximal_lotteries_basic_iterative(self):
    votes = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
    profile = base.PreferenceProfile(votes=votes)
    # "a" is a dominant strategy, so in the iterative version it should be
    # chosen first, leading to a new matrix with the first row and column
    # deleted. This then means that "b" is dominant in the subgame.
    expected_margin_matrix = np.array([
        [0, 1, 3],
        [-1, 0, 1],
        [-3, -1, 0]])
    with self.subTest("Check margin matrix"):
      self.assertTrue(np.array_equal(profile.margin_matrix(),
                                     expected_margin_matrix))
    method = maximal_lotteries.MaximalLotteriesVoting(iterative=True)
    outcome = method.run_election(profile)
    with self.subTest("Check ranking"):
      self.assertListEqual(outcome.ranking, ["a", "b", "c"])
    with self.subTest("Check scores"):
      self.assertAlmostEqual(outcome.scores[0], 3.0)
      self.assertAlmostEqual(outcome.scores[1], 2.0)
      self.assertAlmostEqual(outcome.scores[2], 1.0)

  def test_maximal_lotteries_cycle(self):
    # Cyclical profile leads to a Rock, Paper, Scissors margin game.
    votes = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
    profile = base.PreferenceProfile(votes=votes)
    method = maximal_lotteries.MaximalLotteriesVoting()
    outcome = method.run_election(profile)
    with self.subTest("Check prob 1/3"):
      self.assertAlmostEqual(outcome.scores[0], 1.0 / 3.0)
    with self.subTest("Check uniform"):
      self.assertAlmostEqual(outcome.scores[0], outcome.scores[1])
      self.assertAlmostEqual(outcome.scores[1], outcome.scores[2])

  def test_maximal_lotteries_iterative_cycle(self):
    # Cyclical profile leads to a Rock, Paper, Scissors margin game.
    # Iterative maximal lotteries should yield the same result as the
    # non-iterative version.
    votes = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
    profile = base.PreferenceProfile(votes=votes)
    method = maximal_lotteries.MaximalLotteriesVoting(iterative=True)
    outcome = method.run_election(profile)
    with self.subTest("Check prob 1/3"):
      self.assertAlmostEqual(outcome.scores[0], 1.0 / 3.0)
    with self.subTest("Check uniform"):
      self.assertAlmostEqual(outcome.scores[0], outcome.scores[1])
      self.assertAlmostEqual(outcome.scores[1], outcome.scores[2])


if __name__ == "__main__":
  absltest.main()
