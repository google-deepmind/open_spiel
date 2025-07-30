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
"""Basic tests for Elo."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import elo
import pyspiel

SEED = 0


class EloTest(absltest.TestCase):
  """Elo rating system tests."""

  def test_simple_case_meeple_pentathlon(self):
    # Meeple Pentathlon example from the VasE paper
    # (https://arxiv.org/abs/2312.03121)
    #    1: A > B > C
    #    1: A > C > B
    #    2: C > A > B
    #    1: B > C > A
    # Here, the first and last player have provably equal Elo ratings.
    win_matrix = np.asarray([[0, 4, 2], [1, 0, 2], [3, 3, 0]])
    ratings = elo.compute_ratings_from_matrices(win_matrix)
    self.assertAlmostEqual(ratings[0], ratings[2])
    self.assertLess(ratings[1], ratings[0])
    self.assertLess(ratings[1], ratings[2])

    # Now, from match records.
    match_records = []
    match_records.extend([pyspiel.elo.MatchRecord("A", "B")] * 4)
    match_records.extend([pyspiel.elo.MatchRecord("A", "C")] * 2)
    match_records.extend([pyspiel.elo.MatchRecord("B", "A")] * 1)
    match_records.extend([pyspiel.elo.MatchRecord("B", "C")] * 2)
    match_records.extend([pyspiel.elo.MatchRecord("C", "A")] * 3)
    match_records.extend([pyspiel.elo.MatchRecord("C", "B")] * 3)
    ratings_map = pyspiel.elo.compute_ratings_from_match_records(match_records)
    self.assertAlmostEqual(ratings_map["A"], ratings[0])
    self.assertAlmostEqual(ratings_map["B"], ratings[1])
    self.assertAlmostEqual(ratings_map["C"], ratings[2])

  def test_simple_case_direct_pyspiel(self):
    # Simple case: a > b. First, specify the draws matrix. Calls the Elo
    # wrapper directly.
    ratings1 = pyspiel.elo.compute_ratings_from_matrices(
        win_matrix=[[0, 2], [1, 0]], draw_matrix=[[0, 0], [0, 0]]
    )
    self.assertGreater(ratings1[0], ratings1[1])
    self.assertAlmostEqual(ratings1[1], 0.0, places=6)

    # Testing default when excluding draws matrix.
    ratings2 = pyspiel.elo.compute_ratings_from_matrices(
        win_matrix=[[0, 2], [1, 0]]
    )
    self.assertAlmostEqual(ratings1[0], ratings2[0], places=6)
    self.assertAlmostEqual(ratings1[1], ratings2[1], places=6)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
