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
"""Basic tests for Soft Condorcet Optimization."""

from absl.testing import absltest
import numpy as np
import pyspiel
from open_spiel.python.voting import base
from open_spiel.python.voting import soft_condorcet_optimization as sco

SEED = 0


class SCOTest(absltest.TestCase):
  """Soft Condorcet Optimization tests."""

  def test_simple_case(self):
    # Simple case: a > b > c
    profile = base.PreferenceProfile(votes=[["a", "b", "c"]])
    solver = sco.SoftCondorcetOptimizer(profile, temperature=1)
    ratings, ranking = solver.run_solver(1000, learning_rate=0.01)
    alt_idx = profile.alternatives_dict
    for alt in ranking:
      print(f"{alt}: {ratings[alt_idx[alt]]}")
    self.assertGreater(ratings[0], ratings[1])
    self.assertGreater(ratings[1], ratings[2])

  def test_meeple_pentathlon_sigmoid(self):
    """Meeple pentathlon from the VasE paper using the sigmoid loss."""
    profile = base.PreferenceProfile(
        votes=[
            ["A", "B", "C"],
            ["A", "C", "B"],
            ["C", "A", "B"],
            ["C", "A", "B"],
            ["B", "C", "A"],
        ]
    )
    solver = sco.SoftCondorcetOptimizer(profile, batch_size=4, temperature=1)
    ratings, ranking = solver.run_solver(10000, learning_rate=0.01)
    alt_idx = profile.alternatives_dict
    for alt in ranking:
      print(f"{alt}: {ratings[alt_idx[alt]]}")
    # Correct ranking is C > A > B.
    self.assertGreater(ratings[2], ratings[0])
    self.assertGreater(ratings[0], ratings[1])

  def test_meeple_pentathlon_fenchel_young(self):
    """Meeple pentathlon from the VasE paper using the Fenchel-Young loss."""
    profile = base.PreferenceProfile(
        votes=[
            ["A", "B", "C"],
            ["A", "C", "B"],
            ["C", "A", "B"],
            ["C", "A", "B"],
            ["B", "C", "A"],
        ]
    )
    solver = sco.FenchelYoungOptimizer(
        profile=profile,
        batch_size=4,
        rating_lower_bound=0,
        rating_upper_bound=1000,
        sigma=20,
    )
    ratings, ranking = solver.run_solver(10000, learning_rate=0.1)
    alt_idx = profile.alternatives_dict
    for alt in ranking:
      print(f"{alt}: {ratings[alt_idx[alt]]}")
    # Like Elo, agent A will have a higher rating in Fenchel-Young.
    self.assertGreater(ratings[0], ratings[1])
    self.assertGreater(ratings[0], ratings[2])

  def test_cpp_meeple_pentathlon_sigmoid(self):
    # Tests the C++ implementation of the SCO with sigmoid solver.
    profile = base.PreferenceProfile(
        votes=[
            ["A", "B", "C"],
            ["A", "C", "B"],
            ["C", "A", "B"],
            ["C", "A", "B"],
            ["B", "C", "A"],
        ]
    )
    cpp_sco_solver = pyspiel.sco.SoftCondorcetOptimizer(
        profile.to_list_of_tuples(),
        rating_lower_bound=-100.0,
        rating_upper_bound=100.0,
        batch_size=4,
        temperature=1,
        rng_seed=SEED,
    )
    cpp_sco_solver.run_solver(10000, learning_rate=0.01)
    ratings_dict = cpp_sco_solver.ratings()
    self.assertGreater(ratings_dict["C"], ratings_dict["A"])
    self.assertGreater(ratings_dict["A"], ratings_dict["B"])

  def test_cpp_meeple_pentathlon_fenchel_young(self):
    # Tests the C++ implementation of the FY solver.
    profile = base.PreferenceProfile(
        votes=[
            ["A", "B", "C"],
            ["A", "C", "B"],
            ["C", "A", "B"],
            ["C", "A", "B"],
            ["B", "C", "A"],
        ]
    )
    cpp_fy_solver = pyspiel.sco.FenchelYoungOptimizer(
        profile.to_list_of_tuples(),
        rating_lower_bound=-100.0,
        rating_upper_bound=100.0,
        batch_size=4,
        rng_seed=SEED,
    )
    cpp_fy_solver.run_solver(10000, learning_rate=0.01)
    ratings_dict = cpp_fy_solver.ratings()
    # C is not necessarily better than A here, just like with Elo.
    # But both should have higher ratings than B.
    self.assertGreater(ratings_dict["C"], ratings_dict["B"])
    self.assertGreater(ratings_dict["A"], ratings_dict["B"])


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
