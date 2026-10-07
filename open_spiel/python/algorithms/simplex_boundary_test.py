# Copyright 2019 DeepMind Technologies Limited
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

"""Projection at the maximal feasible probability floor has one solution."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.algorithms import projected_replicator_dynamics as prd


class SimplexBoundaryTest(parameterized.TestCase):

  @parameterized.product(count=[2, 3, 4, 7], dtype=[np.float32, np.float64])
  def test_maximum_floor_returns_the_unique_uniform_distribution(
      self, count, dtype
  ):
    values = np.linspace(0.0, 1.0, count, dtype=dtype)
    original = values.copy()
    result = prd._simplex_projection(values, gamma=1 / count)
    np.testing.assert_allclose(
        result, np.full(count, 1 / count), rtol=1e-12, atol=0
    )
    self.assertAlmostEqual(float(result.sum()), 1.0)
    np.testing.assert_array_equal(values, original)

  @parameterized.parameters(0.51, 1.0, 2.0)
  def test_infeasible_probability_floor_is_rejected(self, gamma):
    with self.assertRaisesRegex(ValueError, 'gamma'):
      prd._simplex_projection(np.array([0.0, 1.0]), gamma)

  @parameterized.parameters(1, 3, 8)
  def test_public_replicator_dynamics_keeps_boundary_strategies_normalized(
      self, iterations
  ):
    matrix = np.array([[4.0, 2.0], [0.0, -1.0]])
    result = prd.projected_replicator_dynamics(
        [matrix, -matrix], prd_iterations=iterations, prd_dt=0.2, prd_gamma=0.5
    )
    for strategy in result:
      np.testing.assert_allclose(strategy, [0.5, 0.5], atol=1e-14)

  @parameterized.parameters(
      {'values': [0.4, 0.7, 0.6], 'gamma': 0.2, 'expected': [0.2, 0.45, 0.35]},
      {'values': [-1.0, 1.0], 'gamma': 0.0, 'expected': [0.0, 1.0]},
      {'values': [0.5, 0.5], 'gamma': 0.49, 'expected': [0.5, 0.5]},
      {'values': [100.0], 'gamma': 1.0, 'expected': [1.0]},
  )
  def test_interior_and_single_action_controls(self, values, gamma, expected):
    np.testing.assert_allclose(
        prd._simplex_projection(np.array(values), gamma), expected
    )


if __name__ == '__main__':
  absltest.main()
