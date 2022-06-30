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

"""Tests for open_spiel.python.algorithms.adidas_utils.helpers.simplex."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import simplex


class SimplexTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('inside', np.array([.25, .75]), np.array([.25, .75])),
      ('outside_1', np.ones(2), 0.5 * np.ones(2)),
      ('outside_2', np.array([2., 0.]), np.array([1., 0.])),
      ('outside_3', np.array([.25, .25]), np.array([.5, .5])),
      )
  def test_euclidean_projection(self, vector, expected_projection):
    projection = simplex.euclidean_projection_onto_simplex(vector, subset=False)
    self.assertListEqual(list(projection), list(expected_projection),
                         msg='projection not accurate')

  @parameterized.named_parameters(
      ('orth', np.array([.75, .75]), np.array([.0, .0])),
      ('oblique', np.array([1., .5]), np.array([.25, -.25])),
      ('tangent', np.array([.25, .25, -.5]), np.array([.25, .25, -.5])),
      )
  def test_tangent_projection(self, vector, expected_projection):
    projection = simplex.project_grad(vector)
    self.assertListEqual(list(projection), list(expected_projection),
                         msg='projection not accurate')

  @parameterized.named_parameters(
      ('orth_1', np.array([0.5, 0.5]), np.array([.75, .75]), 0.),
      ('orth_2', np.array([1., 0.]), np.array([.75, .75]), 0.),
      ('tangent_1', np.array([1., 0.]), np.array([-.5, .5]), 0.),
      ('tangent_2', np.array([1., 0.]), np.array([1., -1.]), np.sqrt(2)),
      )
  def test_grad_norm(self, dist, grad, expected_norm):
    norm = simplex.grad_norm(dist, grad)
    self.assertAlmostEqual(norm, expected_norm, msg='norm not accurate')


if __name__ == '__main__':
  absltest.main()
