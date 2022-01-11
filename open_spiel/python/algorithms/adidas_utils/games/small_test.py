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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.adidas_utils.games.small."""

from absl import logging  # pylint:disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python.algorithms.adidas_utils.games import small
from open_spiel.python.algorithms.adidas_utils.helpers import simplex


class SmallTest(parameterized.TestCase):

  def test_biased_game(self, trials=100, atol=1e-5, rtol=1e-5, seed=1234):
    """Test best responses to sampled opp. actions in BiasedGame are biased."""
    game = small.BiasedGame(seed)
    random = np.random.RandomState(seed)

    successes = []
    for _ in range(trials):
      dirichlet_alpha = np.ones(game.num_strategies()[0])
      dist = random.dirichlet(dirichlet_alpha)  # mixed srategy

      sample_best_responses = np.argmax(game.payoff_tensor()[0], axis=0)
      estimated_best_response = np.dot(sample_best_responses, dist)

      true_best_response = game.best_response(dist)

      successes += [not np.allclose(estimated_best_response, true_best_response,
                                    rtol, atol)]

    perc = 100 * np.mean(successes)
    logging.info('bias rate out of %d is %f', trials, perc)
    self.assertGreaterEqual(
        perc, 99., 'best responses should be biased more often')

  @staticmethod
  def simp_to_euc(a, b, center):
    r"""Transforms a point [a, b] on the simplex to Euclidean space.

      /\   ^ b
     /  \  |
    /____\ --> a

    Args:
      a: horizonal deviation from center
      b: vertical deviation from center
      center: center of ref frame given in [x, y, z] Euclidean coordinates
    Returns:
      1-d np.array of len 3, i.e., np.array([x, y, z])
    """
    transform = np.array([[.5, -.5, 0], [-.5, -.5, 1], [1, 1, 1]]).T
    transform /= np.linalg.norm(transform, axis=0)
    return transform.dot(np.array([a, b, 0])) + center

  @parameterized.named_parameters(
      ('up_down', 0., 0.1, 0., -0.1, -1.),
      ('left_right', -0.1, 0., 0.1, 0., -1.),
      ('up_left', 0., 0.1, -0.1, 0., 0.),
      ('up_right', 0., 0.1, 0.1, 0., 0.),
      ('down_left', 0., -0.1, -0.1, 0., 0.),
      ('down_right', 0., -0.1, 0.1, 0., 0.),
      )
  def test_spiral_game(self, dx_1, dy_1, dx_2, dy_2, expected_cos_sim,
                       trials=100, eps=0.1, seed=1234):
    """Test that gradients on simplex rotate around SpiralGame's center."""
    random = np.random.RandomState(seed)

    successes = []
    for _ in range(trials):
      dx, dy = eps * (random.rand(2) * 2 - 1)
      center = self.simp_to_euc(dx, dy, np.ones(3) / 3.)
      game = small.SpiralGame(center, seed)
      pt = game.payoff_tensor()[0]

      point_1 = self.simp_to_euc(dx_1, dy_1, game.center)
      point_2 = self.simp_to_euc(dx_2, dy_2, game.center)

      grad_1 = simplex.project_grad(pt.dot(point_1))
      grad_2 = simplex.project_grad(pt.dot(point_2))
      norm = np.linalg.norm(grad_1) * np.linalg.norm(grad_2)
      cos_sim = grad_1.dot(grad_2) / norm

      successes += [(np.abs(cos_sim - expected_cos_sim) < 1e-5)]

    perc = 100 * np.mean(successes)
    logging.info('alignment success rate out of %d is %f', trials, perc)
    self.assertGreaterEqual(
        perc, 99., 'gradient field should exhibit cycles')


if __name__ == '__main__':
  absltest.main()
