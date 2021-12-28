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

"""Tests for open_spiel.python.algorithms.projected_replicator_dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import projected_replicator_dynamics


class ProjectedReplicatorDynamicsTest(absltest.TestCase):

  def test_two_players(self):
    test_a = np.array([[2, 1, 0], [0, -1, -2]])
    test_b = np.array([[2, 1, 0], [0, -1, -2]])

    strategies = projected_replicator_dynamics.projected_replicator_dynamics(
        [test_a, test_b],
        prd_initial_strategies=None,
        prd_iterations=50000,
        prd_dt=1e-3,
        prd_gamma=1e-8,
        average_over_last_n_strategies=10)

    self.assertLen(strategies, 2, "Wrong strategy length.")
    self.assertGreater(strategies[0][0], 0.999,
                       "Projected Replicator Dynamics failed in trivial case.")

  def test_three_players(self):
    test_a = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    test_b = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    test_c = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])

    strategies = projected_replicator_dynamics.projected_replicator_dynamics(
        [test_a, test_b, test_c],
        prd_initial_strategies=None,
        prd_iterations=50000,
        prd_dt=1e-3,
        prd_gamma=1e-6,
        average_over_last_n_strategies=10)
    self.assertLen(strategies, 3, "Wrong strategy length.")
    self.assertGreater(strategies[0][0], 0.999,
                       "Projected Replicator Dynamics failed in trivial case.")


if __name__ == "__main__":
  absltest.main()
