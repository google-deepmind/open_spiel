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

"""Tests for distribution."""

from absl.testing import absltest
from open_spiel.python import policy
from open_spiel.python.mfg import games  # pylint: disable=unused-import
from open_spiel.python.mfg.algorithms import distribution
import pyspiel


class DistributionTest(absltest.TestCase):

  def test_basic(self):
    game = pyspiel.load_game("python_mfg_crowd_modelling")
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    state = game.new_initial_state().child(0)
    self.assertAlmostEqual(dist.value(state), 1 / game.size)

  def test_state_support_outside_distrib(self):
    game = pyspiel.load_game("mfg_crowd_modelling_2d", {
        "initial_distribution": "[0|0]",
        "initial_distribution_value": "[1.]",
    })
    uniform_policy = policy.UniformRandomPolicy(game)
    _ = distribution.DistributionPolicy(game, uniform_policy)

  def test_multi_pop(self):
    game = pyspiel.load_game("python_mfg_predator_prey")
    self.assertEqual(game.num_players(), 3)
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    for pop in range(3):
      self.assertAlmostEqual(
          dist.value(game.new_initial_state_for_population(pop)), 1.)


if __name__ == "__main__":
  absltest.main()
