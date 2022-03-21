# Copyright 2022 DeepMind Technologies Limited
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
"""Tests for factory."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.mfg.games import factory
import pyspiel


class FactoryTest(parameterized.TestCase):

  @parameterized.parameters(
      ("mfg_crowd_modelling_2d", None),
      ("mfg_crowd_modelling_2d", "crowd_modelling_2d_10x10"),
      ("mfg_crowd_modelling_2d", "crowd_modelling_2d_four_rooms"),
      ("python_mfg_dynamic_routing", None),
      ("python_mfg_dynamic_routing", "dynamic_routing_line"),
      ("python_mfg_dynamic_routing", "dynamic_routing_braess"),
      ("python_mfg_dynamic_routing",
       "dynamic_routing_sioux_falls_dummy_demand"),
      ("python_mfg_dynamic_routing", "dynamic_routing_sioux_falls"),
      ("python_mfg_predator_prey", None),
      ("python_mfg_predator_prey", "predator_prey_5x5x3"))
  def test_smoke(self, game_name, setting):
    game = factory.create_game_with_setting(game_name, setting)
    self.assertIsInstance(game, pyspiel.Game)


if __name__ == "__main__":
  absltest.main()
