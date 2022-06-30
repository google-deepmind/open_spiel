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
"""Tests for crowd_modelling_2d."""

from absl.testing import absltest

from open_spiel.python.mfg.games import crowd_modelling_2d


class CrowdModelling2DTest(absltest.TestCase):

  def test_grid_to_forbidden_states(self):
    forbidden_states = crowd_modelling_2d.grid_to_forbidden_states([
        "#####",
        "# # #",
        "#   #",
        "#####",
    ])

    self.assertEqual(
        forbidden_states,
        "[0|0;1|0;2|0;3|0;4|0;0|1;2|1;4|1;0|2;4|2;0|3;1|3;2|3;3|3;4|3]")


if __name__ == "__main__":
  absltest.main()
