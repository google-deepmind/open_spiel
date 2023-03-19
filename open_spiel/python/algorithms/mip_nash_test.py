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
"""Tests for open_spiel.python.algorithms.mip_nash."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python.algorithms.mip_nash import mip_nash
import pyspiel


# prisoners' dilemma
pd_game = pyspiel.create_matrix_game(
    [[-2.0, -10.0], [0.0, -5.0]],
    [[-2.0, 0.0], [-10.0, -5.0]])

pd_eq = (np.array([0, 1]), np.array([0, 1]))

# stag hunt
sh_game = pyspiel.create_matrix_game(
    [[10.0, 1.0], [8.0, 5.0]],
    [[10.0, 8.0], [1.0, 5.0]])


sh_eq = (np.array([1, 0]), np.array([1, 0]))

class MIPNash(parameterized.TestCase):
  @parameterized.named_parameters(
      ("pd", pd_game, pd_eq),
      ("sh", sh_game, sh_eq),
  )
  def test_simple_games(self, game, eq):
    computed_eq = mip_nash(game, objective='MAX_SOCIAL_WELFARE')
    with self.subTest("probability"):
      np.testing.assert_array_almost_equal(computed_eq[0], eq[0])
      np.testing.assert_array_almost_equal(computed_eq[1], eq[1])


if __name__ == "__main__":
  absltest.main()