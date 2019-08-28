# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the C++ matrix game utility methods exposed to Python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from open_spiel.python.algorithms import lp_solver
import pyspiel


class MatrixGamesUtilsTest(unittest.TestCase):

  def test_extensive_to_matrix_game(self):
    kuhn_game = pyspiel.load_game("kuhn_poker")
    kuhn_matrix_game = pyspiel.extensive_to_matrix_game(kuhn_game)
    unused_p0_strategy, unused_p1_strategy, p0_sol_val, p1_sol_val = (
        lp_solver.solve_zero_sum_matrix_game(kuhn_matrix_game))
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(p0_sol_val, -1 / 18)
    self.assertAlmostEqual(p1_sol_val, +1 / 18)


if __name__ == "__main__":
  unittest.main()
