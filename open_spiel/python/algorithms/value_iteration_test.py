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

"""Tests for open_spiel.python.algorithms.get_all_states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from open_spiel.python.algorithms import value_iteration
import pyspiel


class ValueIterationTest(unittest.TestCase):

  def test_tic_tac_toe_number_states(self):
    game = pyspiel.load_game("tic_tac_toe")
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=0.01)

    initial_state = "...\n...\n..."
    cross_win_state = "...\n...\n.ox"
    naught_win_state = "x..\noo.\nxx."
    self.assertEqual(values[initial_state], 0)
    self.assertEqual(values[cross_win_state], 1)
    self.assertEqual(values[naught_win_state], -1)


if __name__ == "__main__":
  unittest.main()
