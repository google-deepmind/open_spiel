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
"""Tests for open_spiel.python.egt.nash_averaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms.nash_averaging import nash_averaging
import pyspiel


class NashAveragingTest(absltest.TestCase):

  def test_simple_game(self):
    game = pyspiel.create_matrix_game(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
        [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])

    maxent_nash, nash_avg_value = nash_averaging(game)
    with self.subTest("probability"):
      self.assertSeqeunceAlmostEqual((1/3, 1/3, 1/3), maxent_nash)

    with self.subTest("value"):
      self.assertSeqeunceAlmostEqual((0, 0, 0), nash_avg_value)


if __name__ == "__main__":
  absltest.main()
