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
"""Tests for Fixed Point."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.mfg.algorithms import fixed_point
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
import pyspiel


class FixedPointTest(parameterized.TestCase):

  @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'),
                                  ('cpp', 'mfg_crowd_modelling'))
  def test_run(self, name):
    """Checks if the algorithm works."""
    game = pyspiel.load_game(name)
    fixed_p = fixed_point.FixedPoint(game)

    for _ in range(10):
      fixed_p.iteration()

    fixed_p_policy = fixed_p.get_policy()
    nash_conv_fixed_p = nash_conv.NashConv(game, fixed_p_policy)

    self.assertAlmostEqual(nash_conv_fixed_p.nash_conv(), 55.745, places=3)

  @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'),
                                  ('cpp', 'mfg_crowd_modelling'))
  def test_softmax(self, name):
    """Checks the softmax policy."""
    game = pyspiel.load_game(name)
    fixed_p = fixed_point.FixedPoint(game, temperature=10.0)

    for _ in range(10):
      fixed_p.iteration()

    fixed_p_policy = fixed_p.get_policy()
    nash_conv_fixed_p = nash_conv.NashConv(game, fixed_p_policy)

    self.assertAlmostEqual(nash_conv_fixed_p.nash_conv(), 2.421, places=3)


if __name__ == '__main__':
  absltest.main()
