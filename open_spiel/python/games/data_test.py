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

"""Tests for open_spiel.python.games.data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.algorithms import exploitability
from open_spiel.python.games import data
import pyspiel


class NashEquilibriumtest(parameterized.TestCase):

  @parameterized.parameters((0.), (0.1), (1 / 3))
  def test_exploitability_is_zero_on_nash(self, alpha):
    # A similar test exists in:
    # open_spiel/python/algorithms/exploitability_test.py
    game = pyspiel.load_game("kuhn_poker")
    policy = data.kuhn_nash_equilibrium(alpha=alpha)
    expl = exploitability.exploitability(game, policy)
    self.assertAlmostEqual(0, expl)


if __name__ == "__main__":
  absltest.main()
