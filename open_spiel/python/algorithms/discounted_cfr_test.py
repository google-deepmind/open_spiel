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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.discounted_cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel


class DiscountedCfrTest(absltest.TestCase):

  def test_discounted_cfr_on_kuhn(self):
    game = pyspiel.load_game("kuhn_poker")
    solver = discounted_cfr.DCFRSolver(game)
    for _ in range(300):
      solver.evaluate_and_update_policy()
    average_policy = solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

  def test_discounted_cfr_runs_against_leduc(self):
    game = pyspiel.load_game("leduc_poker")
    solver = discounted_cfr.DCFRSolver(game)
    for _ in range(10):
      solver.evaluate_and_update_policy()
    solver.average_policy()


if __name__ == "__main__":
  absltest.main()
