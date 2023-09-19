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

"""Tests for open_spiel.python.algorithms.efr."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import efr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

_KUHN_GAME = pyspiel.load_game("kuhn_poker")
_LEDUC_GAME = pyspiel.load_game("leduc_poker")

_KUHN_UNIFORM_POLICY = policy.TabularPolicy(_KUHN_GAME)
_LEDUC_UNIFORM_POLICY = policy.TabularPolicy(_LEDUC_GAME)
_DEVIATIONS_ = ["blind action", "informed action", "blind cf", "informed cf", "bps", "cfps", "csps", "tips", "bhv"]

class EFRTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(_DEVIATIONS_)
  def test_policy_zero_is_uniform(self, deviations_name):
    # We use Leduc and not Kuhn, because Leduc has illegal actions and Kuhn does
    # not.
    game = pyspiel.load_game("leduc_poker")
    cfr_solver = efr.EFRSolver(
        game=game,
        deviations_name=deviations_name
        )
    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.current_policy().action_probability_array)
    np.testing.assert_array_equal(
        _LEDUC_UNIFORM_POLICY.action_probability_array,
        cfr_solver.average_policy().action_probability_array)
    
  @parameterized.parameters(
      ["blind cf", "informed cf", "bps", "cfps", "csps", "tips", "bhv"])
  def test_cfr_kuhn_poker(self, deviations_name):
    game = pyspiel.load_game("kuhn_poker")
    efr_solver = efr.EFRSolver(
        game=game,
        deviations_name=deviations_name
        )    
    for _ in range(300):
      efr_solver.evaluate_and_update_policy()
    average_policy = efr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)

if __name__ == "__main__":
  absltest.main()
