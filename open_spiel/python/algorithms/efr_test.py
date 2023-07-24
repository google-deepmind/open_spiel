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


class ModuleLevelFunctionTest(absltest.TestCase):

  def test__update_current_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)

    cumulative_regrets = np.arange(0, 12 * 2).reshape((12, 2))
    expected_policy = cumulative_regrets / np.sum(
        cumulative_regrets, axis=-1, keepdims=True)
    nodes_indices = {
        u"0": 0,
        u"0pb": 1,
        u"1": 2,
        u"1pb": 3,
        u"2": 4,
        u"2pb": 5,
        u"1p": 6,
        u"1b": 7,
        u"2p": 8,
        u"2b": 9,
        u"0p": 10,
        u"0b": 11,
    }
    # pylint: disable=g-complex-comprehension
    info_state_nodes = {
        key: efr._InfoStateNode(
            legal_actions=[0, 1],
            index_in_tabular_policy=None,
            cumulative_regret=dict(enumerate(cumulative_regrets[index])),
            cumulative_policy=None) for key, index in nodes_indices.items()
    }
    available_deviations = ["blind action", "informed action", "blind cf", "informed cf", "bps", "cfps", "csps", "tips", "bhv"]

    # pylint: enable=g-complex-comprehension

    efr._update_current_policy(tabular_policy, info_state_nodes)

    np.testing.assert_array_equal(expected_policy,
                                  tabular_policy.action_probability_array)


class EFRTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      ["blind action", "informed action", "blind cf", "informed cf", "bps", "cfps", "csps", "tips", "bhv"])
  def test_policy_zero_is_uniform(self):
    # We use Leduc and not Kuhn, because Leduc has illegal actions and Kuhn does
    # not.
    game = pyspiel.load_game("leduc_poker")
    cfr_solver = efr._EFRSolver(
        game,
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
  def test_cfr_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    efr_solver = efr.EFRSolver(game)
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
