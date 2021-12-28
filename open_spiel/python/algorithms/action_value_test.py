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

"""Tests for open_spiel.python.algorithms.action_value.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import action_value
import pyspiel


class ActionValuesTest(parameterized.TestCase):

  @parameterized.parameters([["kuhn_poker", 2], ["kuhn_poker", 3],
                             ["leduc_poker", 2]])
  def test_runs_with_uniform_policies(self, game_name, num_players):
    game = pyspiel.load_game(game_name, {"players": num_players})
    calc = action_value.TreeWalkCalculator(game)
    uniform_policy = policy.TabularPolicy(game)
    calc.compute_all_states_action_values([uniform_policy] * num_players)

  def test_kuhn_poker_always_pass_p0(self):
    game = pyspiel.load_game("kuhn_poker")
    calc = action_value.TreeWalkCalculator(game)
    uniform_policy = policy.TabularPolicy(game)
    always_pass_policy = policy.FirstActionPolicy(game).to_tabular()
    returned_values = calc([always_pass_policy, uniform_policy],
                           always_pass_policy)
    root_node_values = calc.get_root_node_values(
        [always_pass_policy, uniform_policy])
    self.assertTrue(
        np.allclose(root_node_values, returned_values.root_node_values))

    # Action 0 == Pass. Action 1 == Bet
    # Some values are 0 because the states are not reached, thus the expected
    # value of that node is undefined.
    np.testing.assert_array_almost_equal(
        np.asarray([
            # Player 0 states
            [-1.0, -0.5],    # '0'
            [-1.0, -2.0],    # '0pb'
            [-0.5, 0.5],     # '1'
            [-1.0, 0.0],     # '1pb'
            [0.0, 1.5],      # '2'
            [-1.0, 2.0],     # '2pb'
            # Player 1 states
            [0.0, 1.0],      # '1p'
            [0, 0],          # Unreachable
            [1.0, 1.0],      # '2p'
            [0, 0],          # Unreachable
            [-1.0, 1.0],     # '0p'
            [0, 0],          # Unreachable
        ]), returned_values.action_values)

    np.testing.assert_array_almost_equal(
        np.asarray([
            # Player 0 states
            1 / 3,  # '0'
            1 / 6,  # '0pb'
            1 / 3,  # '1'
            1 / 6,  # '1pb'
            1 / 3,  # '2'
            1 / 6,  # '2pb'
            # Player 1 states
            1 / 3,  # '1p'
            0.0,  # '1b': zero because player 0 always play pass
            1 / 3,  # 2p'
            0.0,  # '2b': zero because player 0 always play pass
            1 / 3,  # '0p'
            0.0,  # '0b':  zero because player 0 always play pass
        ]),
        returned_values.counterfactual_reach_probs)

    # The reach probabilities are always one, even though we have player 0
    # who only plays pass, because the unreachable nodes for player 0 are
    # terminal nodes: e.g.  'x x b b p' has a player 0 reach of 0, but it is
    # a terminal node, thus it does not appear in the tabular policy
    # states.
    np.testing.assert_array_equal(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        returned_values.player_reach_probs)

    np.testing.assert_array_almost_equal(
        np.asarray([
            np.array([-1/3, -1/6]),
            np.array([-1/6, -1/3]),
            np.array([-1/6, 1/6]),
            np.array([-1/6, 0.]),
            np.array([0., 0.5]),
            np.array([-1/6, 1/3]),
            np.array([0., 1/3]),
            np.array([0., 0.]),
            np.array([1/3, 1/3]),
            np.array([0., 0.]),
            np.array([-1/3, 1/3]),
            np.array([0., 0.])
        ]), returned_values.sum_cfr_reach_by_action_value)


if __name__ == "__main__":
  absltest.main()
