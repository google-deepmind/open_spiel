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


def _uniform_policy(state):
  actions = state.legal_actions()
  p = 1.0 / len(actions)
  return [(a, p) for a in actions]


class ActionValuesTest(parameterized.TestCase):

  @parameterized.parameters(["kuhn_poker", "leduc_poker"])
  def test_runs_with_uniform_policies(self, game_name):
    game = pyspiel.load_game(game_name)
    calc = action_value.TreeWalkCalculator(game)

    calc.compute_all_states_action_values([
        policy.PolicyFromCallable(game, _uniform_policy),
        policy.PolicyFromCallable(game, _uniform_policy)
    ])

  def test_kuhn_poker_always_pass_p0(self):
    game = pyspiel.load_game("kuhn_poker")
    calc = action_value.TreeWalkCalculator(game)

    for always_pass_policy in [
        lambda state: [(0, 1.0), (1, 0.0)],
        # On purpose, we use a policy that do not list all the legal actions.
        lambda state: [(0, 1.0), (1, 0.0)],
    ]:
      tabular_policy = policy.tabular_policy_from_policy(
          game, policy.PolicyFromCallable(game, always_pass_policy))

      # States are ordered using tabular_policy.states_per_player:
      # ['0', '0pb', '1', '1pb', '2', '2pb'] +
      # ['1p', '1b', '2p', '2b', '0p', '0b']
      np.testing.assert_array_equal(
          np.asarray([
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
              [1., 0.],
          ]), tabular_policy.action_probability_array)

      returned_values = calc([
          policy.PolicyFromCallable(game, always_pass_policy),
          policy.PolicyFromCallable(game, _uniform_policy)
      ], tabular_policy)

      # Action 0 == Pass. Action 1 == Bet
      # Some values are 0 because the states are not reached, thus the expected
      # value of that node is undefined.
      np.testing.assert_array_almost_equal(
          np.asarray([
              [-1.0, -0.5],
              [-1.0, -2.0],
              [-0.5, 0.5],
              [-1.0, 0.0],
              [0.0, 1.5],
              [-1.0, 2.0],
              [0.0, 1.0],
              [0, 0],
              [1.0, 1.0],
              [0, 0],
              [-1.0, 1.0],
              [0, 0],
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
