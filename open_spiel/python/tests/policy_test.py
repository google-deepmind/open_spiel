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

"""Tests for open_spiel.python.policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python import policy
from open_spiel.python.algorithms import get_all_states
import pyspiel

_TIC_TAC_TOE_STATES = [
    {
        # ...
        # xoo
        # ..x
        "state": "3, 4, 8, 5",
        "legal_actions": (0, 1, 2, 6, 7)
    },
    {
        # xo.
        # oxx
        # o..
        "state": "4, 1, 0, 3, 5, 6",
        "legal_actions": (2, 7, 8)
    },
    {
        # ...
        # ...
        # ...
        "state": "",
        "legal_actions": (0, 1, 2, 3, 4, 5, 6, 7, 8)
    }
]


def test_policy_on_game(self, game, policy_object):
  """Checks the policy conforms to the conventions.

  Checks the Policy.action_probabilities contains only legal actions (but not
  necessarily all).
  Checks that the probabilities are positive and sum to 1.

  Args:
    self: The Test class. This methid targets as being used as a utility
      function to test policies.
    game: A `pyspiel.Game`, same as the one used in the policy.
    policy_object: A `policy.Policy` object on `game`. to test.
  """

  all_states = get_all_states.get_all_states(
      game,
      depth_limit=-1,
      include_terminals=False,
      include_chance_states=False,
      to_string=lambda s: s.information_state_string())

  for state in all_states.values():
    legal_actions = set(state.legal_actions())
    action_probabilities = policy_object.action_probabilities(state)

    for action in action_probabilities.keys():
      # We want a clearer error message to be able to debug.
      actions_missing = set(legal_actions) - set(action_probabilities.keys())
      illegal_actions = set(action_probabilities.keys()) - set(legal_actions)
      self.assertIn(
          action,
          legal_actions,
          msg="The action {} is present in the policy but is not a legal "
          "actions (these are {})\n"
          "Legal actions missing from policy: {}\n"
          "Illegal actions present in policy: {}".format(
              action, legal_actions, actions_missing, illegal_actions))

    sum_ = 0
    for prob in action_probabilities.values():
      sum_ += prob
      self.assertGreaterEqual(prob, 0)
    self.assertAlmostEqual(1, sum_)


_LEDUC_POKER = pyspiel.load_game("leduc_poker")


class CommonTest(parameterized.TestCase):

  @parameterized.parameters([
      policy.TabularPolicy(_LEDUC_POKER),
      policy.UniformRandomPolicy(_LEDUC_POKER),
      policy.FirstActionPolicy(_LEDUC_POKER),
  ])
  def test_policy_on_leduc(self, policy_object):
    test_policy_on_game(self, _LEDUC_POKER, policy_object)

  @parameterized.named_parameters([
      ("pyspiel.UniformRandom", pyspiel.UniformRandomPolicy(_LEDUC_POKER)),
  ])
  def test_cpp_policies_on_leduc(self, policy_object):
    test_policy_on_game(self, _LEDUC_POKER, policy_object)


class TabularTicTacToePolicyTest(parameterized.TestCase):

  # Enumerating all the states for tic tac toe is quite slow, so we do this
  # ony once.
  @classmethod
  def setUpClass(cls):
    super(TabularTicTacToePolicyTest, cls).setUpClass()
    cls.game = pyspiel.load_game("tic_tac_toe")
    cls.tabular_policy = policy.TabularPolicy(cls.game)

  def test_policy_shape(self):
    # Tic tac toe has 4520 decision states; ref
    # https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.5b00324
    # There are 9 possible moves in the game (one per grid cell).
    # However, the TabularPolicy uses InformationState as keys, which in the
    # case of TicTacToe corresponds to the number of unique sequences (due to
    # perfect recall) requires by several algorithms, i.e. CFR.
    self.assertEqual(self.tabular_policy.action_probability_array.shape,
                     (294778, 9))

  def test_policy_attributes(self):
    # Verify the base class attributes of the policy
    self.assertEqual(self.tabular_policy.player_ids, [0, 1])

  @parameterized.parameters(*_TIC_TAC_TOE_STATES)
  def test_policy_at_state(self, state, legal_actions):
    index = self.tabular_policy.state_lookup[state]
    prob = 1 / len(legal_actions)
    np.testing.assert_array_equal(
        self.tabular_policy.action_probability_array[index],
        [prob if action in legal_actions else 0 for action in range(9)])

  @parameterized.parameters(*_TIC_TAC_TOE_STATES)
  def test_legal_actions_at_state(self, state, legal_actions):
    index = self.tabular_policy.state_lookup[state]
    np.testing.assert_array_equal(
        self.tabular_policy.legal_actions_mask[index],
        [1 if action in legal_actions else 0 for action in range(9)])

  def test_call_for_state(self):
    state = self.game.new_initial_state()
    state.apply_action(3)
    state.apply_action(4)
    state.apply_action(5)
    state.apply_action(6)
    state.apply_action(7)
    self.assertEqual(
        self.tabular_policy.action_probabilities(state), {
            0: 0.25,
            1: 0.25,
            2: 0.25,
            8: 0.25
        })

  def test_states_ordered_by_player(self):
    max_player0_index = max(
        self.tabular_policy.state_lookup[state]
        for state in self.tabular_policy.states_per_player[0])
    min_player1_index = min(
        self.tabular_policy.state_lookup[state]
        for state in self.tabular_policy.states_per_player[1])
    self.assertEqual(max_player0_index + 1, min_player1_index)

  def test_state_in(self):
    # Per state, we have 9 cells each with 3 possible states (o, x, empty)
    # Tic tac toe has 4520 decision states, but the tabular policy indexes by
    # InformationState, which leads to a larger number due to perfect recall
    self.assertEqual(self.tabular_policy.state_in.shape, (294778, 27))

  @parameterized.parameters(*_TIC_TAC_TOE_STATES)
  def test_policy_for_state_string(self, state, legal_actions):
    prob = 1 / len(legal_actions)
    np.testing.assert_array_equal(
        self.tabular_policy.policy_for_key(state),
        [prob if action in legal_actions else 0 for action in range(9)])


class TabularPolicyTest(parameterized.TestCase):

  def test_update_elementwise(self):
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)
    state = "0pb"
    np.testing.assert_array_equal(
        tabular_policy.policy_for_key(state), [0.5, 0.5])
    tabular_policy.policy_for_key(state)[0] = 0.9
    tabular_policy.policy_for_key(state)[1] = 0.1
    np.testing.assert_array_equal(
        tabular_policy.policy_for_key(state), [0.9, 0.1])

  def test_update_slice(self):
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)
    state = "2b"
    np.testing.assert_array_equal(
        tabular_policy.policy_for_key(state), [0.5, 0.5])
    tabular_policy.policy_for_key(state)[:] = [0.8, 0.2]
    np.testing.assert_array_equal(
        tabular_policy.policy_for_key(state), [0.8, 0.2])

  def test_state_ordering_is_deterministic(self):
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)

    expected = {
        "0": 0,
        "0pb": 1,
        "1": 2,
        "1pb": 3,
        "2": 4,
        "2pb": 5,
        "1p": 6,
        "1b": 7,
        "2p": 8,
        "2b": 9,
        "0p": 10,
        "0b": 11,
    }
    self.assertEqual(expected, tabular_policy.state_lookup)

  def test_states(self):
    game = pyspiel.load_game("leduc_poker")
    tabular_policy = policy.TabularPolicy(game)
    i = 0
    for state in tabular_policy.states:
      self.assertEqual(i, tabular_policy.state_index(state))
      i += 1

    self.assertEqual(936, i)

  @parameterized.parameters((policy.FirstActionPolicy, "kuhn_poker"),
                            (policy.UniformRandomPolicy, "kuhn_poker"),
                            (policy.FirstActionPolicy, "leduc_poker"),
                            (policy.UniformRandomPolicy, "leduc_poker"))
  def test_can_turn_policy_into_tabular_policy(self, policy_class, game_name):
    game = pyspiel.load_game(game_name)
    realized_policy = policy_class(game)
    tabular_policy = realized_policy.to_tabular()
    for state in tabular_policy.states:
      self.assertEqual(
          realized_policy.action_probabilities(state),
          tabular_policy.action_probabilities(state))


class TabularRockPaperScissorsPolicyTest(absltest.TestCase):

  # Enumerating all the states for rock-paper-scissors is fast, but
  # we initialize only once for consistency with slower games.
  @classmethod
  def setUpClass(cls):
    super(TabularRockPaperScissorsPolicyTest, cls).setUpClass()
    game = pyspiel.load_game_as_turn_based("matrix_rps")
    cls.tabular_policy = policy.TabularPolicy(game)

  def test_policy_attributes(self):
    # Verify the base class attributes of the policy
    self.assertEqual(self.tabular_policy.player_ids, [0, 1])

  def test_tabular_policy(self):
    # Test that the tabular policy is uniform random in each state.
    np.testing.assert_array_equal(
        self.tabular_policy.action_probability_array,
        [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])

  def test_states_lookup(self):
    # Test that there are two valid states, indexed as 0 and 1.
    game = pyspiel.load_game_as_turn_based("matrix_rps")
    state = game.new_initial_state()
    first_info_state = state.information_state_string()
    state.apply_action(state.legal_actions()[0])
    second_info_state = state.information_state_string()
    self.assertCountEqual(self.tabular_policy.state_lookup,
                          [first_info_state, second_info_state])
    self.assertCountEqual(self.tabular_policy.state_lookup.values(), [0, 1])

  def test_legal_actions_mask(self):
    # Test that all actions are valid in all states.
    np.testing.assert_array_equal(self.tabular_policy.legal_actions_mask,
                                  [[1, 1, 1], [1, 1, 1]])


class UniformRandomPolicyTest(absltest.TestCase):

  def test_policy_attributes(self):
    game = pyspiel.load_game("tiny_bridge_4p")
    uniform_random_policy = policy.UniformRandomPolicy(game)
    self.assertEqual(uniform_random_policy.player_ids, [0, 1, 2, 3])

  def test_policy_at_state(self):
    game = pyspiel.load_game("tic_tac_toe")
    uniform_random_policy = policy.UniformRandomPolicy(game)
    state = game.new_initial_state()
    state.apply_action(2)
    state.apply_action(4)
    state.apply_action(6)
    state.apply_action(8)
    self.assertEqual(
        uniform_random_policy.action_probabilities(state), {
            0: 0.2,
            1: 0.2,
            3: 0.2,
            5: 0.2,
            7: 0.2
        })

  def test_players_have_different_legal_actions(self):
    game = pyspiel.load_game("oshi_zumo")
    uniform_random_policy = policy.UniformRandomPolicy(game)
    state = game.new_initial_state()
    state.apply_actions([46, 49])
    # Started with 50 coins each, now have 4 and 1 respectively
    self.assertEqual(
        uniform_random_policy.action_probabilities(state, player_id=0), {
            0: 0.2,
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2
        })
    self.assertEqual(
        uniform_random_policy.action_probabilities(state, player_id=1), {
            0: 0.5,
            1: 0.5
        })


class MergeTabularPoliciesTest(absltest.TestCase):

  def test_identity(self):
    num_players = 2
    game = pyspiel.load_game("kuhn_poker", {"players": num_players})

    tabular_policies = [  # Policy limited to player.
        policy.TabularPolicy(game, players=(player,))
        for player in range(num_players)
    ]
    for player, tabular_policy in enumerate(tabular_policies):
      tabular_policy.action_probability_array[:] = 0
      tabular_policy.action_probability_array[:, player] = 1.0

    merged_tabular_policy = policy.merge_tabular_policies(
        tabular_policies, game)

    self.assertIdentityPoliciesEqual(tabular_policies, merged_tabular_policy,
                                     game)

  def test_identity_redundant(self):
    num_players = 2
    game = pyspiel.load_game("kuhn_poker", {"players": num_players})

    tabular_policies = [  # Policy for all players.
        policy.TabularPolicy(game, players=None)
        for player in range(num_players)
    ]
    for player, tabular_policy in enumerate(tabular_policies):
      tabular_policy.action_probability_array[:] = 0
      tabular_policy.action_probability_array[:, player] = 1.0

    merged_tabular_policy = policy.merge_tabular_policies(
        tabular_policies, game)

    self.assertIdentityPoliciesEqual(tabular_policies, merged_tabular_policy,
                                     game)

  def test_identity_missing(self):
    num_players = 2
    game = pyspiel.load_game("kuhn_poker", {"players": num_players})

    tabular_policies = [  # Only first player (repeated).
        policy.TabularPolicy(game, players=(0,))
        for player in range(num_players)
    ]
    for player, tabular_policy in enumerate(tabular_policies):
      tabular_policy.action_probability_array[:] = 0
      tabular_policy.action_probability_array[:, player] = 1.0

    merged_tabular_policy = policy.merge_tabular_policies(
        tabular_policies, game)

    for player in range(game.num_players()):
      if player == 0:
        self.assertListEqual(tabular_policies[player].states_per_player[player],
                             merged_tabular_policy.states_per_player[player])
        for p_state in merged_tabular_policy.states_per_player[player]:
          to_index = merged_tabular_policy.state_lookup[p_state]
          from_index = tabular_policies[player].state_lookup[p_state]
          self.assertTrue(
              np.allclose(
                  merged_tabular_policy.action_probability_array[to_index],
                  tabular_policies[player].action_probability_array[from_index])
          )

          self.assertTrue(
              np.allclose(
                  merged_tabular_policy.action_probability_array[to_index,
                                                                 player], 1))
      else:
        # Missing players have uniform policy.
        self.assertEmpty(tabular_policies[player].states_per_player[player])
        for p_state in merged_tabular_policy.states_per_player[player]:
          to_index = merged_tabular_policy.state_lookup[p_state]
          self.assertTrue(
              np.allclose(
                  merged_tabular_policy.action_probability_array[to_index,
                                                                 player], 0.5))

  def assertIdentityPoliciesEqual(self, tabular_policies, merged_tabular_policy,
                                  game):
    for player in range(game.num_players()):
      self.assertListEqual(tabular_policies[player].states_per_player[player],
                           merged_tabular_policy.states_per_player[player])

      for p_state in merged_tabular_policy.states_per_player[player]:
        to_index = merged_tabular_policy.state_lookup[p_state]
        from_index = tabular_policies[player].state_lookup[p_state]
        self.assertTrue(
            np.allclose(
                merged_tabular_policy.action_probability_array[to_index],
                tabular_policies[player].action_probability_array[from_index]))

        self.assertTrue(
            np.allclose(
                merged_tabular_policy.action_probability_array[to_index,
                                                               player], 1))


class JointActionProbTest(absltest.TestCase):

  def test_joint_action_probabilities(self):
    """Test expected behavior of joint_action_probabilities."""
    game = pyspiel.load_game("python_iterated_prisoners_dilemma")
    uniform_policy = policy.UniformRandomPolicy(game)
    joint_action_probs = policy.joint_action_probabilities(
        game.new_initial_state(), uniform_policy)
    self.assertCountEqual(
        list(joint_action_probs), [
            ((0, 0), 0.25),
            ((1, 1), 0.25),
            ((1, 0), 0.25),
            ((0, 1), 0.25),
        ])

  def test_joint_action_probabilities_failure_on_seq_game(self):
    """Test failure of child on sequential games."""
    game = pyspiel.load_game("kuhn_poker")
    with self.assertRaises(AssertionError):
      list(policy.joint_action_probabilities(
          game.new_initial_state(), policy.UniformRandomPolicy(game)))


class ChildTest(absltest.TestCase):

  def test_child_function_expected_behavior_for_seq_game(self):
    """Test expected behavior of child on sequential games."""
    game = pyspiel.load_game("tic_tac_toe")
    initial_state = game.new_initial_state()
    action = 3
    new_state = policy.child(initial_state, action)
    self.assertNotEqual(new_state.history(), initial_state.history())
    expected_new_state = initial_state.child(action)
    self.assertNotEqual(new_state, expected_new_state)
    self.assertEqual(new_state.history(), expected_new_state.history())

  def test_child_function_expected_behavior_for_sim_game(self):
    """Test expected behavior of child on simultaneous games."""
    game = pyspiel.load_game("python_iterated_prisoners_dilemma")
    parameter_state = game.new_initial_state()
    actions = [1, 1]
    new_state = policy.child(parameter_state, actions)
    self.assertEqual(str(new_state), ("p0:D p1:D"))

  def test_child_function_failure_behavior_for_sim_game(self):
    """Test failure behavior of child on simultaneous games."""
    game = pyspiel.load_game("python_iterated_prisoners_dilemma")
    parameter_state = game.new_initial_state()
    with self.assertRaises(AssertionError):
      policy.child(parameter_state, 0)


if __name__ == "__main__":
  absltest.main()
