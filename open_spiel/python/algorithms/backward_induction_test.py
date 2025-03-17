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

"""Tests for open_spiel.python.algorithms.backward_induction."""

from absl.testing import absltest

import pyspiel
from open_spiel.python.algorithms import backward_induction


class BackwardInductionTest(absltest.TestCase):

  def test_tic_tac_toe_draw(self):
    """Tests that backward induction finds that tic-tac-toe is a draw."""
    game = pyspiel.load_game("tic_tac_toe")
    values, policy = backward_induction.backward_induction(game)
    self.assertEqual(values[0], 0.0)  # Draw
    self.assertEqual(values[1], 0.0)  # Draw

  def test_tic_tac_toe_win(self):
    """Tests backward induction in a position with a winning strategy."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    
    # Create board:
    # X . .
    # X O .
    # O . .
    state.apply_action(0)  # X in top-left
    state.apply_action(4)  # O in middle
    state.apply_action(3)  # X in middle-left
    state.apply_action(6)  # O in bottom-left
    
    # X to play, should find the winning move
    values, policy = backward_induction.backward_induction(game, state)
    self.assertEqual(values[0], 1.0)  # X wins
    self.assertEqual(values[1], -1.0)  # O loses
    
    # Best action should be position 2 (top-right)
    self.assertEqual(policy[state.to_string()], 2)
    
    # Try the helper function that returns just the values
    values_only = backward_induction.backward_induction_values(game, state)
    self.assertEqual(values_only[0], 1.0)  # X wins
    self.assertEqual(values_only[1], -1.0)  # O loses

  def test_tic_tac_toe_all_optimal_actions(self):
    """Tests finding all optimal actions in tic-tac-toe."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    
    # Get all optimal actions from the initial state
    values, all_optimal_actions = backward_induction.backward_induction_all_optimal_actions(
        game, state)
    
    # With optimal play, tic-tac-toe is a draw
    self.assertEqual(values[0], 0.0)
    self.assertEqual(values[1], 0.0)
    
    # From the initial state, the center and corners should be optimal
    # (there should be multiple optimal first moves)
    initial_actions = all_optimal_actions[state.to_string()]
    self.assertGreater(len(initial_actions), 0)
    
    # Test tie-breaking policies
    values_first, policy_first = backward_induction.backward_induction(
        game, state, tie_breaking_policy=backward_induction.TieBreakingPolicy.FIRST_ACTION)
    values_last, policy_last = backward_induction.backward_induction(
        game, state, tie_breaking_policy=backward_induction.TieBreakingPolicy.LAST_ACTION)
    
    # The values should be the same regardless of tie-breaking
    self.assertEqual(values_first[0], 0.0)
    self.assertEqual(values_first[1], 0.0)
    self.assertEqual(values_last[0], 0.0)
    self.assertEqual(values_last[1], 0.0)
    
    # But the chosen actions might be different if there are multiple optimal actions
    if len(initial_actions) > 1:
      first_action = policy_first[state.to_string()]
      last_action = policy_last[state.to_string()]
      # Check that both actions are in the set of optimal actions
      self.assertIn(first_action, initial_actions)
      self.assertIn(last_action, initial_actions)

  def test_sequential_goofspiel(self):
    """Tests backward induction on a small sequential-play Goofspiel game."""
    # Create a small sequential version of Goofspiel
    game = pyspiel.load_game(
        "goofspiel",
        {
            "num_cards": 3,
            "points_order": "descending",
            "returns_type": "win_loss",
            "players": 2,
            "imp_info": False,  # Perfect information
        })
    
    self.assertEqual(game.get_type().information,
                      pyspiel.GameType.Information.PERFECT_INFORMATION)
    
    values, policy = backward_induction.backward_induction(game)
    
    # With optimal play, one player wins or it's a draw
    self.assertGreaterEqual(values[0], -1.0)
    self.assertLessEqual(values[0], 1.0)
    self.assertGreaterEqual(values[1], -1.0)
    self.assertLessEqual(values[1], 1.0)
    
    # Values should be consistent with zero-sum property
    self.assertEqual(values[0], -values[1])
    
    # Check that policy is not empty
    self.assertGreater(len(policy), 0)

  def test_imperfect_info_warning(self):
    """Test backward induction's warning for imperfect information games."""
    # Create a kuhn poker game (which has imperfect information)
    game = pyspiel.load_game("kuhn_poker")
    self.assertEqual(game.get_type().information, 
                     pyspiel.GameType.Information.IMPERFECT_INFORMATION)
    
    # Running backward induction without allowing imperfect info should raise an error
    with self.assertRaises(RuntimeError):
      backward_induction.backward_induction(
          game, allow_imperfect_information=False)
    
    # But we can override this check
    values, _ = backward_induction.backward_induction(
        game, allow_imperfect_information=True)
    # We don't check the actual values, just that it returns something
    self.assertEqual(len(values), game.num_players())


if __name__ == "__main__":
  absltest.main() 