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

"""Tests for the Ant Foraging game."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import ant_foraging
import pyspiel


class AntForagingTest(absltest.TestCase):

  def test_game_registered(self):
    """Test that the game is properly registered."""
    game = pyspiel.load_game("python_ant_foraging")
    self.assertEqual(game.get_type().short_name, "python_ant_foraging")

  def test_default_parameters(self):
    """Test default game parameters."""
    game = pyspiel.load_game("python_ant_foraging")
    self.assertEqual(game.num_players(), 2)
    self.assertEqual(game.grid_size, 8)
    self.assertEqual(game.num_food, 3)
    self.assertEqual(game.max_turns, 50)

  def test_initial_state(self):
    """Test the initial game state."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    self.assertEqual(state.current_player(), 0)
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.returns(), [0.0, 0.0])

  def test_legal_actions(self):
    """Test that legal actions are correctly computed."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    legal = state.legal_actions()
    # Should have at least STAY action
    self.assertIn(ant_foraging.Action.STAY, legal)
    # Should have at most 5 actions (stay + 4 directions)
    self.assertLessEqual(len(legal), 5)

  def test_apply_action(self):
    """Test applying an action changes the state."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    initial_player = state.current_player()
    legal = state.legal_actions()
    state.apply_action(legal[0])

    # Player should change after applying action
    self.assertNotEqual(state.current_player(), initial_player)

  def test_turn_progression(self):
    """Test that turns progress correctly with multiple ants."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    # First ant moves
    self.assertEqual(state.current_player(), 0)
    state.apply_action(state.legal_actions()[0])

    # Second ant's turn
    self.assertEqual(state.current_player(), 1)
    state.apply_action(state.legal_actions()[0])

    # Back to first ant
    self.assertEqual(state.current_player(), 0)

  def test_state_string(self):
    """Test that state can be converted to string."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    state_str = str(state)
    self.assertIsInstance(state_str, str)
    self.assertIn("Turn", state_str)
    self.assertIn("Food", state_str)

  def test_game_terminates(self):
    """Test that game eventually terminates."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    # Play until terminal
    step_count = 0
    while not state.is_terminal() and step_count < 500:
      legal = state.legal_actions()
      state.apply_action(legal[0])
      step_count += 1

    self.assertTrue(state.is_terminal())

  def test_random_simulation(self):
    """Run random simulations to check for crashes."""
    game = pyspiel.load_game("python_ant_foraging")

    for _ in range(3):
      state = game.new_initial_state()
      while not state.is_terminal():
        legal = state.legal_actions()
        action = np.random.choice(legal)
        state.apply_action(action)

      # Game should complete without error
      returns = state.returns()
      self.assertLen(returns, game.num_players())

  def test_action_to_string(self):
    """Test action to string conversion."""
    game = pyspiel.load_game("python_ant_foraging")
    state = game.new_initial_state()

    action_str = state.action_to_string(0, ant_foraging.Action.UP)
    self.assertIn("up", action_str)


if __name__ == "__main__":
  absltest.main()
