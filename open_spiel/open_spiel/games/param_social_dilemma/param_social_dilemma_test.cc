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

"""Tests for parameterized social dilemma game (C++ implementation)."""

import unittest

import pyspiel

from open_spiel.python.games import param_social_dilemma


class CppParamSocialDilemmaTest(unittest.TestCase):
    """Test cases for C++ parameterized social dilemma game."""

    def test_cpp_game_creation(self):
        """Test C++ game creation with different parameters."""
        # Test 2-player game
        game = pyspiel.load_game("param_social_dilemma", 
                                         {"num_players": 2, "num_actions": 2})
        self.assertEqual(game.get_type().short_name, "param_social_dilemma")
        self.assertEqual(game.num_players(), 2)
        self.assertEqual(game.num_distinct_actions(), 2)
        
        # Test 3-player game
        game = pyspiel.load_game("param_social_dilemma", 
                                         {"num_players": 3, "num_actions": 2})
        self.assertEqual(game.num_players(), 3)
        self.assertEqual(game.num_distinct_actions(), 2)

    def test_cpp_vs_python_equivalence(self):
        """Test that C++ and Python implementations produce equivalent results."""
        # Create equivalent games
        cpp_game = pyspiel.load_game("param_social_dilemma", 
                                         {"num_players": 2, "num_actions": 2})
        py_game = pyspiel.load_game("python_param_social_dilemma", 
                                        {"num_players": 2, "num_actions": 2})
        
        # Test same actions produce same results
        cpp_state = cpp_game.new_initial_state()
        py_state = py_game.new_initial_state()
        
        # Apply cooperate-cooperate
        cpp_state.apply_actions([0, 0])
        py_state.apply_actions([0, 0])
        
        cpp_rewards = cpp_state.rewards()
        py_rewards = py_state.rewards()
        
        # Should be equivalent (allowing for minor floating point differences)
        for i in range(2):
            self.assertAlmostEqual(cpp_rewards[i], py_rewards[i], places=5)

    def test_cpp_reward_noise(self):
        """Test C++ reward noise functionality."""
        # Test with Gaussian noise
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "reward_noise_std": 0.1,
            "reward_noise_type": "gaussian",
            "seed": 42
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        rewards = state.rewards()
        
        # Rewards should be noisy (not exactly 3)
        self.assertNotEqual(rewards[0], 3.0)
        self.assertNotEqual(rewards[1], 3.0)
        # But should be close
        self.assertAlmostEqual(rewards[0], 3.0, delta=0.5)
        self.assertAlmostEqual(rewards[1], 3.0, delta=0.5)

    def test_cpp_termination(self):
        """Test C++ game termination mechanics."""
        # Create game with deterministic termination
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "termination_probability": 1.0  # Always terminate
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        
        # Should be chance node
        self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
        
        # Apply stop
        state.apply_action(1)  # STOP
        self.assertTrue(state.is_terminal())

    def test_cpp_custom_payoffs(self):
        """Test C++ custom payoff matrix."""
        custom_payoff = [10, 0, 20, 5, 10, 20, 0, 5]  # Coordination game
        
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "payoff_matrix": custom_payoff
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])  # Both cooperate
        rewards = state.rewards()
        
        self.assertEqual(rewards[0], 10)
        self.assertEqual(rewards[1], 10)

    def test_cpp_three_player(self):
        """Test C++ 3-player game functionality."""
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 3,
            "num_actions": 2,
            "termination_probability": 0.0  # Never terminate for testing
        })
        
        state = game.new_initial_state()
        
        # All cooperate
        state.apply_actions([0, 0, 0])
        rewards = state.rewards()
        self.assertEqual(len(rewards), 3)
        
        # All should get positive rewards for cooperation
        for reward in rewards:
            self.assertGreater(reward, 0)

    def test_cpp_action_history(self):
        """Test C++ action history tracking."""
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "termination_probability": 0.0
        })
        
        state = game.new_initial_state()
        
        # Apply some actions
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        state.apply_actions([1, 0])
        
        # Check history string
        history_str = state.to_string()
        self.assertIn("p0:", history_str)
        self.assertIn("p1:", history_str)
        self.assertIn("CD", history_str)  # Player 0: Cooperate, Defect
        self.assertIn("DC", history_str)  # Player 1: Defect, Cooperate

    def test_cpp_observation_tensor(self):
        """Test C++ observation tensor creation."""
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "max_game_length": 10
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        
        # Get observation tensor
        tensor = state.observation_tensor(0)
        expected_size = 10 * 2 + 1  # max_length * num_players + iteration
        self.assertEqual(len(tensor), expected_size)
        
        # Should have action history encoded
        self.assertEqual(tensor[0], 0)  # Player 0's first action
        self.assertEqual(tensor[10], 1)  # Player 1's first action

    def test_cpp_parameter_validation(self):
        """Test C++ parameter validation."""
        # Test invalid number of players
        with self.assertRaises(Exception):
            pyspiel.load_game("param_social_dilemma", {"num_players": 1})
        
        # Test invalid number of actions
        with self.assertRaises(Exception):
            pyspiel.load_game("param_social_dilemma", {"num_actions": 1})

    def test_cpp_utility_bounds(self):
        """Test C++ utility bound calculations."""
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": 2,
            "num_actions": 2,
            "max_game_length": 5
        })
        
        min_utility = game.min_utility()
        max_utility = game.max_utility()
        
        # Should be based on payoff matrix and max game length
        self.assertLess(min_utility, max_utility)
        self.assertLessEqual(min_utility, 0)  # Some payoffs can be negative
        self.assertGreater(max_utility, 0)


if __name__ == "__main__":
    unittest.main()
