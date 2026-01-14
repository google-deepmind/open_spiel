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

"""Tests for parameterized social dilemma game."""

import unittest

import numpy as np
import pyspiel

from open_spiel.python.games import param_social_dilemma
from open_spiel.python.games.param_social_dilemma_bots import (
    create_bot, get_available_bot_types
)


class ParamSocialDilemmaTest(unittest.TestCase):
    """Test cases for the parameterized social dilemma game."""

    def setUp(self):
        """Set up test fixtures."""
        self.game_2p = pyspiel.load_game("python_param_social_dilemma", 
                                         {"num_players": 2, "num_actions": 2})
        self.game_3p = pyspiel.load_game("python_param_social_dilemma", 
                                         {"num_players": 3, "num_actions": 2})
        self.game_4p_3a = pyspiel.load_game("python_param_social_dilemma", 
                                              {"num_players": 4, "num_actions": 3})

    def test_game_creation(self):
        """Test game creation with different parameters."""
        # Test 2-player game
        self.assertEqual(self.game_2p.num_players(), 2)
        self.assertEqual(self.game_2p.num_distinct_actions(), 2)
        
        # Test 3-player game
        self.assertEqual(self.game_3p.num_players(), 3)
        self.assertEqual(self.game_3p.num_distinct_actions(), 2)
        
        # Test 4-player, 3-action game
        self.assertEqual(self.game_4p_3a.num_players(), 4)
        self.assertEqual(self.game_4p_3a.num_distinct_actions(), 3)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid number of players
        with self.assertRaises(ValueError):
            pyspiel.load_game("python_param_social_dilemma", {"num_players": 1})
        
        # Test invalid number of actions
        with self.assertRaises(ValueError):
            pyspiel.load_game("python_param_social_dilemma", {"num_actions": 1})

    def test_initial_state(self):
        """Test initial state properties."""
        state = self.game_2p.new_initial_state()
        
        # Should be simultaneous move
        self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
        
        # Should not be terminal
        self.assertFalse(state.is_terminal())
        
        # Should have legal actions
        for player in range(self.game_2p.num_players()):
            legal_actions = state.legal_actions(player)
            self.assertEqual(len(legal_actions), 2)
            self.assertEqual(legal_actions, [0, 1])

    def test_action_application(self):
        """Test applying actions and state transitions."""
        state = self.game_2p.new_initial_state()
        
        # Apply cooperate-cooperate
        state.apply_actions([0, 0])
        
        # Should now be chance node
        self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
        
        # Check rewards
        rewards = state.rewards()
        self.assertEqual(len(rewards), 2)
        # Both players should get cooperation reward (3 in default PD)
        self.assertEqual(rewards[0], 3)
        self.assertEqual(rewards[1], 3)

    def test_payoff_matrix(self):
        """Test payoff matrix calculations."""
        # Test default 2-player prisoner's dilemma
        state = self.game_2p.new_initial_state()
        
        # Cooperate-Defect: Player 0 gets 0, Player 1 gets 5
        state.apply_actions([0, 1])
        rewards = state.rewards()
        self.assertEqual(rewards[0], 0)
        self.assertEqual(rewards[1], 5)
        
        # Defect-Defect: Both get 1
        state = self.game_2p.new_initial_state()
        state.apply_actions([1, 1])
        rewards = state.rewards()
        self.assertEqual(rewards[0], 1)
        self.assertEqual(rewards[1], 1)

    def test_three_player_game(self):
        """Test 3-player game mechanics."""
        state = self.game_3p.new_initial_state()
        
        # All cooperate
        state.apply_actions([0, 0, 0])
        rewards = state.rewards()
        self.assertEqual(len(rewards), 3)
        
        # All should get positive rewards for cooperation
        for reward in rewards:
            self.assertGreater(reward, 0)

    def test_game_termination(self):
        """Test game termination mechanics."""
        # Create game with high termination probability for testing
        game = pyspiel.load_game("python_param_social_dilemma", 
                                {"num_players": 2, 
                                 "termination_probability": 1.0})  # Always terminate
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        
        # Should be chance node
        self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
        
        # Apply chance outcome (should always be STOP)
        outcomes = state.chance_outcomes()
        self.assertEqual(len(outcomes), 2)
        
        # Apply stop
        state.apply_action(1)  # STOP action
        self.assertTrue(state.is_terminal())

    def test_dynamic_payoffs(self):
        """Test dynamic payoff function."""
        def increasing_payoffs(base_matrix, timestep):
            """Payoffs that increase over time."""
            multiplier = 1 + 0.1 * timestep
            return [[cell * multiplier for cell in row] for row in base_matrix]
        
        game = pyspiel.load_game("python_param_social_dilemma", {
            "num_players": 2,
            "payoff_function": increasing_payoffs
        })
        
        state1 = game.new_initial_state()
        state1.apply_actions([0, 0])
        rewards1 = state1.rewards()
        
        # Continue to next round
        state1.apply_action(0)  # CONTINUE
        state1.apply_actions([0, 0])
        rewards2 = state1.rewards()
        
        # Rewards should be higher in second round
        self.assertGreater(rewards2[0], rewards1[0])
        self.assertGreater(rewards2[1], rewards1[1])

    def test_stochastic_rewards(self):
        """Test stochastic reward noise."""
        game = pyspiel.load_game("python_param_social_dilemma", {
            "num_players": 2,
            "reward_noise": {"type": "gaussian", "std": 0.1},
            "seed": 42
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        rewards = state.rewards()
        
        # Rewards should be close to base value but with noise
        self.assertAlmostEqual(rewards[0], 3, delta=0.5)
        self.assertAlmostEqual(rewards[1], 3, delta=0.5)

    def test_action_history(self):
        """Test action history tracking."""
        state = self.game_2p.new_initial_state()
        
        # Apply some actions
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        state.apply_actions([1, 0])
        
        # Check history strings
        history_p0 = state.action_history_string(0)
        history_p1 = state.action_history_string(1)
        
        self.assertEqual(history_p0, "CD")  # Cooperate, then Defect
        self.assertEqual(history_p1, "DC")  # Defect, then Cooperate

    def test_observation_tensor(self):
        """Test observation tensor creation."""
        game = pyspiel.load_game("python_param_social_dilemma", 
                                {"num_players": 2, "max_game_length": 10})
        state = game.new_initial_state()
        observer = game.make_py_observer()
        
        # Apply some actions
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        
        observer.set_from(state, 0)
        tensor = observer.tensor
        
        # Tensor should have correct size
        expected_size = 10 * 2 + 1  # max_length * num_players + iteration
        self.assertEqual(len(tensor), expected_size)
        
        # Should have action history encoded
        self.assertEqual(tensor[0], 0)  # Player 0's first action
        self.assertEqual(tensor[10], 1)  # Player 1's first action

    def test_custom_payoff_matrix(self):
        """Test custom payoff matrix."""
        custom_payoff = [
            [[10, 0], [20, 5]],  # Player 0 payoffs
            [[10, 20], [0, 5]]   # Player 1 payoffs
        ]
        
        game = pyspiel.load_game("python_param_social_dilemma", {
            "num_players": 2,
            "payoff_matrix": custom_payoff
        })
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])  # Both cooperate
        rewards = state.rewards()
        
        self.assertEqual(rewards[0], 10)
        self.assertEqual(rewards[1], 10)

    def test_returns_accumulation(self):
        """Test that returns accumulate correctly over multiple rounds."""
        state = self.game_2p.new_initial_state()
        
        # Play multiple rounds
        for _ in range(3):
            state.apply_actions([0, 0])  # Both cooperate
            state.apply_action(0)  # Continue
        
        returns = state.returns()
        # Should have accumulated 3 rounds of cooperation rewards (3 each)
        self.assertEqual(returns[0], 9)
        self.assertEqual(returns[1], 9)


class ParamSocialDilemmaBotsTest(unittest.TestCase):
    """Test cases for the social dilemma bots."""

    def setUp(self):
        """Set up test fixtures."""
        self.game = pyspiel.load_game("python_param_social_dilemma", 
                                      {"num_players": 2, "num_actions": 2})

    def test_bot_creation(self):
        """Test bot creation for all available types."""
        bot_types = get_available_bot_types()
        
        for bot_type in bot_types:
            bot = create_bot(bot_type, 0, self.game)
            self.assertIsNotNone(bot)
            self.assertEqual(bot._player_id, 0)

    def test_always_cooperate_bot(self):
        """Test AlwaysCooperateBot behavior."""
        bot = create_bot("always_cooperate", 0, self.game)
        state = self.game.new_initial_state()
        
        action = bot.step(state)
        self.assertEqual(action, 0)  # Should always cooperate

    def test_always_defect_bot(self):
        """Test AlwaysDefectBot behavior."""
        bot = create_bot("always_defect", 0, self.game)
        state = self.game.new_initial_state()
        
        action = bot.step(state)
        self.assertEqual(action, 1)  # Should always defect

    def test_tit_for_tat_bot(self):
        """Test TitForTatBot behavior."""
        bot = create_bot("tit_for_tat", 0, self.game)
        state = self.game.new_initial_state()
        
        # First move should be cooperation
        action = bot.step(state)
        self.assertEqual(action, 0)
        
        # Simulate opponent defecting
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        
        # Next move should copy opponent's last action (defect)
        action = bot.step(state)
        self.assertEqual(action, 1)

    def test_grim_trigger_bot(self):
        """Test GrimTriggerBot behavior."""
        bot = create_bot("grim_trigger", 0, self.game)
        state = self.game.new_initial_state()
        
        # Should cooperate initially
        action = bot.step(state)
        self.assertEqual(action, 0)
        
        # Simulate opponent defecting
        state.apply_actions([0, 1])
        state.apply_action(0)  # Continue
        
        # Should now defect forever
        action = bot.step(state)
        self.assertEqual(action, 1)
        
        # Even if opponent cooperates, should still defect
        state.apply_actions([1, 0])
        state.apply_action(0)  # Continue
        action = bot.step(state)
        self.assertEqual(action, 1)

    def test_random_bot(self):
        """Test RandomBot behavior."""
        bot = create_bot("random", 0, self.game)
        state = self.game.new_initial_state()
        
        # Should return valid action
        action = bot.step(state)
        self.assertIn(action, [0, 1])

    def test_bot_restart(self):
        """Test bot restart functionality."""
        bot = create_bot("tit_for_tat", 0, self.game)
        state = self.game.new_initial_state()
        
        # Play some moves
        bot.step(state)
        state.apply_actions([0, 1])
        state.apply_action(0)
        bot.step(state)
        
        # Restart bot
        bot.restart_at(state)
        
        # Should behave like initial state again
        action = bot.step(state)
        self.assertEqual(action, 0)  # Should cooperate on first move after restart


class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_bots_tournament(self):
        """Test a simple tournament between bots."""
        game = pyspiel.load_game("python_param_social_dilemma", 
                                {"num_players": 2, "termination_probability": 0.0})  # Never terminate
        
        # Create two bots
        bot1 = create_bot("tit_for_tat", 0, game)
        bot2 = create_bot("always_cooperate", 1, game)
        
        state = game.new_initial_state()
        
        # Play 5 rounds
        for round_num in range(5):
            if state.is_terminal():
                break
                
            # Get actions from bots
            action1 = bot1.step(state)
            action2 = bot2.step(state)
            
            # Apply actions
            state.apply_actions([action1, action2])
            
            # Handle chance node
            if state.current_player() == pyspiel.PlayerId.CHANCE:
                # Always continue
                state.apply_action(0)
        
        # Check that game progressed
        returns = state.returns()
        self.assertGreater(returns[0], 0)
        self.assertGreater(returns[1], 0)

    def test_three_player_bots(self):
        """Test bots in 3-player game."""
        game = pyspiel.load_game("python_param_social_dilemma", 
                                {"num_players": 3, "termination_probability": 0.0})
        
        bots = [
            create_bot("always_cooperate", 0, game),
            create_bot("always_defect", 1, game),
            create_bot("tit_for_tat", 2, game)
        ]
        
        state = game.new_initial_state()
        
        # Play a few rounds
        for round_num in range(3):
            if state.is_terminal():
                break
                
            actions = [bot.step(state) for bot in bots]
            state.apply_actions(actions)
            
            if state.current_player() == pyspiel.PlayerId.CHANCE:
                state.apply_action(0)  # Continue
        
        # Should have completed some rounds
        returns = state.returns()
        self.assertEqual(len(returns), 3)


if __name__ == "__main__":
    unittest.main()
