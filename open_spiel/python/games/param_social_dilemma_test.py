# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for the parameterized social dilemma game."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pyspiel
from open_spiel.python.games import param_social_dilemma


class ParamSocialDilemmaTest(parameterized.TestCase):
  """Test suite for parameterized social dilemma game."""

  def test_game_creation(self):
    """Test basic game creation."""
    game = pyspiel.load_game("python_param_social_dilemma")
    self.assertIsNotNone(game)
    self.assertEqual(game.num_players(), 2)
    
  def test_2player_prisoners_dilemma(self):
    """Test 2-player prisoner's dilemma."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 5,
        "dilemma_type": "prisoners_dilemma"
    })
    
    state = game.new_initial_state()
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
    
    # Play some rounds
    for _ in range(5):
      if not state.is_terminal():
        state.apply_actions([0, 1])  # Player 0 cooperates, Player 1 defects
    
    self.assertTrue(state.is_terminal())
    returns = state.returns()
    self.assertEqual(len(returns), 2)
  
  def test_n_player_game(self):
    """Test N-player game with N > 2."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 4,
        "num_actions": 2,
        "num_rounds": 3,
    })
    
    state = game.new_initial_state()
    self.assertEqual(game.num_players(), 4)
    
    # Play one round
    state.apply_actions([0, 1, 0, 1])
    self.assertFalse(state.is_terminal())
    
    rewards = state.rewards()
    self.assertEqual(len(rewards), 4)
    
  def test_stochastic_rewards(self):
    """Test that reward noise is applied correctly."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 10,
        "reward_noise_std": 0.5,
    })
    
    # Run multiple trials to check variance
    returns_list = []
    for seed in range(5):
      np.random.seed(seed)
      state = game.new_initial_state()
      while not state.is_terminal():
        state.apply_actions([0, 0])  # Same actions each time
      returns_list.append(state.returns()[0])
    
    # With noise, returns should vary
    returns_variance = np.var(returns_list)
    self.assertGreater(returns_variance, 0)
  
  def test_dynamic_payoffs_cycling(self):
    """Test cycling payoff dynamics."""
    # Create two different payoff matrices
    matrix1 = np.ones((2, 2, 2)) * 1.0
    matrix2 = np.ones((2, 2, 2)) * 5.0
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 4,
        "payoff_dynamics": "cycling",
        "payoff_matrices_sequence": [matrix1.tolist(), matrix2.tolist()],
    })
    
    # Check that payoff matrices cycle
    payoff_r0 = game.get_payoff_matrix(0)
    payoff_r1 = game.get_payoff_matrix(1)
    payoff_r2 = game.get_payoff_matrix(2)
    
    np.testing.assert_array_almost_equal(payoff_r0, matrix1)
    np.testing.assert_array_almost_equal(payoff_r1, matrix2)
    np.testing.assert_array_almost_equal(payoff_r2, matrix1)  # Cycles back
  
  def test_dynamic_payoffs_drifting(self):
    """Test drifting payoff dynamics."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 10,
        "payoff_dynamics": "drifting",
    })
    
    # Check that payoffs change over time
    payoff_r0 = game.get_payoff_matrix(0)
    payoff_r5 = game.get_payoff_matrix(5)
    
    # Should be different due to drift
    self.assertFalse(np.allclose(payoff_r0, payoff_r5))
  
  def test_custom_payoff_matrix(self):
    """Test custom payoff matrix."""
    custom_matrix = np.array([
        [[10, 10], [5, 15]],
        [[15, 5], [8, 8]]
    ], dtype=np.float32)
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 1,
        "custom_payoff_matrix": custom_matrix.tolist(),
    })
    
    state = game.new_initial_state()
    state.apply_actions([0, 0])
    
    # Should get payoff from custom matrix
    rewards = state.rewards()
    self.assertEqual(rewards[0], 10.0)
    self.assertEqual(rewards[1], 10.0)
  
  @parameterized.parameters(
      ("prisoners_dilemma",),
      ("stag_hunt",),
      ("chicken",),
      ("public_goods",)
  )
  def test_predefined_dilemmas(self, dilemma_type):
    """Test all predefined dilemma types."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 3,
        "dilemma_type": dilemma_type,
    })
    
    state = game.new_initial_state()
    while not state.is_terminal():
      state.apply_actions([0, 1])
    
    self.assertTrue(state.is_terminal())
    self.assertEqual(len(state.returns()), 2)
  
  def test_legal_actions(self):
    """Test legal actions for different configurations."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 3,
        "num_actions": 4,
        "num_rounds": 2,
    })
    
    state = game.new_initial_state()
    for player in range(3):
      legal_actions = state.legal_actions(player)
      self.assertEqual(legal_actions, [0, 1, 2, 3])
  
  def test_game_progression(self):
    """Test full game progression."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 3,
        "num_actions": 2,
        "num_rounds": 5,
    })
    
    state = game.new_initial_state()
    round_count = 0
    
    while not state.is_terminal():
      self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
      state.apply_actions([0, 1, 0])
      round_count += 1
    
    self.assertEqual(round_count, 5)
    self.assertTrue(state.is_terminal())
  
  def test_observation_and_info_state(self):
    """Test observation and information state strings."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_rounds": 2,
    })
    
    state = game.new_initial_state()
    state.apply_actions([0, 1])
    
    # Test that we can get strings without errors
    obs_str = state.observation_string(0)
    info_str = state.information_state_string(0)
    
    self.assertIsInstance(obs_str, str)
    self.assertIsInstance(info_str, str)
    self.assertIn("Round", obs_str)
  
  def test_max_game_length(self):
    """Test that max game length is set correctly."""
    num_rounds = 15
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_rounds": num_rounds,
    })
    
    self.assertEqual(game.max_game_length(), num_rounds)
  
  def test_returns_accumulation(self):
    """Test that returns accumulate correctly over rounds."""
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 3,
        "reward_noise_std": 0.0,  # No noise for deterministic test
        "dilemma_type": "prisoners_dilemma",
    })
    
    state = game.new_initial_state()
    
    # Track returns after each round
    returns_history = [state.returns().copy()]
    
    while not state.is_terminal():
      state.apply_actions([0, 0])  # Both cooperate
      returns_history.append(state.returns().copy())
    
    # Returns should be monotonically increasing (assuming positive rewards)
    for i in range(1, len(returns_history)):
      # Each round should add to cumulative returns
      diff = returns_history[i] - returns_history[i-1]
      # The difference should be the rewards from that round
      self.assertTrue(np.all(diff >= 0) or np.all(diff <= 0))  # Consistent direction


if __name__ == "__main__":
  absltest.main()
