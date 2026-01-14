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

"""Example demonstrating the parameterized social dilemma game.

This script showcases various features:
1. Different dilemma types
2. N-player games (N > 2)
3. Dynamic payoff matrices
4. Stochastic rewards
5. Custom payoff configurations
"""

import numpy as np
import pyspiel


def print_separator(title):
  """Print a formatted separator."""
  print("\n" + "="*70)
  print(f"  {title}")
  print("="*70 + "\n")


def example_basic_prisoners_dilemma():
  """Basic 2-player Prisoner's Dilemma."""
  print_separator("Example 1: Basic 2-Player Prisoner's Dilemma")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 5,
      "dilemma_type": "prisoners_dilemma"
  })
  
  state = game.new_initial_state()
  print(f"Game: {game.get_type().long_name}")
  print(f"Players: {game.num_players()}")
  print(f"Actions per player: {game.num_distinct_actions()}")
  print(f"Rounds: {game.max_game_length()}\n")
  
  # Simulate a game where both players cooperate
  print("Simulating mutual cooperation...")
  while not state.is_terminal():
    state.apply_actions([0, 0])  # Action 0 = Cooperate
    print(f"  Round {len(state._action_history)}: "
          f"Rewards = {state.rewards()}, "
          f"Cumulative = {state.returns()}")
  
  print(f"\nFinal returns: {state.returns()}")


def example_n_player_game():
  """N-player social dilemma (N > 2)."""
  print_separator("Example 2: 5-Player Social Dilemma")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 5,
      "num_actions": 2,
      "num_rounds": 3,
      "dilemma_type": "public_goods"
  })
  
  state = game.new_initial_state()
  print(f"Players: {game.num_players()}")
  print(f"Actions: Cooperate (0) or Defect (1)\n")
  
  # Round 1: All cooperate
  print("Round 1: All players cooperate")
  state.apply_actions([0, 0, 0, 0, 0])
  print(f"  Rewards: {state.rewards()}")
  print(f"  Returns: {state.returns()}\n")
  
  # Round 2: Mixed strategies
  print("Round 2: Mixed strategies [C, D, C, D, C]")
  state.apply_actions([0, 1, 0, 1, 0])
  print(f"  Rewards: {state.rewards()}")
  print(f"  Returns: {state.returns()}\n")
  
  # Round 3: All defect
  print("Round 3: All players defect")
  state.apply_actions([1, 1, 1, 1, 1])
  print(f"  Rewards: {state.rewards()}")
  print(f"  Final Returns: {state.returns()}")


def example_dynamic_payoffs():
  """Dynamic payoff matrices that change over time."""
  print_separator("Example 3: Dynamic Payoff Matrices (Cycling)")
  
  # Create two different payoff matrices
  # Matrix 1: High cooperation payoff
  matrix_cooperation = np.array([
      [[5, 5], [0, 6]],
      [[6, 0], [1, 1]]
  ], dtype=np.float32)
  
  # Matrix 2: Low cooperation payoff
  matrix_competition = np.array([
      [[2, 2], [0, 3]],
      [[3, 0], [1, 1]]
  ], dtype=np.float32)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 6,
      "payoff_dynamics": "cycling",
      "payoff_matrices_sequence": [
          matrix_cooperation.tolist(),
          matrix_competition.tolist()
      ]
  })
  
  state = game.new_initial_state()
  print("Payoff matrices cycle between cooperation-favoring and competition-favoring\n")
  
  for round_num in range(6):
    matrix = game.get_payoff_matrix(round_num)
    cooperation_payoff = matrix[0, 0, 0]  # Both cooperate
    
    state.apply_actions([0, 0])  # Always cooperate
    
    print(f"Round {round_num + 1}: "
          f"Cooperation payoff = {cooperation_payoff:.1f}, "
          f"Rewards = {state.rewards()}")
  
  print(f"\nFinal returns: {state.returns()}")


def example_stochastic_rewards():
  """Stochastic rewards with Gaussian noise."""
  print_separator("Example 4: Stochastic Rewards")
  
  print("Running same strategy with different random seeds...\n")
  
  for seed in range(3):
    np.random.seed(seed)
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 5,
        "reward_noise_std": 0.5,
        "dilemma_type": "prisoners_dilemma"
    })
    
    state = game.new_initial_state()
    
    # Always play the same actions
    while not state.is_terminal():
      state.apply_actions([0, 0])
    
    print(f"Seed {seed}: Final returns = {state.returns()}")
  
  print("\nNote: Same actions lead to different outcomes due to reward noise!")


def example_custom_payoff():
  """Custom payoff matrix for specific experimental needs."""
  print_separator("Example 5: Custom Payoff Matrix")
  
  # Create an asymmetric game
  custom_matrix = np.array([
      [[8, 10], [2, 12]],  # Player 0's payoffs
      [[10, 2], [12, 5]]   # Player 1's payoffs (different from P0)
  ], dtype=np.float32)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 1,
      "custom_payoff_matrix": custom_matrix.tolist()
  })
  
  print("Custom asymmetric payoff matrix:")
  print("\nPlayer 0 payoffs:")
  print(f"  (C,C): {custom_matrix[0,0,0]}, (C,D): {custom_matrix[0,1,0]}")
  print(f"  (D,C): {custom_matrix[1,0,0]}, (D,D): {custom_matrix[1,1,0]}")
  print("\nPlayer 1 payoffs:")
  print(f"  (C,C): {custom_matrix[0,0,1]}, (D,C): {custom_matrix[0,1,1]}")
  print(f"  (C,D): {custom_matrix[1,0,1]}, (D,D): {custom_matrix[1,1,1]}")
  
  print("\nTesting all action combinations:\n")
  
  for a0 in [0, 1]:
    for a1 in [0, 1]:
      state = game.new_initial_state()
      state.apply_actions([a0, a1])
      a0_name = "C" if a0 == 0 else "D"
      a1_name = "C" if a1 == 0 else "D"
      print(f"  ({a0_name},{a1_name}): P0 gets {state.rewards()[0]:.1f}, "
            f"P1 gets {state.rewards()[1]:.1f}")


def example_all_dilemma_types():
  """Demonstrate all predefined dilemma types."""
  print_separator("Example 6: All Predefined Dilemma Types")
  
  dilemma_types = ["prisoners_dilemma", "stag_hunt", "chicken", "public_goods"]
  
  for dilemma in dilemma_types:
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 1,
        "dilemma_type": dilemma
    })
    
    print(f"\n{dilemma.replace('_', ' ').title()}:")
    
    # Test all four action combinations
    for a0 in [0, 1]:
      for a1 in [0, 1]:
        state = game.new_initial_state()
        state.apply_actions([a0, a1])
        a0_name = "C" if a0 == 0 else "D"
        a1_name = "C" if a1 == 0 else "D"
        print(f"  ({a0_name},{a1_name}): {state.rewards()}")


def example_drifting_payoffs():
  """Demonstrate drifting payoff dynamics."""
  print_separator("Example 7: Drifting Payoff Dynamics")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 10,
      "payoff_dynamics": "drifting",
      "dilemma_type": "prisoners_dilemma"
  })
  
  print("Cooperative payoff over time (with sinusoidal drift):\n")
  
  for round_num in range(10):
    matrix = game.get_payoff_matrix(round_num)
    coop_payoff = matrix[0, 0, 0]
    print(f"  Round {round_num}: {coop_payoff:.2f}")
  
  print("\nNote: Payoffs drift gradually, creating non-stationary environment!")


def main():
  """Run all examples."""
  print("\n" + "#"*70)
  print("#  PARAMETERIZED SOCIAL DILEMMA GAME - EXAMPLES")
  print("#"*70)
  
  example_basic_prisoners_dilemma()
  example_n_player_game()
  example_dynamic_payoffs()
  example_stochastic_rewards()
  example_custom_payoff()
  example_all_dilemma_types()
  example_drifting_payoffs()
  
  print("\n" + "#"*70)
  print("#  All examples completed!")
  print("#"*70 + "\n")


if __name__ == "__main__":
  main()
