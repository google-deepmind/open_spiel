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

"""Demonstration of using param_social_dilemma with OpenSpiel algorithms.

This script shows how to integrate the parameterized social dilemma game
with various MARL algorithms available in OpenSpiel.
"""

import numpy as np
import pyspiel

# Try to import common MARL utilities
try:
  from open_spiel.python.algorithms import fictitious_play
  HAVE_FP = True
except ImportError:
  HAVE_FP = False
  print("Warning: fictitious_play not available")


def example_random_simulation():
  """Simple random simulation of the game."""
  print("\n" + "="*70)
  print("  RANDOM SIMULATION")
  print("="*70 + "\n")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 3,
      "num_actions": 2,
      "num_rounds": 10,
      "reward_noise_std": 0.2
  })
  
  print(f"Game: {game.get_type().long_name}")
  print(f"Players: {game.num_players()}")
  print(f"Actions: {game.num_distinct_actions()}")
  print(f"Max game length: {game.max_game_length()}\n")
  
  # Run 5 random episodes
  for episode in range(5):
    print(f"Episode {episode + 1}:")
    state = game.new_initial_state()
    episode_return = np.zeros(game.num_players())
    
    while not state.is_terminal():
      # Random actions
      actions = [np.random.randint(0, game.num_distinct_actions()) 
                for _ in range(game.num_players())]
      state.apply_actions(actions)
      episode_return += state.rewards()
    
    print(f"  Returns: {episode_return}\n")


def example_policy_iteration():
  """Demonstrate evaluating fixed policies against each other."""
  print("\n" + "="*70)
  print("  POLICY EVALUATION")
  print("="*70 + "\n")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 20,
      "dilemma_type": "prisoners_dilemma"
  })
  
  # Define simple policies
  def always_cooperate_policy(state, player):
    return 0
  
  def always_defect_policy(state, player):
    return 1
  
  def random_policy(state, player):
    return np.random.randint(0, 2)
  
  policies = {
      "Cooperate": always_cooperate_policy,
      "Defect": always_defect_policy,
      "Random": random_policy
  }
  
  # Test all combinations
  print("Testing all policy combinations over 100 episodes:\n")
  
  for p1_name, p1_policy in policies.items():
    for p2_name, p2_policy in policies.items():
      returns_sum = np.zeros(2)
      num_episodes = 100
      
      for episode in range(num_episodes):
        np.random.seed(episode)
        state = game.new_initial_state()
        
        while not state.is_terminal():
          actions = [p1_policy(state, 0), p2_policy(state, 1)]
          state.apply_actions(actions)
        
        returns_sum += state.returns()
      
      avg_returns = returns_sum / num_episodes
      print(f"{p1_name:10s} vs {p2_name:10s}: "
            f"P1={avg_returns[0]:6.2f}, P2={avg_returns[1]:6.2f}")


def example_learning_curves():
  """Show how cooperation emerges (or doesn't) over time."""
  print("\n" + "="*70)
  print("  COOPERATION EMERGENCE")
  print("="*70 + "\n")
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 1,  # One-shot for simplicity
      "dilemma_type": "stag_hunt"
  })
  
  print("Simulating reinforcement learning-like behavior:")
  print("(Simple Q-learning style update without actual training)\n")
  
  # Initialize Q-tables (state-less for simplicity)
  q_tables = [
      np.zeros((2,)),  # Player 0: Q-values for [cooperate, defect]
      np.zeros((2,)),  # Player 1: Q-values for [cooperate, defect]
  ]
  
  learning_rate = 0.1
  epsilon = 0.3  # Exploration rate
  
  num_episodes = 50
  cooperation_rates = []
  
  for episode in range(num_episodes):
    state = game.new_initial_state()
    
    # Epsilon-greedy action selection
    actions = []
    for player in range(2):
      if np.random.random() < epsilon:
        action = np.random.randint(0, 2)  # Explore
      else:
        action = np.argmax(q_tables[player])  # Exploit
      actions.append(action)
    
    # Execute actions
    state.apply_actions(actions)
    rewards = state.rewards()
    
    # Update Q-values
    for player in range(2):
      action = actions[player]
      q_tables[player][action] += learning_rate * (rewards[player] - q_tables[player][action])
    
    # Track cooperation rate
    cooperation_rate = sum(a == 0 for a in actions) / len(actions)
    cooperation_rates.append(cooperation_rate)
    
    if (episode + 1) % 10 == 0:
      recent_coop = np.mean(cooperation_rates[-10:])
      print(f"Episode {episode + 1:3d}: Cooperation rate = {recent_coop:.2%}")
  
  print(f"\nFinal Q-values:")
  print(f"  Player 0: Cooperate={q_tables[0][0]:.2f}, Defect={q_tables[0][1]:.2f}")
  print(f"  Player 1: Cooperate={q_tables[1][0]:.2f}, Defect={q_tables[1][1]:.2f}")


def example_reward_shaping():
  """Example showing effect of reward noise on learning."""
  print("\n" + "="*70)
  print("  REWARD NOISE IMPACT")
  print("="*70 + "\n")
  
  noise_levels = [0.0, 0.5, 1.0, 2.0]
  
  print("Comparing variance in outcomes with different noise levels:\n")
  
  for noise_std in noise_levels:
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 10,
        "reward_noise_std": noise_std,
        "dilemma_type": "prisoners_dilemma"
    })
    
    # Run same strategy multiple times
    returns_list = []
    
    for seed in range(20):
      np.random.seed(seed)
      state = game.new_initial_state()
      
      while not state.is_terminal():
        state.apply_actions([0, 0])  # Both always cooperate
      
      returns_list.append(state.returns()[0])
    
    mean_return = np.mean(returns_list)
    std_return = np.std(returns_list)
    
    print(f"Noise std={noise_std:.1f}: Mean={mean_return:6.2f}, Std={std_return:5.2f}")


def example_scalability():
  """Test computational cost with varying number of players."""
  print("\n" + "="*70)
  print("  SCALABILITY TEST")
  print("="*70 + "\n")
  
  import time
  
  player_counts = [2, 4, 6, 8, 10]
  
  print("Testing game execution time with different player counts:\n")
  
  for num_players in player_counts:
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": num_players,
        "num_actions": 2,
        "num_rounds": 100
    })
    
    start_time = time.time()
    
    # Run 10 episodes
    for _ in range(10):
      state = game.new_initial_state()
      
      while not state.is_terminal():
        actions = [np.random.randint(0, 2) for _ in range(num_players)]
        state.apply_actions(actions)
    
    elapsed = time.time() - start_time
    
    print(f"{num_players:2d} players: {elapsed:6.3f}s for 10 episodes ({elapsed/10:6.3f}s per episode)")


def example_different_dilemmas():
  """Compare outcomes across different social dilemma types."""
  print("\n" + "="*70)
  print("  COMPARING SOCIAL DILEMMA TYPES")
  print("="*70 + "\n")
  
  dilemma_types = ["prisoners_dilemma", "stag_hunt", "chicken", "public_goods"]
  
  strategies = {
      "Both Cooperate": ([0, 0], "C-C"),
      "Both Defect": ([1, 1], "D-D"),
      "Mixed (P0 Coop)": ([0, 1], "C-D"),
      "Mixed (P0 Defect)": ([1, 0], "D-C"),
  }
  
  for dilemma in dilemma_types:
    print(f"\n{dilemma.replace('_', ' ').title()}:")
    print("-" * 50)
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 1,
        "dilemma_type": dilemma
    })
    
    for strategy_name, (actions, label) in strategies.items():
      state = game.new_initial_state()
      state.apply_actions(actions)
      
      rewards = state.rewards()
      print(f"  {label}: P0={rewards[0]:5.1f}, P1={rewards[1]:5.1f}")


def main():
  """Run all demonstration examples."""
  print("\n" + "#"*70)
  print("#  PARAM SOCIAL DILEMMA - ALGORITHM INTEGRATION DEMO")
  print("#"*70)
  
  example_random_simulation()
  example_policy_evaluation()
  example_learning_curves()
  example_reward_shaping()
  example_scalability()
  example_different_dilemmas()
  
  print("\n" + "#"*70)
  print("#  All demonstrations completed!")
  print("#"*70 + "\n")


if __name__ == "__main__":
  main()
