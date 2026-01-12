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

"""Visualization utilities for parameterized social dilemma game.

This script provides text-based visualizations to help understand
game dynamics, payoff structures, and strategy outcomes.
"""

import numpy as np
import pyspiel


def visualize_payoff_matrix(game, round_num=0):
  """Visualize the payoff matrix as a text table.
  
  Args:
    game: ParamSocialDilemmaGame instance
    round_num: Round number for which to show payoffs
  """
  matrix = game.get_payoff_matrix(round_num)
  
  print(f"\nPayoff Matrix (Round {round_num}):")
  print("=" * 60)
  
  if game._num_players == 2:
    # Standard 2x2 matrix visualization
    print("\n              Player 1")
    print("           Cooperate  Defect")
    print("Player 0  ┌─────────┬─────────┐")
    
    # Row 0: Cooperate
    p0_cc = matrix[0, 0, 0]
    p1_cc = matrix[0, 0, 1]
    p0_cd = matrix[0, 1, 0]
    p1_cd = matrix[0, 1, 1]
    
    print(f"Cooperate │ {p0_cc:5.1f},{p1_cc:5.1f} │ {p0_cd:5.1f},{p1_cd:5.1f} │")
    print("          ├─────────┼─────────┤")
    
    # Row 1: Defect
    p0_dc = matrix[1, 0, 0]
    p1_dc = matrix[1, 0, 1]
    p0_dd = matrix[1, 1, 0]
    p1_dd = matrix[1, 1, 1]
    
    print(f"Defect    │ {p0_dc:5.1f},{p1_dc:5.1f} │ {p0_dd:5.1f},{p1_dd:5.1f} │")
    print("          └─────────┴─────────┘")
  else:
    # For N > 2, show simplified mean-field structure
    print(f"\n(Mean-field approximation for {game._num_players} players)")
    print("\n              Avg Others")
    print("           Cooperate  Defect")
    print("Own Act   ┌─────────┬─────────┐")
    
    print(f"Cooperate │  {matrix[0, 0, 0]:6.2f}  │  {matrix[0, 1, 0]:6.2f}  │")
    print("          ├─────────┼─────────┤")
    print(f"Defect    │  {matrix[1, 0, 0]:6.2f}  │  {matrix[1, 1, 0]:6.2f}  │")
    print("          └─────────┴─────────┘")


def visualize_dynamics(game, num_rounds_to_show=10):
  """Visualize how payoffs change over time.
  
  Args:
    game: ParamSocialDilemmaGame instance
    num_rounds_to_show: Number of rounds to display
  """
  print(f"\nPayoff Dynamics over Time:")
  print("=" * 60)
  
  print(f"\nShowing cooperation payoff (C,C) for first {num_rounds_to_show} rounds:")
  print("\nRound | Payoff (P0) | Payoff (P1) | Visualization")
  print("-" * 60)
  
  for round_num in range(min(num_rounds_to_show, game._num_rounds)):
    matrix = game.get_payoff_matrix(round_num)
    
    if game._num_players == 2:
      p0_payoff = matrix[0, 0, 0]
      p1_payoff = matrix[0, 0, 1]
    else:
      p0_payoff = matrix[0, 0, 0]
      p1_payoff = p0_payoff  # Same for others in mean-field
    
    # Create simple bar visualization
    bar_length = int(p0_payoff)
    bar = "█" * bar_length
    
    print(f"{round_num:5d} | {p0_payoff:11.2f} | {p1_payoff:11.2f} | {bar}")


def visualize_strategy_heatmap(game):
  """Create a text-based heatmap of all strategy combinations.
  
  Args:
    game: ParamSocialDilemmaGame instance (2-player only)
  """
  if game._num_players != 2:
    print("Strategy heatmap only available for 2-player games")
    return
  
  print("\nStrategy Outcome Heatmap:")
  print("=" * 60)
  print("(Showing Player 0's payoff for each strategy combination)\n")
  
  # Run all strategy combinations
  num_actions = game._num_actions
  
  # Create result matrix
  results = np.zeros((num_actions, num_actions))
  
  for a0 in range(num_actions):
    for a1 in range(num_actions):
      state = game.new_initial_state()
      
      while not state.is_terminal():
        state.apply_actions([a0, a1])
      
      results[a0, a1] = state.returns()[0]
  
  # Display as text heatmap
  print("       Player 1 Actions")
  print("       ", end="")
  for a1 in range(num_actions):
    print(f"  A{a1}  ", end="")
  print()
  
  print("P0 Act ┌" + "─" * (7 * num_actions - 1) + "┐")
  
  for a0 in range(num_actions):
    print(f"  A{a0}  │", end="")
    for a1 in range(num_actions):
      value = results[a0, a1]
      
      # Color-code based on value (using characters)
      if value > (results.max() + results.min()) / 2:
        symbol = "█"  # High value
      elif value > results.min() + (results.max() - results.min()) / 4:
        symbol = "▓"  # Medium-high
      elif value > results.min() + (results.max() - results.min()) / 8:
        symbol = "▒"  # Medium-low
      else:
        symbol = "░"  # Low value
      
      print(f" {symbol}{value:4.1f} ", end="")
    print("│")
  
  print("       └" + "─" * (7 * num_actions - 1) + "┘")
  
  print(f"\nLegend: █ High ({results.max():.1f}) → ░ Low ({results.min():.1f})")


def visualize_game_trajectory(game, strategies, strategy_names=None):
  """Visualize a complete game trajectory.
  
  Args:
    game: ParamSocialDilemmaGame instance
    strategies: List of functions, one per player, that return actions
    strategy_names: Optional list of strategy names for display
  """
  if strategy_names is None:
    strategy_names = [f"Strategy{i}" for i in range(game._num_players)]
  
  print("\nGame Trajectory Visualization:")
  print("=" * 60)
  
  # Print header
  print("\nPlayers:")
  for i, name in enumerate(strategy_names):
    print(f"  Player {i}: {name}")
  
  print("\nRound | Actions | Rewards | Cumulative Returns")
  print("-" * 60)
  
  state = game.new_initial_state()
  round_num = 0
  
  while not state.is_terminal():
    # Get actions from strategies
    actions = [strategy(state, i) for i, strategy in enumerate(strategies)]
    
    # Apply actions
    state.apply_actions(actions)
    
    # Format output
    action_str = ",".join(f"{a}" for a in actions)
    reward_str = "[" + ",".join(f"{r:5.1f}" for r in state.rewards()) + "]"
    return_str = "[" + ",".join(f"{r:5.1f}" for r in state.returns()) + "]"
    
    print(f"{round_num:5d} | [{action_str:6s}] | {reward_str:20s} | {return_str}")
    
    round_num += 1
  
  print("\n" + "=" * 60)
  print(f"Final Returns: {state.returns()}")


def compare_dilemma_types():
  """Compare outcomes across different dilemma types."""
  print("\nSocial Dilemma Type Comparison:")
  print("=" * 70)
  
  dilemmas = ["prisoners_dilemma", "stag_hunt", "chicken", "public_goods"]
  
  for dilemma in dilemmas:
    print(f"\n{dilemma.replace('_', ' ').title()}:")
    print("-" * 70)
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "num_rounds": 1,
        "dilemma_type": dilemma
    })
    
    visualize_payoff_matrix(game, round_num=0)
    
    # Identify Nash equilibria (simplified check)
    print("\nStrategy Analysis:")
    matrix = game.get_payoff_matrix(0)
    
    # Check (C, C)
    state = game.new_initial_state()
    state.apply_actions([0, 0])
    cc_payoff = state.returns()
    
    # Check (D, D)
    state = game.new_initial_state()
    state.apply_actions([1, 1])
    dd_payoff = state.returns()
    
    # Check mixed
    state = game.new_initial_state()
    state.apply_actions([0, 1])
    cd_payoff = state.returns()
    
    print(f"  (C,C) returns: {cc_payoff}")
    print(f"  (D,D) returns: {dd_payoff}")
    print(f"  (C,D) returns: {cd_payoff}")


def main():
  """Run all visualizations."""
  print("\n" + "#"*70)
  print("#  PARAMETERIZED SOCIAL DILEMMA - VISUALIZATIONS")
  print("#"*70)
  
  # Example 1: Basic payoff matrix
  print("\n" + "="*70)
  print("Example 1: Payoff Matrix Visualization")
  print("="*70)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "dilemma_type": "prisoners_dilemma"
  })
  visualize_payoff_matrix(game, round_num=0)
  
  # Example 2: Dynamic payoffs
  print("\n" + "="*70)
  print("Example 2: Dynamic Payoffs Over Time")
  print("="*70)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "payoff_dynamics": "drifting",
      "num_rounds": 15
  })
  visualize_dynamics(game, num_rounds_to_show=15)
  
  # Example 3: Strategy heatmap
  print("\n" + "="*70)
  print("Example 3: Strategy Outcome Heatmap")
  print("="*70)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "dilemma_type": "stag_hunt",
      "num_rounds": 10
  })
  visualize_strategy_heatmap(game)
  
  # Example 4: Game trajectory
  print("\n" + "="*70)
  print("Example 4: Game Trajectory")
  print("="*70)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_rounds": 5
  })
  
  # Define simple strategies
  always_coop = lambda s, p: 0
  always_defect = lambda s, p: 1
  
  visualize_game_trajectory(
      game,
      [always_coop, always_defect],
      ["AlwaysCooperate", "AlwaysDefect"]
  )
  
  # Example 5: Compare dilemma types
  print("\n" + "="*70)
  print("Example 5: Compare All Dilemma Types")
  print("="*70)
  
  compare_dilemma_types()
  
  print("\n" + "#"*70)
  print("#  All visualizations completed!")
  print("#"*70 + "\n")


if __name__ == "__main__":
  main()
