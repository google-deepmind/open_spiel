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

"""Game statistics analyzer for OpenSpiel games.

This utility computes and displays key complexity metrics for any OpenSpiel game,
including state space size, branching factor, game depth, and information type.

Usage:
  # Analyze a small game with full traversal
  python game_stats.py --game_name=tic_tac_toe

  # Analyze a large game with sampling
  python game_stats.py --game_name=chess --num_samples=1000

  # Export results to JSON
  python game_stats.py --game_name=kuhn_poker --output_json=stats.json
"""

import argparse
import json
import random
import sys
from typing import Dict, Any, Optional, Set, Tuple

import pyspiel


def load_game_safely(game_name: str) -> Optional[pyspiel.Game]:
  """Load a game with error handling.

  Args:
    game_name: Name of the game to load.

  Returns:
    Game object or None if loading fails.
  """
  try:
    return pyspiel.load_game(game_name)
  except (pyspiel.SpielError, RuntimeError) as e:
    print("ERROR: Failed to load game '{0}'".format(game_name),
          file=sys.stderr)
    print("Original error: {0}".format(str(e)), file=sys.stderr)
    
    # Suggest similar games
    try:
      import difflib
      all_games = [g.short_name for g in pyspiel.registered_games()]
      suggestions = difflib.get_close_matches(game_name, all_games, n=3)
      if suggestions:
        print("\nDid you mean: {0}?".format(", ".join(suggestions)),
              file=sys.stderr)
    except Exception:
      pass
    
    return None


def analyze_game_dfs(game: pyspiel.Game,
                      max_states: int = 10000) -> Dict[str, Any]:
  """Analyze a game using depth-first search traversal.

  Args:
    game: The game to analyze.
    max_states: Maximum number of states to visit before switching to sampling.

  Returns:
    Dictionary containing game statistics.
  """
  visited_states: Set[str] = set()
  terminal_count = 0
  total_branching = 0
  branching_count = 0
  depths = []
  
  def dfs(state, depth: int) -> bool:
    """DFS helper that returns True if we should continue."""
    nonlocal terminal_count, total_branching, branching_count
    
    state_str = state.history_str()
    if state_str in visited_states:
      return True
    
    visited_states.add(state_str)
    
    if len(visited_states) >= max_states:
      return False  # Stop traversal
    
    if state.is_terminal():
      terminal_count += 1
      depths.append(depth)
      return True
    
    # Count branching factor for non-terminal states
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
    elif state.is_simultaneous_node():
      # For simultaneous games, count actions for first player
      num_actions = len(state.legal_actions(0))
    else:
      num_actions = len(state.legal_actions())
    
    if num_actions > 0:
      total_branching += num_actions
      branching_count += 1
    
    # Explore children
    if state.is_chance_node():
      for action, _ in state.chance_outcomes():
        child = state.child(action)
        if not dfs(child, depth + 1):
          return False
    elif state.is_simultaneous_node():
      # Sample one joint action for simultaneous games
      actions = []
      for pid in range(game.num_players()):
        legal = state.legal_actions(pid)
        if legal:
          actions.append(legal[0])
        else:
          actions.append(0)
      child = state.child(actions)
      if not dfs(child, depth + 1):
        return False
    else:
      for action in state.legal_actions():
        child = state.child(action)
        if not dfs(child, depth + 1):
          return False
    
    return True
  
  initial_state = game.new_initial_state()
  fully_explored = dfs(initial_state, 0)
  
  avg_branching = (total_branching / branching_count 
                   if branching_count > 0 else 0)
  avg_depth = sum(depths) / len(depths) if depths else 0
  
  return {
      "total_states": len(visited_states),
      "terminal_states": terminal_count,
      "avg_branching_factor": avg_branching,
      "avg_depth": avg_depth,
      "fully_explored": fully_explored,
  }


def analyze_game_sampling(game: pyspiel.Game,
                          num_samples: int) -> Dict[str, Any]:
  """Analyze a game using random sampling.

  Args:
    game: The game to analyze.
    num_samples: Number of random rollouts to perform.

  Returns:
    Dictionary containing estimated game statistics.
  """
  terminal_count = num_samples  # All rollouts reach terminal
  total_branching = 0
  branching_count = 0
  depths = []
  
  for _ in range(num_samples):
    state = game.new_initial_state()
    depth = 0
    
    while not state.is_terminal():
      depth += 1
      
      # Count branching factor
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        num_actions = len(outcomes)
        action = random.choices(action_list, weights=prob_list)[0]
      elif state.is_simultaneous_node():
        num_actions = 0
        actions = []
        for pid in range(game.num_players()):
          legal = state.legal_actions(pid)
          num_actions += len(legal)
          actions.append(random.choice(legal) if legal else 0)
        action = actions
      else:
        legal = state.legal_actions()
        num_actions = len(legal)
        action = random.choice(legal) if legal else 0
      
      if num_actions > 0:
        total_branching += num_actions
        branching_count += 1
      
      # Apply action
      if state.is_simultaneous_node():
        state.apply_actions(action)
      else:
        state.apply_action(action)
    
    depths.append(depth)
  
  avg_branching = (total_branching / branching_count 
                   if branching_count > 0 else 0)
  avg_depth = sum(depths) / len(depths) if depths else 0
  
  return {
      "total_states": "~{0} (estimated)".format(num_samples * int(avg_depth)),
      "terminal_states": "~{0} (sampled)".format(terminal_count),
      "avg_branching_factor": avg_branching,
      "avg_depth": avg_depth,
      "fully_explored": False,
  }


def get_game_properties(game: pyspiel.Game) -> Dict[str, Any]:
  """Extract basic game properties.

  Args:
    game: The game to analyze.

  Returns:
    Dictionary of game properties.
  """
  game_type = game.get_type()
  
  return {
      "name": game_type.short_name,
      "num_players": game.num_players(),
      "perfect_information": (
          game_type.information == pyspiel.GameType.Information.PERFECT_INFORMATION
      ),
      "dynamics": str(game_type.dynamics),
      "utility": str(game_type.utility),
      "max_game_length": game.max_game_length(),
      "min_utility": game.min_utility(),
      "max_utility": game.max_utility(),
  }


def print_statistics(properties: Dict[str, Any],
                     stats: Dict[str, Any]) -> None:
  """Print statistics in a human-readable format.

  Args:
    properties: Game properties dictionary.
    stats: Game statistics dictionary.
  """
  separator = "=" * 70
  print("\n{0}".format(separator))
  print("Game Statistics: {0}".format(properties["name"]))
  print("{0}\n".format(separator))
  
  print("Game Properties:")
  print("  Players: {0}".format(properties["num_players"]))
  print("  Perfect Information: {0}".format(properties["perfect_information"]))
  print("  Dynamics: {0}".format(properties["dynamics"]))
  print("  Utility Type: {0}".format(properties["utility"]))
  print("  Max Game Length: {0}".format(properties["max_game_length"]))
  print("  Utility Range: [{0:.1f}, {1:.1f}]".format(
      properties["min_utility"], properties["max_utility"]))
  
  print("\nComplexity Metrics:")
  print("  Total States: {0}".format(stats["total_states"]))
  print("  Terminal States: {0}".format(stats["terminal_states"]))
  print("  Avg Branching Factor: {0:.2f}".format(
      stats["avg_branching_factor"]))
  print("  Avg Game Depth: {0:.2f}".format(stats["avg_depth"]))
  
  if stats["fully_explored"]:
    print("\n  Analysis: Full game tree explored")
  else:
    print("\n  Analysis: Estimated via sampling")
  
  print("{0}\n".format(separator))


def main():
  """Main entry point for the game statistics analyzer."""
  parser = argparse.ArgumentParser(
      description="Analyze OpenSpiel game complexity and statistics",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  python game_stats.py --game_name=tic_tac_toe
  python game_stats.py --game_name=chess --num_samples=1000
  python game_stats.py --game_name=kuhn_poker --output_json=stats.json
      """)
  
  parser.add_argument(
      "--game_name",
      type=str,
      required=True,
      help="Name of the OpenSpiel game to analyze")
  
  parser.add_argument(
      "--num_samples",
      type=int,
      default=None,
      help="Number of random rollouts for large games (default: auto)")
  
  parser.add_argument(
      "--output_json",
      type=str,
      default=None,
      help="Optional: Save results to JSON file")
  
  parser.add_argument(
      "--max_states",
      type=int,
      default=10000,
      help="Max states for DFS before switching to sampling (default: 10000)")
  
  args = parser.parse_args()
  
  # Load game
  game = load_game_safely(args.game_name)
  if game is None:
    sys.exit(1)
  
  # Get basic properties
  properties = get_game_properties(game)
  
  # Analyze game
  print("Analyzing game '{0}'...".format(args.game_name))
  
  if args.num_samples is not None:
    # Use sampling
    print("Using sampling with {0} rollouts...".format(args.num_samples))
    stats = analyze_game_sampling(game, args.num_samples)
  else:
    # Try DFS first, fall back to sampling if too large
    print("Attempting full exploration (max {0} states)...".format(
        args.max_states))
    stats = analyze_game_dfs(game, args.max_states)
    
    if not stats["fully_explored"]:
      print("Game too large, switching to sampling (1000 rollouts)...")
      stats = analyze_game_sampling(game, 1000)
  
  # Print results
  print_statistics(properties, stats)
  
  # Save to JSON if requested
  if args.output_json:
    output = {
        "properties": properties,
        "statistics": stats,
    }
    # Convert non-serializable values to strings
    for key, value in output["statistics"].items():
      if not isinstance(value, (int, float, str, bool, type(None))):
        output["statistics"][key] = str(value)
    
    with open(args.output_json, "w") as f:
      json.dump(output, f, indent=2)
    print("Results saved to: {0}".format(args.output_json))


if __name__ == "__main__":
  main()
