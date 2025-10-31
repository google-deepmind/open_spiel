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

"""Python spiel example with improved error handling.

This example demonstrates the core OpenSpiel API by playing through a complete
game with random actions. It handles all three types of game nodes:
  - Chance nodes (random events like dice rolls, card dealing)
  - Simultaneous nodes (all players move at once)
  - Decision nodes (single player's turn)

Usage:
  # Play tic-tac-toe (default)
  python example.py

  # Play a different game
  python example.py --game_string=breakthrough

  # Play with custom parameters
  python example.py --game_string="kuhn_poker(players=3)"

For a list of all available games, run:
  python -c "import pyspiel; print(pyspiel.registered_games())"
"""

import difflib
import random
import sys
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

FLAGS = flags.FLAGS

# Game strings can just contain the name or the name followed by parameters
# and arguments, e.g. "breakthrough(rows=6,columns=6)"
flags.DEFINE_string("game_string", "tic_tac_toe", "Game string")


def load_game_with_error_handling(game_string):
  """Load a game with helpful error messages if it fails.

  Args:
    game_string: String identifier for the game, optionally with parameters.

  Returns:
    A pyspiel.Game object.

  Raises:
    ValueError: If the game cannot be loaded, with suggestions for fixes.
  """
  try:
    return pyspiel.load_game(game_string)
  except (pyspiel.SpielError, RuntimeError) as e:
    # Get list of all registered games for suggestions
    all_games_types = pyspiel.registered_games()
    # Convert GameType objects to strings (short names)
    all_games = [game.short_name for game in all_games_types]
    
    # Extract just the game name (before any parameters)
    game_name = game_string.split("(")[0].strip()
    
    # Find similar game names using fuzzy matching
    suggestions = difflib.get_close_matches(
        game_name, all_games, n=5, cutoff=0.4
    )
    
    # Build a helpful error message
    error_msg = f"\n{'='*70}\n"
    error_msg += f"ERROR: Failed to load game '{game_string}'\n"
    error_msg += f"{'='*70}\n"
    error_msg += f"\nOriginal error: {str(e)}\n"
    
    if suggestions:
      error_msg += f"\n Did you mean one of these games?\n"
      for i, suggestion in enumerate(suggestions, 1):
        error_msg += f"   {i}. {suggestion}\n"
    else:
      error_msg += f"\n No similar game names found.\n"
    
    error_msg += f"\n Tips:\n"
    error_msg += f"   - Game names are case-sensitive\n"
    error_msg += f"   - Use underscores (e.g., 'tic_tac_toe' not 'tictactoe')\n"
    error_msg += f"   - Check parameter syntax: game_name(param1=value1,param2=value2)\n"
    error_msg += f"\n To see all {len(all_games)} available games, run:\n"
    error_msg += f"   python -c \"import pyspiel; print(pyspiel.registered_games())\"\n"
    error_msg += f"\n   Or use --help to see common examples.\n"
    error_msg += f"{'='*70}\n"
    
    # Print to stderr for better visibility
    print(error_msg, file=sys.stderr)
    sys.exit(1)


def main(_):
  games_list = pyspiel.registered_games()
  print("Registered games:")
  print(games_list)

  action_string = None

  print("\n" + "="*70)
  print(f"Creating game: '{FLAGS.game_string}'")
  print("="*70 + "\n")
  
  # Load game with improved error handling
  game = load_game_with_error_handling(FLAGS.game_string)
  
  # Print game information
  print(f"✓ Successfully loaded game: {game.get_type().short_name}")
  print(f"  Players: {game.num_players()}")
  print(f"  Type: {game.get_type().dynamics}")
  print()

  # Create the initial state
  try:
    state = game.new_initial_state()
  except Exception as e:
    print(f"\n ERROR: Failed to create initial state: {e}", file=sys.stderr)
    print("This might indicate a problem with the game implementation.", 
          file=sys.stderr)
    sys.exit(1)

  # Print the initial state
  print("Initial state:")
  print(str(state))
  print()

  move_count = 0
  max_moves = 10000  # Prevent infinite loops
  
  while not state.is_terminal():
    move_count += 1
    
    # Safety check to prevent infinite loops
    if move_count > max_moves:
      print(f"\n WARNING: Reached maximum move limit ({max_moves})",
            file=sys.stderr)
      print("The game may be stuck in an infinite loop.", file=sys.stderr)
      sys.exit(1)
    
    try:
      # The state can be three different types: chance node,
      # simultaneous node, or decision node
      if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        num_actions = len(outcomes)
        print("Chance node, got " + str(num_actions) + " outcomes")
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        print("Sampled outcome: ",
              state.action_to_string(state.current_player(), action))
        state.apply_action(action)
      elif state.is_simultaneous_node():
        # Simultaneous node: sample actions for all players.
        chosen_actions = []
        for pid in range(game.num_players()):
          legal_actions = state.legal_actions(pid)
          if not legal_actions:
            raise ValueError(f"No legal actions for player {pid} in simultaneous node")
          chosen_actions.append(np.random.choice(legal_actions))
        print("Chosen actions: ", [
            state.action_to_string(pid, action)
            for pid, action in enumerate(chosen_actions)
        ])
        state.apply_actions(chosen_actions)
      else:
        # Decision node: sample action for the single current player
        legal_actions = state.legal_actions(state.current_player())
        if not legal_actions:
          print(f"\n ERROR: No legal actions for player {state.current_player()}",
                file=sys.stderr)
          print(f"Current state:\n{state}", file=sys.stderr)
          sys.exit(1)
        
        action = random.choice(legal_actions)
        action_string = state.action_to_string(state.current_player(), action)
        print("Player ", state.current_player(), ", randomly sampled action: ",
              action_string)
        state.apply_action(action)
      print(str(state))
    except Exception as e:
      print(f"\n ERROR during game play at move {move_count}: {e}",
            file=sys.stderr)
      print(f"Current state:\n{state}", file=sys.stderr)
      print("\nThis might be a bug in the game implementation.", file=sys.stderr)
      raise

  # Game is now done. Print utilities for each player
  print("\n" + "="*70)
  print(f" Game finished after {move_count} moves!")
  print("="*70)
  
  try:
    returns = state.returns()
    print("\n Final Results:")
    for pid in range(game.num_players()):
      result = returns[pid]
      if result > 0:
        outcome = "Won"
      elif result < 0:
        outcome = "Lost"
      else:
        outcome = "Draw"
      print(f"   Player {pid}: {result:+.2f} {outcome}")
  except Exception as e:
    print(f"\n  WARNING: Could not retrieve final returns: {e}",
          file=sys.stderr)
    print("The game finished but final scores are unavailable.", file=sys.stderr)


if __name__ == "__main__":
  app.run(main)
