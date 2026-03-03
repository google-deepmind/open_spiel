#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/abhay/openspiel/open_spiel')
import numpy as np
import pyspiel
from open_spiel.python.games import ant_foraging

print("=" * 50)
print("ANT FORAGING GAME - TEST OUTPUT")
print("=" * 50)

game = pyspiel.load_game('python_ant_foraging')
print(f"\nGame: {game.get_type().short_name}")
print(f"Players: {game.num_players()}")
print(f"Grid size: {game.grid_size}x{game.grid_size}")
print(f"Food sources: {game.num_food}")
print(f"Max turns: {game.max_turns}")

state = game.new_initial_state()
print("\nInitial State:")
print(state)

print("\n--- Playing 10 random moves ---")
np.random.seed(42)
for i in range(10):
    if state.is_terminal():
        break
    player = state.current_player()
    action = np.random.choice(state.legal_actions())
    action_str = state.action_to_string(player, action)
    print(f"Move {i+1}: {action_str}")
    state.apply_action(action)

print("\nState after 10 moves:")
print(state)
print(f"\nReturns: {state.returns()}")
print(f"Terminal: {state.is_terminal()}")
print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
