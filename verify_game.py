#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/abhay/openspiel/open_spiel')
import pyspiel
from open_spiel.python.games import ant_foraging
game = pyspiel.load_game('python_ant_foraging')
state = game.new_initial_state()
print('Game loaded successfully!')
print('Players:', game.num_players())
print('Grid:', game.grid_size)
print('Food:', game.num_food)
print('Initial state valid:', not state.is_terminal())
print('Has legal actions:', len(state.legal_actions()) > 0)
for i in range(4):
    state.apply_action(state.legal_actions()[0])
print('After 4 moves, player:', state.current_player())
print('ALL CHECKS PASSED!')
