import pyspiel
import numpy as np

game = pyspiel.load_game('quoridor(wall_count=0,board_size=5,num_players=4)')
print(f"Loading game {game}")
state = game.new_initial_state()
print(state)
while not state.is_terminal():
    legal_actions = state.legal_actions()
    action = np.random.choice(state.legal_actions())
    print(f"Player {state.current_player()} make move : {state.action_to_string(action)}")
    state.apply_action(action)
    print(state)
    __import__('pdb').set_trace()
    print(state.is_terminal())
