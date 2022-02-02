import pyspiel
import numpy as np
import sys

def command_line_action(state):
	legal_actions = state.legal_actions()
	action_map = {state.action_to_string(action): action for action in legal_actions}
	action = -1
	while action not in legal_actions:
		print(f"Player {state.current_player()}: Chose action from {action_map}")
		action_str = input()
		try:
			action = action_map[str(action_str)]
		except KeyError:
			print("Invalid action")
			continue
	return action

game = pyspiel.load_game('quoridor(wall_count=2,board_size=5,num_players=4,ansi_color_output=True)')
print(f"Loading game {game}")
state = game.new_initial_state()
print(state)
while not state.is_terminal():
    action = command_line_action(state)
    print(f"Player {state.current_player()} make move : {state.action_to_string(action)}")
    state.apply_action(action)
    print(state)
    print(state.is_terminal())
