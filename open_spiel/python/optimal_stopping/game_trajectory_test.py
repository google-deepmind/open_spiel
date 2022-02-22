import random
import pyspiel
import open_spiel.python.games.optimal_stopping
import numpy as np

# game = pyspiel.load_game("python_iterated_prisoners_dilemma")
game = pyspiel.load_game("python_optimal_stopping")
game = pyspiel.convert_to_turn_based(game)
state = game.new_initial_state()
while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.is_chance_node():
        # Sample a chance event outcome.
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
    else:
        # The algorithm can pick an action based on an observation (fully observable
        # games) or an information state (information available for that player)
        # We arbitrarily select the first available action as an example.
        action = random.choice(legal_actions)
        # action = legal_actions[0]

    # print(f"state:{state}, action:{action}")
    state.apply_action(action)