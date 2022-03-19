import random
import pyspiel
import numpy as np
import time
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig


params = OptimalStoppingGameConfig.default_params()
params["use_beliefs"] = True
params["T_max"] = 6
game = pyspiel.load_game("python_optimal_stopping_game_sequential", params)
state = game.new_initial_state()
while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.is_chance_node():
        # Sample a chance event outcome.
        outcomes_with_probs = state.chance_outcomes()
        print(f"chance outcomes:{state.chance_outcomes()}")
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
    else:
        # The algorithm can pick an action based on an observation (fully observable
        # games) or an information state (information available for that player)
        # We arbitrarily select the first available action as an example.
        action = random.choice(legal_actions)
        # action = legal_actions[0]

    state.apply_action(action)
    print(f"state:{state.history()}, rewards: {state._rewards}")