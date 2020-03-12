from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

import numpy as np

"""
Test NE solver.
"""

# Games

# (1) Matching Pennies
MP_p1_meta_game = np.array([[1, -1], [-1, 1]])
MP_p2_meta_game = np.array([[-1, 1], [1, -1]])
MP_meta_games = [MP_p1_meta_game, MP_p2_meta_game]

#(2) Battle of Sexes
BOS_p1_meta_game = np.array([[3, 0], [0, 2]])
BOS_p2_meta_game = np.array([[2, 0], [0, 3]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]

#(3) Bar Crowding Game (3 players)
BC_p1_meta_game = np.array([[[-1, 2],[1, 1]], [[2, 0], [1, 1]]])
BC_p2_meta_game = np.array([[[-1, 1],[2, 1]], [[2, 1], [0, 1]]])
BC_p3_meta_game = np.array([[[-1, 2],[2, 0]], [[1, 1], [1, 1]]])
BC_meta_games = [BC_p1_meta_game, BC_p2_meta_game, BC_p3_meta_game]

#(4) Rock Paper Scissors
RPS_p1_meta_game = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
RPS_p2_meta_game = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
RPS_meta_games = [RPS_p1_meta_game, RPS_p2_meta_game]

#(5) Hunt Games
HT_p1_meta_game = np.array([[4, 1], [3, 2]])
HT_p2_meta_game = np.array([[4, 3], [1, 2]])
HT_meta_games = [HT_p1_meta_game, HT_p2_meta_game]

game_name = 'HT'

if game_name == 'MP':
    meta_games = MP_meta_games
elif game_name == 'BOS':
    meta_games = BOS_meta_games
elif game_name == 'BC':
    meta_games = BC_meta_games
elif game_name == 'RPS':
    meta_games = RPS_meta_games
elif game_name == 'HT':
    meta_games = HT_meta_games
else:
    raise ValueError("Game does not exist.")


strategies = replicator_dynamics(meta_games)
print(strategies)