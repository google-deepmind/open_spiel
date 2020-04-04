import numpy as np

from open_spiel.python.algorithms.psro_v2.eval_utils import regret

BOS_p1_meta_game = np.array([[3, 0, 9],
                             [0, 2, 9],
                             [9, 9, 9]])
BOS_p2_meta_game = np.array([[2, 0, 9],
                             [0, 3, 100],
                             [9, 9, 9]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]

_regret = regret(BOS_meta_games, 2)
print(_regret)
