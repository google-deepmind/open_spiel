import numpy as np
import copy

# from open_spiel.python.algorithms.psro_v2.eval_utils import regret
#
# BOS_p1_meta_game = np.array([[3, 0, 9, 100],
#                              [0, 2, 9, 100],
#                              [9, 9, 9, 100],
#                              [100, 100, 100, 100]])
# BOS_p2_meta_game = np.array([[2, 0, 9, 100],
#                              [0, 3, 9, 100],
#                              [9, 9, 9, 100],
#                              [100, 100, 100, 100]])
# BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]
#
# _regret = regret(BOS_meta_games, 4)
# print(_regret)


for i in range(16):
    print(i%5)