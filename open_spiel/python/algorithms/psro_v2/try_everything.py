import numpy as np

# BOS_p1_meta_game = np.array([[3, 0], [0, 2]])
# BOS_p2_meta_game = np.array([[2, 0], [0, 3]])
# BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]
#
#
# meta_games = BOS_meta_games
# nash = [np.array([0.5, 0.5]), np.array([1, 0])]
#
# num_players = len(meta_games)
# for current_player in range(num_players):
#     meta_game = np.array(meta_games[current_player])
#     for dim in range(num_players):
#         newshape = -np.ones(num_players, dtype=np.int64)
#         newshape[dim] = len(nash[dim])
#         meta_game = np.reshape(nash[dim], newshape=newshape) * meta_game
#         print(meta_game)
#
#     print(np.sum(meta_game))
#     print("***************")


a = np.array([1,2,3,4])
print(np.maximum(a, 3))