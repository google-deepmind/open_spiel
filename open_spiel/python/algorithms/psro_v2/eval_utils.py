import numpy as np
from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver

def regret(meta_games, subgame_index):
    """
    Calculate the regret based on a complete payoff matrix for PSRO.
    Assume all players have the same number of policies.
    :param meta_games: meta_games in PSRO
    :param subgame_index: the subgame to evaluate.
    :return: a list of regret, one for each player.
    """
    num_policy = np.shape(meta_games[0])[0]
    num_players = len(meta_games)
    if num_policy == subgame_index:
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)

    index = [slice(0, subgame_index) for _ in range(num_players)]
    submeta_games = [subgame[tuple(index)] for subgame in meta_games]
    nash = nash_solver(submeta_games, solver="gambit")

    nash_payoffs = []
    deviation_payoffs = []

    for current_player in range(num_players):
        meta_game = submeta_games[current_player]
        for dim in range(num_players):
            newshape = -np.ones(num_players, dtype=np.int64)
            newshape[dim] = len(nash[dim])
            meta_game = np.reshape(nash[dim], newshape=newshape) * meta_game

        nash_payoff = np.sum(meta_game)
        nash_payoffs.append(nash_payoff)

    extended_nash = []
    for dist in nash:
        ex_nash = np.zeros(num_policy)
        ex_nash[:len(dist)] = dist
        extended_nash.append(ex_nash)

    for current_player in range(num_players):
        _meta_game = meta_games[current_player]
        for player in range(num_players):
            if current_player == player:
                continue
            newshape = -np.ones(num_players, dtype=np.int64)
            newshape[player] = num_policy
            _meta_game = np.reshape(extended_nash[player], newshape=newshape) * _meta_game

        axis = np.delete(np.arange(num_players), current_player)
        deviation = np.max(np.sum(_meta_game, axis=tuple(axis)))
        deviation_payoffs.append(deviation)

    regret = np.maximum(np.array(deviation_payoffs)-np.array(nash_payoffs), 0)

    return regret


class SElogs(object):
    def __init__(self,
                 slow_oracle_period,
                 fast_oracle_period,
                 meta_strategy_methods):

        self.slow_oracle_period = slow_oracle_period
        self.fast_oracle_period = fast_oracle_period
        self.meta_strategy_methods = meta_strategy_methods

        self._slow_oracle_iters = []
        self._fast_oracle_iters = []

        self.regrets = []
        self.nashconv = []

    def update_regrets(self, regrets):
        self.regrets.append(regrets)

    def get_regrets(self):
        return self.regrets

    def update_nashconv(self, nashconv):
        self.nashconv.append(nashconv)

    def get_nashconv(self):
        return self.nashconv

    def update_slow_iters(self, iter):
        self._slow_oracle_iters.append(iter)

    def get_slow_iters(self):
        return self._slow_oracle_iters

    def update_fast_iters(self, iter):
        self._fast_oracle_iters.append(iter)

    def get_fast_iters(self):
        return self._fast_oracle_iters