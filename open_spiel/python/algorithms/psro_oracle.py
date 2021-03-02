from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import attr
import numpy as np

from open_spiel.python import policy
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import lp_solver
from numpy import array

"""
Implements PSRO with an oracle best response
"""


@attr.s
class _InfoStateNode(object):
    """An object wrapping values associated to an information state."""
    # The list of the legal actions.
    legal_actions = attr.ib()
    index_in_tabular_policy = attr.ib()
    # Map from information states string representations and actions to the
    # counterfactual regrets, accumulated over the policy iterations
    cumulative_regret = attr.ib(factory=lambda: collections.defaultdict(float))
    # Same as above for the cumulative of the policy probabilities computed
    # during the policy iterations
    cumulative_policy = attr.ib(factory=lambda: collections.defaultdict(float))

def _full_best_response_policy(br_infoset_dict):
    """Turns a dictionary of best response action selections into a full policy.

  Args:
    br_infoset_dict: A dictionary mapping information state to a best response
      action.

  Returns:
    A function `state` -> list of (action, prob)
  """

    def wrap(state):
        infostate_key = state.information_state_string(state.current_player())
        br_action = br_infoset_dict[infostate_key]
        ap_list = []
        for action in state.legal_actions():
            ap_list.append((action, 1.0 if action == br_action else 0.0))
        return ap_list

    return wrap

def _policy_dict_at_state(callable_policy, state):
    """Turns a policy function into a dictionary at a specific state.

Args:
  callable_policy: A function from `state` -> lis of (action, prob),
  state: the specific state to extract the policy from.

Returns:
  A dictionary of action -> prob at this state.
"""

    infostate_policy_list = callable_policy(state)
    infostate_policy = {}
    for ap in infostate_policy_list:
        infostate_policy[ap[0]] = ap[1]
    return infostate_policy


def sample_episode(state, brs):
    """Samples an episode according to the policies, starting from state.

  Args:
    state: Pyspiel state representing the current state.
    policies: List of policy representing the policy executed by each player.

  Returns:
    The result of the call to returns() of the final state in the episode.
        Meant to be a win/loss integer.
  """
    if state.is_terminal():
        return np.array(state.returns(), dtype=np.float32)

    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
    else:
        player = state.current_player()
        br = brs[player]
        try:
            br_policy = _policy_dict_at_state(br, state)
        except:
            # if there is a key error then just have arbitrary policy
            br_policy = {}
            for action in state.legal_actions(player):
                br_policy[action] = 1/len(state.legal_actions(player))
        outcomes, probs = zip(*br_policy.items())

    state.apply_action(np.random.choice(outcomes, p=probs))
    return sample_episode(state, brs)


def sample_episodes(brs, num_episodes, game):
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num_episodes: Number of episodes to execute to estimate average return of
        policies.

    Returns:
      Average episode return over num episodes.
    """
    totals = np.zeros(2)
    for _ in range(num_episodes):
        totals += sample_episode(game.new_initial_state(),
                                 brs).reshape(-1)
    return totals / num_episodes


def get_br_to_strat(strat, payoffs, strat_is_row=True, verbose=False):
    if strat_is_row:
        weighted_payouts = strat @ payoffs
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmin(weighted_payouts)] = 1
        idx = np.argmin(weighted_payouts)
    else:
        weighted_payouts = payoffs @ strat.T
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmax(weighted_payouts)] = 1
        idx = np.argmax(weighted_payouts)
    return br, idx


def fictitious_play(payoffs, iters=2000, verbose=False):
    row_dim = payoffs.shape[0]
    col_dim = payoffs.shape[1]
    row_pop = np.random.uniform(0, 1, (1, row_dim))
    row_pop = row_pop / row_pop.sum(axis=1)[:, None]
    row_averages = row_pop
    col_pop = np.random.uniform(0, 1, (1, col_dim))
    col_pop = col_pop / col_pop.sum(axis=1)[:, None]
    col_averages = col_pop
    exps = []
    for i in range(iters):
        row_average = np.average(row_pop, axis=0)
        col_average = np.average(col_pop, axis=0)

        row_br, idx = get_br_to_strat(col_average, payoffs, strat_is_row=False, verbose=False)
        col_br, idx = get_br_to_strat(row_average, payoffs, strat_is_row=True, verbose=False)

        exp1 = row_average @ payoffs @ col_br.T
        exp2 = row_br @ payoffs @ col_average.T
        exps.append(exp2 - exp1)
        if verbose:
            print(exps[-1], "exploitability")

        row_averages = np.vstack((row_averages, row_average))
        col_averages = np.vstack((col_averages, col_average))

        row_pop = np.vstack((row_pop, row_br))
        col_pop = np.vstack((col_pop, col_br))
    return row_averages, col_averages, exps


def _get_br_to_policy(game, policy):
    brs = []
    for i in range(2):
        br_info = exploitability.best_response(game, policy, i)
        full_br_policy = _full_best_response_policy(br_info["best_response_action"])
        brs.append(full_br_policy)
    return brs

# Does extensive form version of double oracle algorithm
# at each information set, the mixed strategy is a reach-weighted mixture of the strategies in the population
# since we have best responses, this just means a renormalized distribution of the strategies
# that would have taken moves to arrive at that infoset

# br_list is list of four lists. First list is br pop for player 1 (0), second list is list of probs for those brs
# third list is br pop for player 2 (1), fourth is list of probs for those brs

# available brs is list of two lists. First is list of indices of best responses available at that state for player 1 (0)
# second is list of indices of best responses at that state for player 2 (1)

class PSRO(object):
    def __init__(self, game, br_list, num_episodes=100):
        self._current_policy = policy.TabularPolicy(game)
        self._info_state_nodes = {}
        self._game = game
        self._num_episodes = num_episodes
        self._br_list = br_list
        # self._br_list = [[], [], [], []]
        self._root_node = self._game.new_initial_state()
        self._initialize_info_state_nodes(self._root_node)
        # self._update_policy(self._root_node, [[i for i in range(len(self._br_list[0]))],
        #                                       [i for i in range(len(self._br_list[2]))]])
        self._update_policy(self._root_node, [[], []])
        self._iterations = 0

    def evaluate(self):
        self._get_new_meta_nash_probs()
        self._update_policy(self._root_node, [[i for i in range(len(self._br_list[0]))],
                                              [i for i in range(len(self._br_list[2]))]])

    def iteration(self):
        self._add_brs_to_current_policy()
        self._get_new_meta_nash_probs()
        self._update_policy(self._root_node, [[i for i in range(len(self._br_list[0]))],
                                              [i for i in range(len(self._br_list[2]))]])
        self._iterations += 1

    def _initialize_info_state_nodes(self, state):
        """Initializes info_state_nodes.

    Create one _InfoStateNode per infoset. We could also initialize the node
    when we try to access it and it does not exist.

    Args:
      state: The current state in the tree walk. This should be the root node
        when we call this function from a CFR solver.
    """
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                self._initialize_info_state_nodes(state.child(action))
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        info_state_node = self._info_state_nodes.get(info_state)
        if info_state_node is None:
            legal_actions = state.legal_actions(current_player)
            info_state_node = _InfoStateNode(
                legal_actions=legal_actions,
                index_in_tabular_policy=self._current_policy.state_lookup[info_state])
            self._info_state_nodes[info_state] = info_state_node

        for action in info_state_node.legal_actions:
            self._initialize_info_state_nodes(state.child(action))

    # available brs is a list of 2 lists. The first list is a list of indices of available brs for player 0, the second
    # is a list of indices of available brs for player 1
    def _update_policy(self, state: object, available_brs: object) -> object:
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                # print('Chance node', action, state)
                self._update_policy(state.child(action), available_brs)
            return

        player = state.current_player()
        legal_actions = state.legal_actions(player)

        # available_brs is list of two lists, avail br indices for each player
        if len(available_brs[player]) == 0:
            # can choose arbitrary strategy, here I use uniform
            info_state = state.information_state_string(player)
            state_policy = self._current_policy.policy_for_key(info_state)
            for action in legal_actions:
                state_policy[action] = 1 / len(legal_actions)
                self._update_policy(state.child(action), available_brs)
            return

        available_brs_player_1 = [self._br_list[0][i] for i in available_brs[0]]
        available_brs_player_2 = [self._br_list[2][i] for i in available_brs[1]]

        probs_player_1 = [self._br_list[1][i] for i in available_brs[0]]
        probs_player_2 = [self._br_list[3][i] for i in available_brs[1]]

        if player == 0:
            brs = available_brs_player_1
            probs = probs_player_1
        else:
            brs = available_brs_player_2
            probs = probs_player_2

        # update policy at this state
        br_policies = []
        br_policies_probs = []
        br_indices = []
        for i in range(len(brs)):
            br = brs[i]
            try:
                br_policy = _policy_dict_at_state(br, state)
                br_policies.append(br_policy)
                br_policies_probs.append(probs[i])
                br_indices.append(available_brs[player][i])
            except KeyError as err:
                # print(f"player: {player} key error: {err}")
                # if there is a key error that's because the BR didn't see that state so nothing is
                # initialized at that infoset so we can just have an arbitrary policy
                br_policy = {}
                for action in state.legal_actions(player):
                    br_policy[action] = 0
                br_policy[0] = 1
                br_policies.append(br_policy)
                br_policies_probs.append(probs[i])
                br_indices.append(available_brs[player][i])

        assert len(br_policies_probs) == len(br_policies)

        norm_br_policies_probs = [float(i) / sum(br_policies_probs) for i in br_policies_probs]

        info_state = state.information_state_string(player)
        state_policy = self._current_policy.policy_for_key(info_state)
        for action in legal_actions:
            if len(br_policies_probs) == 0:
                state_policy[action] = 1/len(legal_actions)
            else:
                action_prob = 0
                for i in range(len(br_policies)):
                    br_policy = br_policies[i]
                    if br_policy[action] == 1:
                        action_prob += norm_br_policies_probs[i]
                state_policy[action] = action_prob

        # update policy for children
        # just need new available brs which only change for current player
        new_available_brs_for_player = []
        for action in legal_actions:
            for i in range(len(br_policies)):
                br_policy = br_policies[i]
                if br_policy[action] == 1:
                    new_available_brs_for_player.append(br_indices[i])
            new_available_brs = available_brs.copy()
            new_available_brs[player] = new_available_brs_for_player
            self._update_policy(state.child(action), new_available_brs)

    def _add_brs_to_current_policy(self):
        # get brs and add them to br list
        # initialize probs for new brs to 0 until meta Nash calculation
        brs = _get_br_to_policy(self._game, self._current_policy)
        new_br_list = self._br_list.copy()
        new_br_list[0].append(brs[0])
        new_br_list[1].append(0)
        new_br_list[2].append(brs[1])
        new_br_list[3].append(0)
        self._br_list = new_br_list.copy()

    def _get_new_meta_nash_probs(self):
        num_row_strats = len(self._br_list[0])
        num_col_strats = len(self._br_list[2])
        emperical_game_matrix = np.zeros((num_row_strats, num_col_strats))
        for i in range(num_row_strats):
            for j in range(num_col_strats):
                row_strat = self._br_list[0][i]
                col_strat = self._br_list[2][j]
                avg_return = sample_episodes([row_strat, col_strat], self._num_episodes, self._game)
                # avg return array of payoffs for each player, since zero sum, just keep track of row player (player 0)
                emperical_game_matrix[i, j] = avg_return[0]
        nash_prob_1, nash_prob_2, _, _ = lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game(emperical_game_matrix, -emperical_game_matrix))
        norm_pos1 = abs(nash_prob_1) / sum(abs(nash_prob_1))
        norm_pos2 = abs(nash_prob_2) / sum(abs(nash_prob_2))
        self._br_list[1] = np.squeeze(array(norm_pos1)).tolist()
        self._br_list[3] = np.squeeze(array(norm_pos2)).tolist()