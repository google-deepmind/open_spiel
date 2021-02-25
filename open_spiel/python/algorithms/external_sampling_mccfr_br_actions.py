# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np
import pyspiel

# Indices in the information sets for the regrets and average policy sums.
_REGRET_INDEX = 0
_AVG_POLICY_INDEX = 1


class AverageType(enum.Enum):
    SIMPLE = 0
    FULL = 1

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


class ExternalSamplingSolver(object):
    """An implementation of external sampling MCCFR."""

    def __init__(self, game, br_list, average_type=AverageType.SIMPLE):
        self._game = game
        self._infostates = {}  # infostate keys -> [regrets, avg strat]
        self._num_players = game.num_players()
        self._br_list = br_list

        # How to average the strategy. The 'simple' type does the averaging for
        # player i + 1 mod num_players on player i's regret update pass; in two
        # players this corresponds to the standard implementation (updating the
        # average policy at opponent nodes). In n>2 players, this can be a problem
        # for several reasons: first, it does not compute the estimate as described
        # by the (unbiased) stochastically-weighted averaging in chapter 4 of
        # Lanctot 2013  commonly used in MCCFR because the denominator (important
        # sampling correction) should include all the other sampled players as well
        # so the sample reach no longer cancels with reach of the player updating
        # their average policy. Second, if one player assigns zero probability to an
        # action (leading to a subtree), the average policy of a different player in
        # that subtree is no longer updated. Hence, the full averaging does not
        # update the average policy in the regret passes but does a separate pass to
        # update the average policy. Nevertheless, we set the simple type as the
        # default because it is faster, seems to work better empirically, and it
        # matches what was done in Pluribus (Brown and Sandholm. Superhuman AI for
        # multiplayer poker. Science, 11, 2019).
        self._average_type = average_type

        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
                "MCCFR requires sequential games. If you're trying to run it " +
                'on a simultaneous (or normal-form) game, please first transform it ' +
                'using turn_based_simultaneous_game.')

    def iteration(self):
        """Performs one iteration of external sampling.
    An iteration consists of one episode for each player as the update
    player.
    """
        for player in range(self._num_players):
            self._update_regrets(self._game.new_initial_state(), player)
        if self._average_type == AverageType.FULL:
            reach_probs = np.ones(self._num_players, dtype=np.float64)
            self._full_update_average(self._game.new_initial_state(), reach_probs)

    def _lookup_infostate_info(self, info_state_key, num_legal_actions):
        """Looks up an information set table for the given key.
    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.
    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape
        [num_legal_actions].
          The average is weighted using `my_reach`
    """
        retrieved_infostate = self._infostates.get(info_state_key, None)
        if retrieved_infostate is not None:
            return retrieved_infostate

        # Start with a small amount of regret and total accumulation, to give a
        # uniform policy: this will get erased fast.
        self._infostates[info_state_key] = [
            np.ones(num_legal_actions, dtype=np.float64) / 1000.0,
            np.ones(num_legal_actions, dtype=np.float64) / 1000.0,
        ]
        return self._infostates[info_state_key]

    def _add_regret(self, info_state_key, action_idx, amount):
        self._infostates[info_state_key][_REGRET_INDEX][action_idx] += amount

    def _add_avstrat(self, info_state_key, action_idx, amount):
        self._infostates[info_state_key][_AVG_POLICY_INDEX][action_idx] += amount

    def callable_avg_policy(self):
        """Returns the average joint policy as a callable.
    The callable has a signature of the form string (information
    state key) -> list of (action, prob).
    """

        def wrap(state):
            info_state_key = state.information_state(state.current_player())

            br_policies = []
            for br in self._br_list:
                try:
                    br_policy = _policy_dict_at_state(br[state.current_player()], state)
                    br_policies.append(br_policy)
                except KeyError as err:
                    # print(f"player: {state.current_player()} key error: {err}")
                    # if there is a key error that's because the BR didn't see that state so nothing is
                    # initialized at that infoset so we can just have an arbitrary policy
                    br_policy = {}
                    for action in state.legal_actions(state.current_player()):
                        br_policy[action] = 0
                    br_policy[0] = 1
                    br_policies.append(br_policy)
            br_actions = []
            for action in state.legal_actions():
                for br_policy in br_policies:
                    if br_policy[action] == 1:
                        br_actions.append(action)
            legal_actions = list(set(br_actions))


            # legal_actions = state.legal_actions()
            infostate_info = self._lookup_infostate_info(info_state_key,
                                                         len(legal_actions))
            avstrat = (
                    infostate_info[_AVG_POLICY_INDEX] /
                    infostate_info[_AVG_POLICY_INDEX].sum())
            return [(legal_actions[i], avstrat[i]) for i in range(len(legal_actions))]

        return wrap

    def _regret_matching(self, regrets, num_legal_actions):
        """Applies regret matching to get a policy.
    Args:
      regrets: numpy array of regrets for each action.
      num_legal_actions: number of legal actions at this state.
    Returns:
      numpy array of the policy indexed by the index of legal action in the
      list.
    """
        positive_regrets = np.maximum(regrets,
                                      np.zeros(num_legal_actions, dtype=np.float64))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
        else:
            return positive_regrets / sum_pos_regret

    def _full_update_average(self, state, reach_probs):
        """Performs a full update average.
    Args:
      state: the open spiel state to run from
      reach_probs: array containing the probability of reaching the state
        from the players point of view
    """
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action in state.legal_actions():
                self._full_update_average(state.child(action), reach_probs)
            return

        # If all the probs are zero, no need to keep going.
        sum_reach_probs = np.sum(reach_probs)
        if sum_reach_probs == 0:
            return

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)

        br_policies = []
        for br in self._br_list:
            try:
                br_policy = _policy_dict_at_state(br[cur_player], state)
                br_policies.append(br_policy)
            except KeyError as err:
                # print(f"player: {cur_player} key error: {err}")
                # if there is a key error that's because the BR didn't see that state so nothing is
                # initialized at that infoset so we can just have an arbitrary policy
                br_policy = {}
                for action in state.legal_actions(cur_player):
                    br_policy[action] = 0
                br_policy[0] = 1
                br_policies.append(br_policy)
        br_actions = []
        for action in state.legal_actions():
            for br_policy in br_policies:
                if br_policy[action] == 1:
                    br_actions.append(action)
        legal_actions = list(set(br_actions))


        # legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        infostate_info = self._lookup_infostate_info(info_state_key,
                                                     num_legal_actions)
        policy = self._regret_matching(infostate_info[_REGRET_INDEX],
                                       num_legal_actions)

        for action_idx in range(num_legal_actions):
            new_reach_probs = np.copy(reach_probs)
            new_reach_probs[cur_player] *= policy[action_idx]
            self._full_update_average(
                state.child(legal_actions[action_idx]), new_reach_probs)

        # Now update the cumulative policy
        for action_idx in range(num_legal_actions):
            self._add_avstrat(info_state_key, action_idx,
                              reach_probs[cur_player] * policy[action_idx])

    def _update_regrets(self, state, player):
        """Runs an episode of external sampling.
    Args:
      state: the open spiel state to run from
      player: the player to update regrets for
    Returns:
      value: is the value of the state in the game
      obtained as the weighted average of the values
      of the children
    """
        if state.is_terminal():
            return state.player_return(player)

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            return self._update_regrets(state.child(outcome), player)

        cur_player = state.current_player()
        info_state_key = state.information_state(cur_player)

        br_policies = []
        for br in self._br_list:
            try:
                br_policy = _policy_dict_at_state(br[cur_player], state)
                br_policies.append(br_policy)
            except KeyError as err:
                # print(f"player: {cur_player} key error: {err}")
                # if there is a key error that's because the BR didn't see that state so nothing is
                # initialized at that infoset so we can just have an arbitrary policy
                br_policy = {}
                for action in state.legal_actions(cur_player):
                    br_policy[action] = 0
                br_policy[0] = 1
                br_policies.append(br_policy)
        br_actions = []
        for action in state.legal_actions():
            for br_policy in br_policies:
                if br_policy[action] == 1:
                    br_actions.append(action)
        legal_actions = list(set(br_actions))


        # legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        infostate_info = self._lookup_infostate_info(info_state_key,
                                                     num_legal_actions)
        policy = self._regret_matching(infostate_info[_REGRET_INDEX],
                                       num_legal_actions)

        value = 0
        child_values = np.zeros(num_legal_actions, dtype=np.float64)
        if cur_player != player:
            # Sample at opponent node
            action_idx = np.random.choice(np.arange(num_legal_actions), p=policy)
            value = self._update_regrets(
                state.child(legal_actions[action_idx]), player)
        else:
            # Walk over all actions at my node
            for action_idx in range(num_legal_actions):
                child_values[action_idx] = self._update_regrets(
                    state.child(legal_actions[action_idx]), player)
                value += policy[action_idx] * child_values[action_idx]

        if cur_player == player:
            # Update regrets.
            for action_idx in range(num_legal_actions):
                self._add_regret(info_state_key, action_idx,
                                 child_values[action_idx] - value)
        # Simple average does averaging on the opponent node. To do this in a game
        # with more than two players, we only update the player + 1 mod num_players,
        # which reduces to the standard rule in 2 players.
        if self._average_type == AverageType.SIMPLE and cur_player == (
                player + 1) % self._num_players:
            for action_idx in range(num_legal_actions):
                self._add_avstrat(info_state_key, action_idx, policy[action_idx])

        return value
