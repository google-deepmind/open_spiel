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

import numpy as np
import pyspiel

# Indices in the information sets for the regrets and average policy sums.
_REGRET_INDEX = 0
_AVG_POLICY_INDEX = 1

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


class OutcomeSamplingSolver(object):
    """An implementation of outcome sampling MCCFR.

  Uses stochastically-weighted averaging.

  For details, see Chapter 4 (p. 49) of:
  http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
  (Lanctot, 2013. "Monte Carlo Sampling and Regret Minimization for Equilibrium
  Computation and Decision-Making in Games")
  """

    def __init__(self, game, br_list):
        self._game = game
        self._br_list = br_list
        self._infostates = {}  # infostate keys -> [regrets, avg strat]
        self._num_players = game.num_players()
        # This is the epsilon exploration factor. When sampling episodes, the
        # updating player will sampling according to expl * uniform + (1 - expl) *
        # current_policy.
        self._expl = 0.6

        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
                "MCCFR requires sequential games. If you're trying to run it " +
                "on a simultaneous (or normal-form) game, please first transform it " +
                "using turn_based_simultaneous_game.")

    def iteration(self):
        """Performs one iteration of outcome sampling.

    An iteration consists of one episode for each player as the update player.
    """
        for update_player in range(self._num_players):
            state = self._game.new_initial_state()
            self._episode(
                state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

    def _lookup_infostate_info(self, info_state_key, num_legal_actions):
        """Looks up an information set table for the given key.

    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.

    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape [num_legal_actions].
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
            # print(legal_actions, 'legal actions in policy callable')
            # legal_actions = state.legal_actions()

            infostate_info = self._lookup_infostate_info(info_state_key,
                                                         len(legal_actions))
            avstrat = (
                    infostate_info[_AVG_POLICY_INDEX] /
                    infostate_info[_AVG_POLICY_INDEX].sum())
            # print(avstrat, 'avstrat')
            # print(legal_actions, 'legal actions')
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

    def _episode(self, state, update_player, my_reach, opp_reach, sample_reach):
        """Runs an episode of outcome sampling.

    Args:
      state: the open spiel state to run from (will be modified in-place).
      update_player: the player to update regrets for (the other players update
        average strategies)
      my_reach: reach probability of the update player
      opp_reach: reach probability of all the opponents (including chance)
      sample_reach: reach probability of the sampling (behavior) policy

    Returns:
      A tuple of (util, reach_tail), where:
        - util is the utility of the update player divided by the sample reach
          of the trajectory, and
        - reach_tail is the product of all players' reach probabilities
          to the terminal state (from the state that was passed in).
    """
        if state.is_terminal():
            return state.player_return(update_player) / sample_reach, 1.0

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            state.apply_action(outcome)
            return self._episode(state, update_player, my_reach, opp_reach,
                                 sample_reach)

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

        num_legal_actions = len(legal_actions)
        infostate_info = self._lookup_infostate_info(info_state_key,
                                                     num_legal_actions)
        policy = self._regret_matching(infostate_info[_REGRET_INDEX],
                                       num_legal_actions)
        if cur_player == update_player:
            uniform_policy = (
                    np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
            sampling_policy = (
                    self._expl * uniform_policy + (1.0 - self._expl) * policy)
        else:
            sampling_policy = policy
        sampled_action_idx = np.random.choice(
            np.arange(num_legal_actions), p=sampling_policy)
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_action_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_action_idx]
        new_sample_reach = sample_reach * sampling_policy[sampled_action_idx]
        state.apply_action(legal_actions[sampled_action_idx])
        util, reach_tail = self._episode(state, update_player, new_my_reach,
                                         new_opp_reach, new_sample_reach)
        new_reach_tail = policy[sampled_action_idx] * reach_tail
        # The following updates are based on equations 4.9 - 4.15 (Sec 4.2) of
        # http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
        if cur_player == update_player:
            # update regrets. Note the w here already includes the sample reach of the
            # trajectory (from root to terminal) in util due to the base case above.
            w = util * opp_reach
            for action_idx in range(num_legal_actions):
                if action_idx == sampled_action_idx:
                    self._add_regret(info_state_key, action_idx,
                                     w * (reach_tail - new_reach_tail))
                else:
                    self._add_regret(info_state_key, action_idx, -w * new_reach_tail)
        else:
            # update avg strat
            for action_idx in range(num_legal_actions):
                self._add_avstrat(info_state_key, action_idx,
                                  opp_reach * policy[action_idx] / sample_reach)
        return util, new_reach_tail
