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

"""Computes a Best-Response policy using list of pure strategies (br_list).
The only legal actions available when computing the best response
 are the actions that are suggested by at least one of the
pure strategies

The goal if this file is to be the main entry-point for BR APIs in Python.

TODO(author2): Also include computation using the more efficient C++
`TabularBestResponse` implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from open_spiel.python import policy as pyspiel_policy


def _memoize_method(method):
    """Memoize a single-arg instance method using an on-object cache."""
    cache_name = "cache_" + method.__name__

    def wrap(self, arg):
        key = str(arg)
        cache = vars(self).setdefault(cache_name, {})
        if key not in cache:
            cache[key] = method(self, arg)
        return cache[key]

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


class BestResponsePolicy(pyspiel_policy.Policy):
    """Computes the best response to a specified strategy."""

    def __init__(self, game, br_list, player_id, policy, root_state=None):
        """Initializes the best-response calculation.

    Args:
      game: The game to analyze.
      player_id: The player id of the best-responder.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
    """
        self._br_list = br_list
        self._num_players = game.num_players()
        self._player_id = player_id
        self._policy = policy
        if root_state is None:
            root_state = game.new_initial_state()
        self._root_state = root_state
        self.infosets = self.info_sets(root_state)

    def info_sets(self, state):
        """Returns a dict of infostatekey to list of (state, cf_probability)."""
        infosets = collections.defaultdict(list)
        for s, p in self.decision_nodes(state):
            infosets[s.information_state_string(self._player_id)].append((s, p))
        return dict(infosets)

    def decision_nodes(self, parent_state):
        """Yields a (state, cf_prob) pair for each descendant decision node."""
        if not parent_state.is_terminal():
            if parent_state.current_player() == self._player_id:
                yield (parent_state, 1.0)
            for action, p_action in self.transitions(parent_state):
                for state, p_state in self.decision_nodes(parent_state.child(action)):
                    yield (state, p_state * p_action)

    def transitions(self, state):
        """Returns a list of (action, cf_prob) pairs from the specifed state."""
        if state.current_player() == self._player_id:
            # Counterfactual reach probabilities exclude the best-responder's actions,
            # hence return probability 1.0 for every action.

            br_policies = []
            for br in self._br_list:
                try:
                    br_policy = _policy_dict_at_state(br[state.current_player()], state)
                    # br_policy = br[state.current_player()][state]
                    br_policies.append(br_policy)
                except KeyError as err:
                    # print(f"player: {state.current_player()} transitions key error: {err}")
                    # if there is a key error that's because the BR didn't see that state so nothing is
                    # initialized at that infoset so we can just have an arbitrary policy
                    br_policy = {}
                    for action in state.legal_actions(state.current_player()):
                        br_policy[action] = 0
                    br_policy[0] = 1
                    br_policies.append(br_policy)
            # br_policies = [_policy_dict_at_state(br[state.current_player()], state) for br in self._br_list]
            trans = []
            for action in state.legal_actions():
                for br in br_policies:
                    if br[action] == 1:
                        trans.append((action, 1.0))
            trans = list(set(trans))
            return trans
            # return [(action, 1.0) for action in state.legal_actions()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        else:
            return list(self._policy.action_probabilities(state).items())

    @_memoize_method
    def value(self, state):
        """Returns the value of the specified state to the best-responder."""
        if state.is_terminal():
            return state.player_return(self._player_id)
        elif state.current_player() == self._player_id:
            action = self.best_response_action(state)
            return self.q_value(state, action)
        else:
            return sum(p * self.q_value(state, a) for a, p in self.transitions(state))

    def q_value(self, state, action):
        """Returns the value of the (state, action) to the best-responder."""
        return self.value(state.child(action))

    @_memoize_method
    def best_response_action(self, state, player_id=None):
        """Returns the best response for this information state."""
        if player_id is not None:
            infostate = state.information_state_string(player_id)
        # hack to get line 103 in exploitability_br_actions to work
        elif isinstance(state, str):
            infostate = state
        else:
            infostate = state.information_state_string(self._player_id)
        infoset = self.infosets[infostate]
        legal_actions = infoset[0][0].legal_actions()

        br_policies = []
        for br in self._br_list:
            try:
                if player_id is not None:
                    br_policy = _policy_dict_at_state(br[player_id], state)
                    # br_policy = br[player_id][state]
                else:
                    br_policy = _policy_dict_at_state(br[self._player_id], state)
                    # br_policy = br[state.current_player()][state]
                br_policies.append(br_policy)
            except KeyError as err:
                # print(f"player: {state.current_player()} best response action key error: {err}")
                # if there is a key error that's because the BR didn't see that state so nothing is
                # initialized at that infoset so we can just have an arbitrary policy
                br_policy = {}
                for action in state.legal_actions(state.current_player()):
                    br_policy[action] = 0
                br_policy[0] = 1
                br_policies.append(br_policy)

        # br_policies = [_policy_dict_at_state(br[state.current_player()], state) for br in self._br_list]
        br_legal_actions = []
        for action in legal_actions:
            for br in br_policies:
                if br[action] == 1:
                    br_legal_actions.append(action)
        br_legal_actions = list(set(br_legal_actions))

        # print(self._br_list)
        # print(br_policies)
        # print(br_legal_actions)
        # print(state)
        # print([self.q_value(s, br_legal_actions[0]) for s, cf_p in infoset])
        # Get actions from the first (state, cf_prob) pair in the infoset list.
        # Return the best action by counterfactual-reach-weighted state-value.
        return max(
            # infoset[0][0].legal_actions(),
            br_legal_actions,
            key=lambda a: sum(cf_p * self.q_value(s, a) for s, cf_p in infoset))

    def action_probabilities(self, state, player_id=None):
        """Returns the policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
        if player_id is None:
            player_id = state.current_player()
        return {self.best_response_action(state.information_state_string(player_id), state): 1}
