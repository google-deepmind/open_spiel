# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: 2023 James Flynn
# Original: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py
"""Python implementation of the extensive-form regret minimization algorithm.

One iteration of EFR consists of:
1) Compute current strategy from regrets (e.g. using Regret Matching).
2) Compute values using the current strategy
3) Compute regrets from these values

The average policy converges to a Nash Equilibrium rather than the current policy as in CFR.
"""
import copy
from collections import defaultdict
import attr

import numpy as np
from scipy.linalg import lstsq

import pyspiel
from open_spiel.python import policy


@attr.s
class _InfoStateNode(object):
    """An object wrapping values associated to an information state."""
    # The list of the legal actions.
    legal_actions = attr.ib()
    index_in_tabular_policy = attr.ib()
    # The newly availible deviations + the old ones
    relizable_deviations = attr.ib()
    # Player -> state -> action -> prob
    current_history_probs = attr.ib()

    # An array representing
    history = attr.ib()

    cumulative_regret = attr.ib(factory=lambda: defaultdict(float))
    # Same as above for the cumulative of the policy probabilities computed
    # during the policy iterations
    cumulative_policy = attr.ib(factory=lambda: defaultdict(float))
    y_values = attr.ib(factory=lambda: defaultdict(float))


class _EFRSolverBase(object):
    def __init__(self, game, _deviation_gen):
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, ()

        self._game = game
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()

        # This is for returning the current policy and average policy to a caller
        self._current_policy = policy.TabularPolicy(game)
        self._average_policy = self._current_policy.__copy__()
        self._deviation_gen = _deviation_gen

        self._info_state_nodes = {}
        hist = {player: [] for player in range(self._num_players)}
        unif_probs = [[] for _ in range(self._num_players)],
        empty_path_indices = [[] for _ in range(self._num_players)]
        self._initialize_info_state_nodes(
            self._root_node, hist, unif_probs, empty_path_indices)

        self._iteration = 1  # For possible linear-averaging.

    def return_cumulative_regret(self):
        return {list(self._info_state_nodes.keys())[i]: list(self._info_state_nodes.values())[i].cumulative_regret for i in range(len(self._info_state_nodes.keys()))}

    def current_policy(self):
        return self._current_policy

    def average_policy(self):
        _update_average_policy(self._average_policy, self._info_state_nodes)
        return self._average_policy

    def _initialize_info_state_nodes(self, state, history, uniform_probs_to_state, path_indices):
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                self._initialize_info_state_nodes(state.child(
                    action), history, uniform_probs_to_state, path_indices)
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        info_state_node = self._info_state_nodes.get(info_state)
        if info_state_node is None:
            legal_actions = state.legal_actions(current_player)
            info_state_node = _InfoStateNode(
                legal_actions=legal_actions,
                index_in_tabular_policy=self._current_policy.state_lookup[info_state],
                relizable_deviations=None,
                history=history[current_player].copy(),
                current_history_probs=copy.deepcopy(
                    path_indices[current_player])
            )
            prior_possible_actions = []
            for i in range(len(info_state_node.current_history_probs)):
                prior_possible_actions.append(
                    info_state_node.current_history_probs[i][0])
            prior_possible_actions.append(info_state_node.legal_actions)

            info_state_node.relizable_deviations = self._deviation_gen(len(
                info_state_node.legal_actions), info_state_node.history, prior_possible_actions)
            self._info_state_nodes[info_state] = info_state_node

        legal_actions = state.legal_actions(current_player)
        new_uniform_probs_to_state = copy.deepcopy(uniform_probs_to_state)
        assert len(new_uniform_probs_to_state[current_player]) == len(
            history[current_player])

        new_uniform_probs_to_state[current_player].append(
            {legal_actions[i]: 1/len(legal_actions) for i in range(len(legal_actions))})
        for action in info_state_node.legal_actions:
            # Speedup
            new_path_indices = copy.deepcopy(path_indices)
            new_path_indices[current_player].append(
                [legal_actions, info_state_node.index_in_tabular_policy])
            # Speedup
            new_history = copy.deepcopy(history)
            new_history[current_player].append(action)
            assert len(new_history[current_player]) == len(
                new_path_indices[current_player])

            self._initialize_info_state_nodes(state.child(
                action), new_history, new_uniform_probs_to_state, new_path_indices)

    def _update_current_policy(self, state, current_policy):
        """Updated in order so that memory reach probs are defined wrt to the new strategy
        """

        if state.is_terminal():
            return
        elif not state.is_chance_node():
            current_player = state.current_player()
            info_state = state.information_state_string(current_player)
            info_state_node = self._info_state_nodes[info_state]
            deviations = info_state_node.relizable_deviations
            # print(info_state)
            for devation in range(len(deviations)):
                # change too infostate
                mem_reach_probs = create_probs_from_index(
                    info_state_node.current_history_probs, current_policy)
                deviation_reach_prob = deviations[devation].player_deviation_reach_probability(
                    mem_reach_probs)
                info_state_node.y_values[deviations[devation]] = info_state_node.y_values[deviations[devation]] + max(
                    0, info_state_node.cumulative_regret[devation])*deviation_reach_prob

            # Might be incorrect
            state_policy = current_policy.policy_for_key(info_state)
            # print
            for action, value in self._regret_matching(info_state_node.legal_actions, info_state_node).items():
                state_policy[action] = value

            for action in info_state_node.legal_actions:
                new_state = state.child(action)
                self._update_current_policy(new_state, current_policy)
        else:
            for action, _ in state.chance_outcomes():
                new_state = state.child(action)
                self._update_current_policy(new_state, current_policy)
    # Path to state probability ignores chance probabilty as this is stored as new_reach_probabilities[-1]

    def _compute_cumulative_immediate_regret_for_player(self, state, policies,
                                                        reach_probabilities, player):
        if state.is_terminal():
            return np.asarray(state.returns())

        if state.is_chance_node():
            state_value = 0.0
            for action, action_prob in state.chance_outcomes():
                assert action_prob > 0
                new_state = state.child(action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob

                state_value += action_prob * self._compute_cumulative_immediate_regret_for_player(
                    new_state, policies, new_reach_probabilities, player)
            return state_value

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        # No need to continue on this history branch as no update will be performed
        # for any player.
        # The value we return here is not used in practice. If the conditional
        # statement is True, then the last taken action has probability 0 of
        # occurring, so the returned value is not impacting the parent node value.
        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        state_value = np.zeros(self._num_players)

        # The utilities of the children states are computed recursively. As the
        # regrets are added to the information state regrets for each state in that
        # information state, the recursive call can only be made once per child
        # state. Therefore, the utilities are cached.
        children_utilities = {}

        info_state_node = self._info_state_nodes[info_state]
        # Reset y values
        info_state_node.y_values = defaultdict(float)
        if policies is None:
            info_state_policy = self._get_infostate_policy(info_state)
        else:
            info_state_policy = policies[current_player](info_state)

        reach_prob = reach_probabilities[current_player]
        for action in state.legal_actions():
            action_prob = info_state_policy.get(action, 0.)
            info_state_node.cumulative_policy[action] = info_state_node.cumulative_policy[action] + \
                action_prob * reach_prob
            new_state = state.child(action)
            new_reach_probabilities = reach_probabilities.copy()
            assert action_prob <= 1
            new_reach_probabilities[current_player] *= action_prob
            child_utility = self._compute_cumulative_immediate_regret_for_player(
                new_state, policies=policies, reach_probabilities=new_reach_probabilities, player=player)

            state_value += action_prob * child_utility
            children_utilities[action] = child_utility

        counterfactual_reach_prob = (np.prod(
            reach_probabilities[:current_player]) * np.prod(reach_probabilities[current_player + 1:]))

        state_value_for_player = state_value[current_player]
        deviations = info_state_node.relizable_deviations
        for deviation_index in range(len(deviations)):
            # FIX ADD DICT TO ARRAY CONVERSION FUNCTION
            deviation = deviations[deviation_index]
            deviation_strategy = deviation.deviate(
                strat_dict_to_array(self._get_infostate_policy(info_state)))

            player_child_utilities = np.array(list(children_utilities.values()))[
                :, current_player]
            devation_cf_value = np.inner(np.transpose(
                deviation_strategy), player_child_utilities)

            memory_reach_probs = create_probs_from_index(
                info_state_node.current_history_probs, self.current_policy())
            player_current_memory_reach_prob = deviation.player_deviation_reach_probability(
                memory_reach_probs)

            deviation_regret = player_current_memory_reach_prob * \
                ((devation_cf_value*counterfactual_reach_prob) -
                 (counterfactual_reach_prob * state_value_for_player))

            info_state_node.cumulative_regret[deviation_index] += deviation_regret
        return state_value

    def _get_infostate_policy(self, info_state_str):
        """Returns an {action: prob} dictionary for the policy on `info_state`."""
        info_state_node = self._info_state_nodes[info_state_str]
        prob_vec = self._current_policy.action_probability_array[
            info_state_node.index_in_tabular_policy]
        return {
            action: prob_vec[action] for action in info_state_node.legal_actions
        }


def __get_infostate_policy_array(self, info_state_str):
    info_state_node = self._info_state_nodes[info_state_str]
    return self._current_policy.action_probability_array[
        info_state_node.index_in_tabular_policy]


class _EFRSolver(_EFRSolverBase):
    def __init__(self, game, _deviation_gen):
        super().__init__(game, _deviation_gen)

    def evaluate_and_update_policy(self):
        """Performs a single step of policy evaluation and policy improvement."""
        self._compute_cumulative_immediate_regret_for_player(
            self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            player=None)
        self._update_current_policy(self._root_node, self._current_policy)
        self._iteration += 1


class EFRSolver(_EFRSolver):
    def __init__(self, game, deviations_name):

        # Takes the deviation sets used for learning from Deviation_Sets
        external_only = False
        deviation_sets = None

        if deviations_name == "blind action":
            deviation_sets = return_blind_action
            external_only = True
        elif deviations_name == "informed action":
            deviation_sets = return_informed_action
        elif deviations_name == "blind cf" or deviations_name == "blind counterfactual":
            deviation_sets = return_blind_CF
            external_only = True
        elif deviations_name == "informed cf" or deviations_name == "informed counterfactual":
            deviation_sets = return_informed_CF
        elif deviations_name == "bps" or deviations_name == "blind partial sequence":
            deviation_sets = return_blind_partial_sequence
            external_only = True
        elif deviations_name == "cfps" or deviations_name == "cf partial sequence"\
            or deviations_name == "counterfactual partial sequence":
            deviation_sets = return_cf_partial_sequence
        elif deviations_name == "csps" or deviations_name == "casual partial sequence":
            deviation_sets = return_cs_partial_sequence
        elif deviations_name == "tips" or deviations_name == "twice informed partial sequence":
            deviation_sets = return_twice_informed_partial_sequence
        elif deviations_name == "bhv" or deviations_name == "single target behavioural"\
            or deviations_name == "behavioural":
            deviation_sets = return_behavourial
        else:
            print("Unsupported Deviation Set")
            return None
        super(EFRSolver, self).__init__(game, _deviation_gen=deviation_sets)
        self._external_only = external_only

    def _regret_matching(self, legal_actions, info_set_node):
        """Returns an info state policy by applying regret-matching function
           over all deviations and time selection functions.
        Args:
          cumulative_regrets: A {deviation: y value} dictionary.
          legal_actions: the list of legal actions at this state.

        Returns:
          A dict of action -> prob for all legal actions.
        """
        z = sum(info_set_node.y_values.values())
        info_state_policy = {}

        # The fixed point solution can be directly obtained through the weighted regret matrix
        # if only external deviations are used
        if self._external_only and z > 0:
            weighted_deviation_matrix = np.zeros(
                (len(legal_actions), len(legal_actions)))
            for dev in list(info_set_node.y_values.keys()):
                weighted_deviation_matrix += (
                    info_set_node.y_values[dev]/z) * dev.return_transform_matrix()
            new_strategy = weighted_deviation_matrix[:, 0]
            for index in range(len(legal_actions)):
                info_state_policy[legal_actions[index]] = new_strategy[index]

        # Full regret matching by finding the least squares solution to the fixed point
        # Last row of matrix and the column entry ensures the solution is a strategy (otherwise would have to normalise)
        elif z > 0:
            num_actions = len(info_set_node.legal_actions)
            weighted_deviation_matrix = -np.eye(num_actions)

            for dev in list(info_set_node.y_values.keys()):
                weighted_deviation_matrix += (
                    info_set_node.y_values[dev]/z) * dev.return_transform_matrix()

            normalisation_row = np.ones(num_actions)
            weighted_deviation_matrix = np.vstack(
                [weighted_deviation_matrix, normalisation_row])
            b = np.zeros(num_actions+1)
            b[num_actions] = 1
            b = np.reshape(b, (num_actions+1, 1))

            strategy = lstsq(weighted_deviation_matrix, b)[0]

            # Adopt same clipping strategy as paper author's code
            strategy[np.where(strategy < 0)] = 0
            strategy[np.where(strategy > 1)] = 1

            strategy = strategy/sum(strategy)
            for index in range(len(strategy)):
                info_state_policy[info_set_node.legal_actions[index]
                                  ] = strategy[index]
        # Use a uniform strategy as sum of all regrets is negative
        else:
            for index in range(len(legal_actions)):
                info_state_policy[legal_actions[index]]\
                = 1.0 / len(legal_actions)
        return info_state_policy


def _update_average_policy(average_policy, info_state_nodes):
    """Updates in place `average_policy` to the average of all policies iterated.

    This function is a module level function to be reused by both CFRSolver and
    CFRBRSolver.

    Args:
      average_policy: A `policy.TabularPolicy` to be updated in-place.
      info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
    """
    for info_state, info_state_node in info_state_nodes.items():
        info_state_policies_sum = info_state_node.cumulative_policy
        state_policy = average_policy.policy_for_key(info_state)
        probabilities_sum = sum(info_state_policies_sum.values())
        if probabilities_sum == 0:
            num_actions = len(info_state_node.legal_actions)
            for action in info_state_node.legal_actions:
                state_policy[action] = 1 / num_actions
        else:
            for action, action_prob_sum in info_state_policies_sum.items():
                state_policy[action] = action_prob_sum / probabilities_sum


def strat_dict_to_array(strategy_dictionary):
    """
    A helper function to convert the strategy dictionary action -> prob value to an array.
    Args:
      strategy_dictionary: a dictionary action -> prob value.
    Returns:
      strategy_array: an array with the ith action's value at the i-1th index.
    """
    actions = list(strategy_dictionary.keys())
    strategy_array = np.zeros((len(actions), 1))
    for action in range(len(actions)):
        strategy_array[action][0] = strategy_dictionary[actions[action]]
    return strategy_array


def array_to_strat_dict(strategy_array, legal_actions):
    """
    A helper function to convert a strategy array to an action -> prob value dictionary.
    Args:
      strategy_array: an array with the ith action's value at the i-1th index.
      legal_actions: the list of all legal actions at the current state.
    Returns:
      strategy_dictionary: a dictionary action -> prob value.
    """
    strategy_dictionary = {}
    for action in legal_actions:
        strategy_dictionary[action] = strategy_array[action]
    return strategy_dictionary


def create_probs_from_index(indices, current_policy):
    path_to_state = []
    if indices is None or len(indices) == 0:
        return []
    for index in indices:
        strat_dict = array_to_strat_dict(
            current_policy.action_probability_array[index[1]], index[0])
        path_to_state.append(strat_dict)
    return path_to_state


# Deviation set definitions
def return_blind_action(num_actions, history, _):
    """
    Returns an array of all Blind Action deviations with respect to an information set.
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all Blind Action deviations 
      that are realizable at the
      information set. 
    """
    memory_weights = [np.full(len(history), 1)]
    prior_actions_in_memory = history
    return return_all_external_deviations(num_actions, memory_weights,
                                          prior_actions_in_memory, history)


def return_informed_action(num_actions, history, _):
    """
    Returns an array of all Informed Action deviations with respect to an information set.
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all Informed Action deviations that are realizable at the
      information set. 
    """
    memory_weights = [np.full(len(history), 1)]
    prior_actions_in_memory = history
    return return_all_non_identity_internal_deviations(num_actions, memory_weights, prior_actions_in_memory, history)


def return_blind_CF(num_actions, history, _):
    """
    Returns an array of all Blind Counterfactual deviations with respect to an information set.
    Note: EFR using only Blind Counterfactual deviations is equivalent to vanilla Counterfactual
    Regret Minimisation (CFR).
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all Blind CF deviations 
      that are realizable at the information set. 
    """
    memory_weights = [None]
    prior_actions_in_memory = np.zeros(len(history))
    return return_all_external_deviations(num_actions, memory_weights, prior_actions_in_memory, history)


def return_informed_CF(num_actions, history, _):
    memory_weights = [None]
    prior_actions_in_memory = history
    return return_all_non_identity_internal_deviations(num_actions, memory_weights, prior_actions_in_memory, history)


def return_blind_partial_sequence(num_actions, history, _):
    """
    Returns an array of all Blind Partial Sequence deviations (BPS) 
    with respect to an information set
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all BPS deviations 
      that are realizable at the information set. 
    """
    prior_actions_in_memory = history
    memory_weights = [None]
    if len(history) > 0:
        memory_weights.append(np.ones(len(history)))
    for i in range(len(history)):
        possible_memory_weight = np.zeros(len(history))
        possible_memory_weight[0:i] = np.full(i, 1.0)
        memory_weights.append(possible_memory_weight)
    return return_all_external_deviations(num_actions, memory_weights, prior_actions_in_memory, history)


def return_cf_partial_sequence(num_actions, history, _):
    """
    Returns an array of all Counterfactual Partial Sequence deviations (CFPS)
    with respect to an information set
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all CFPS deviations 
      that are realizable at the information set. 
    """
    prior_actions_in_memory = history
    memory_weights = [None]
    if len(history) > 0:
        memory_weights.append(np.ones(len(history)))
    for i in range(len(history)):
        possible_memory_weight = np.zeros(len(history))
        possible_memory_weight[0:i] = np.full(i, 1.0)
        memory_weights.append(possible_memory_weight)
    return return_all_non_identity_internal_deviations(num_actions, memory_weights, prior_actions_in_memory, history)


def return_cs_partial_sequence(num_actions, history, prior_legal_actions):
    """
    Returns an array of all Casual Partial Sequence deviations with respect to an information set.
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
      prior_legal_actions: an array containing the index in .... that 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all 
      Casual Partial Sequence deviations that are realizable at the
      information set. 
    """
    prior_actions_in_memory = history
    external_memory_weights = [None]

    for i in range(len(history)):
        possible_memory_weight = np.zeros(len(history))
        possible_memory_weight[0:i] = np.full(i, 1.0)
        external_memory_weights.append(possible_memory_weight)

    external = return_all_external_modified_deviations(
        num_actions, external_memory_weights, prior_legal_actions, prior_actions_in_memory, history)
    internal = return_blind_action(num_actions, history, None)

    cf_ext = return_informed_CF(num_actions, history, None)
    cf_int = return_blind_CF(num_actions, history, None)

    return np.concatenate((external, internal, cf_ext, cf_int))


def return_cs_partial_sequence_orginal(num_actions, history, prior_legal_actions):
    """
    Returns an array of all Casual Partial Sequence deviations with respect to an information set.
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
      prior_legal_actions: an array containing the index in .... that 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all 
      Casual Partial Sequence deviations that are realizable at the information set. 
    """
    prior_actions_in_memory = history
    external_memory_weights = [None]

    for i in range(len(history)):
        possible_memory_weight = np.zeros(len(history))
        possible_memory_weight[0:i] = np.full(i, 1.0)
        external_memory_weights.append(possible_memory_weight)

    external = return_all_external_modified_deviations(
        num_actions, external_memory_weights, prior_legal_actions, prior_actions_in_memory, history)
    internal = return_informed_action(num_actions, history, None)

    cf_ext = return_informed_CF(num_actions, history, None)
    return np.concatenate((external, internal, cf_ext))


def return_twice_informed_partial_sequence(num_actions, history, prior_legal_actions):
    """
    Returns an array of all Twice Informed Partial Sequence (TIPS) deviations 
    with respect to an information set.
    Args:
      num_actions: the integer of all actions that can be taken at that information set
      history: an array containing the prior 
      prior_legal_actions: an array containing the index in .... that 
    Returns:
      an array of LocalDeviationWithTimeSelection objects that represent all TIPS deviations that are realizable at the
      information set. 
    """
    prior_actions_in_memory = history
    memory_weights = [None]

    for i in range(len(history)):
        possible_memory_weight = np.zeros(len(history))
        possible_memory_weight[0:i] = np.full(i, 1.0)
        memory_weights.append(possible_memory_weight)

    internal = return_all_internal_modified_deviations(
        num_actions, memory_weights, prior_legal_actions, prior_actions_in_memory, history)

    cf_int = return_informed_CF(num_actions, history, None)
    return np.concatenate((internal, cf_int))


def generate_all_action_permutations(current_stem, remaining_actions):
    if len(remaining_actions) == 0:
        return [np.array(current_stem)]
    else:
        next_actions = remaining_actions[0]
        permutations = []
        for action in next_actions:
            next_stem = current_stem.copy()
            next_stem.append(action)
            next_remaining_actions = remaining_actions[1:]
            prev_permutations = generate_all_action_permutations(
                next_stem, next_remaining_actions)
            for i in prev_permutations:
                permutations.append(i)
        return permutations
# Includes identity


def return_behavourial(num_actions, history, prior_legal_actions):
    deviations = []
    if len(history) == 0:
        internal = return_all_non_identity_internal_deviations(
            num_actions, [None], [None], history)
        for i in internal:
            deviations.append(i)
    else:
        for deviation_info in range(len(history)):
            prior_possible_memory_actions = generate_all_action_permutations(
                [], prior_legal_actions[:deviation_info+1])
            memory_weights = np.concatenate(
                (np.ones(deviation_info), np.zeros(len(history) - deviation_info)))
            for prior_memory_actions in prior_possible_memory_actions:
                prior_memory_actions = np.concatenate(
                    (prior_memory_actions, np.zeros(len(history) - len(prior_memory_actions))))
                for i in range(len(history) - len(prior_memory_actions)):
                    prior_memory_actions.append(0)
                prior_memory_actions_cp = prior_memory_actions.copy()
                internal = return_all_non_identity_internal_deviations(
                    num_actions, [memory_weights], prior_memory_actions_cp, prior_memory_actions_cp)
                for i in internal:
                    deviations.append(i)

    return deviations


class LocalDeviationWithTimeSelection(object):
    local_swap_transform = attr.ib()

    # Which actions have been forgotten (0) or remembered (1) according to the memory state
    prior_actions_weight = attr.ib()

    # Which actions have been take according to the memory state
    prior_memory_actions = attr.ib()

    use_unmodified_history = attr.ib()

    def __init__(self, target, source, num_actions, prior_actions_weight, prior_memory_actions,
                  is_external, use_unmodified_history=True):
        """"
        Args:
        target: the action that will be played when the deviation is triggered
        source: the action that will trigger the target action if (used only by internal deviations, i.e is_external = False)  
        num_actions: the integer of actions
        prior_actions_weight:
        is_external: a boolean use to determine whether to create an internal or external type deviation
        use_unmodified_history: 
        """
        self.local_swap_transform = LocalSwapTransform(
            target, source, num_actions, is_external=is_external)
        self.prior_actions_weight = prior_actions_weight
        self.prior_memory_actions = prior_memory_actions
        self.use_unmodified_history = use_unmodified_history

    # If a pure strategy, a pure strategy will be returned (aka function works for both actions and strategies as input)
    def deviate(self, strategy):
        return self.local_swap_transform.deviate(strategy)

    def return_transform_matrix(self):
        return self.local_swap_transform.matrix_transform

    def player_deviation_reach_probability(self, prior_possible_action_probabilities):
      if self.prior_actions_weight is None or self.prior_memory_actions is None or prior_possible_action_probabilities is None:
                return 1.0

        memory_action_probabilities = np.ones(len(self.prior_actions_weight))
        # Reconstruct memory probabilities from history provided to the deviation to reach info set and the current memory probs
        memory_weightings = self.prior_actions_weight.copy()
        if self.use_unmodified_history:
            for state in range(len(self.prior_memory_actions)):
                if not self.prior_actions_weight[state] == 0:
                    memory_action_probabilities[state] = (
                        prior_possible_action_probabilities[state][self.prior_memory_actions[state]])
                else:
                    memory_action_probabilities[state] = 1
                    memory_weightings[state] = 1
        path_probability = np.multiply(
            memory_weightings, memory_action_probabilities)
        memory_reach_probability = np.prod(path_probability)
        return memory_reach_probability

    def __eq__(self, other):
        if self.local_swap_transform == other.local_swap_transform:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.local_swap_transform)

# Methods to return all


def return_all_non_identity_internal_deviations(num_actions, possible_prior_weights, prior_memory_actions, _):
    deviations = []
    for prior_actions_weight in possible_prior_weights:
        for target in range(num_actions):
            for source in range(num_actions):
                if not source == target:
                    deviations.append(LocalDeviationWithTimeSelection(
                        target, source, num_actions, prior_actions_weight, prior_memory_actions, False))
    return deviations

# EXCLUDES IDENTITY


def return_all_internal_modified_deviations(num_actions,  possible_prior_weights, possible_prior_memory_actions, prior_memory_actions, _):
    deviations = []
    for prior_actions_weight in possible_prior_weights:
        try:
            modification_index = np.where(prior_actions_weight == 0)[0][0]
        except IndexError:
            modification_index = 0
        if modification_index == len(prior_memory_actions):
            for target in range(num_actions):
                for source in range(num_actions):
                    if not source == target:
                        deviations.append(LocalDeviationWithTimeSelection(
                            target, source, num_actions, prior_actions_weight, prior_memory_actions, False))
        else:
            previous_action = prior_memory_actions[modification_index]
            for alt_action in possible_prior_memory_actions[modification_index]:
                prior_memory_actions[modification_index] = alt_action
                for target in range(num_actions):
                    for source in range(num_actions):
                        if not source == target:
                            deviations.append(LocalDeviationWithTimeSelection(
                                target, source, num_actions, prior_actions_weight, prior_memory_actions.copy(), False))
                prior_memory_actions[modification_index] = previous_action
    return deviations


def return_all_external_deviations(num_actions,  possible_prior_weights, prior_memory_actions, _):
    deviations = []
    for prior_actions_weight in possible_prior_weights:
        for target in range(num_actions):
            deviations.append(LocalDeviationWithTimeSelection(
                target, target, num_actions, prior_actions_weight, prior_memory_actions, True))
    return deviations

# Modify last action as required


def return_all_external_modified_deviations(num_actions,  possible_prior_weights, possible_prior_memory_actions, prior_memory_actions, _):
    deviations = []
    for prior_actions_weight in possible_prior_weights:
        try:
            modification_index = np.where(prior_actions_weight == 0)[0][0]
        except IndexError:
            modification_index = 0
        if modification_index == len(prior_memory_actions):
            for target in range(num_actions):
                deviations.append(LocalDeviationWithTimeSelection(
                    target, target, num_actions, prior_actions_weight, prior_memory_actions, True))
        else:
            previous_action = prior_memory_actions[modification_index]
            for alt_action in possible_prior_memory_actions[modification_index]:
                prior_memory_actions[modification_index] = alt_action
                for target in range(num_actions):
                    deviations.append(LocalDeviationWithTimeSelection(
                        target, target, num_actions, prior_actions_weight, prior_memory_actions.copy(), True))
                prior_memory_actions[modification_index] = previous_action
    return deviations


def return_identity_deviation(num_actions,  possible_prior_weights, prior_memory_actions, _):
    deviations = []
    for prior_actions_weight in possible_prior_weights:
        deviations.append(LocalDeviationWithTimeSelection(
            0, 0, num_actions, prior_actions_weight, prior_memory_actions, False))
    return deviations


# A swap transformation given by the matrix_transform for an information state of
class LocalSwapTransform(object):
    """
    TODO
    """
    source_action = attr.ib()
    target_action = attr.ib()
    matrix_transform = attr.ib()
    actions_num = attr.ib()
    is_external = attr.ib()

    def __init__(self, target, source, actions_num, is_external=True):
        self.source_action = source
        self.target_action = target
        self.actions_num = actions_num
        if is_external:
            self.source_action = None
            self.matrix_transform = np.zeros((actions_num, actions_num))
            self.matrix_transform[target] = np.ones(actions_num)
        else:
            self.matrix_transform = np.eye(actions_num)
            self.matrix_transform[target][source] = 1
            self.matrix_transform[source][source] = 0

    def __repr__(self) -> str:
        return "Shifting probabilty from Action: "+str(self.source_action) + " to Action: "+str(self.target_action)

    def __eq__(self, __o: object) -> bool:
        if self.source_action == __o.source_action and self.target_action == __o.target_action and self.actions_num == __o.actions_num:
            return True
        else:
            return False

    def __hash__(self):
        separator = " "
        return hash(str(self.source_action)+separator+str(self.target_action)+separator+str(self.actions_num) + separator + str(self.is_external))

    # If a pure strategy, a pure strategy will be returned (aka function works for both actions and strategies as input)
    def deviate(self, strategy):
        """
        Returns the deviation strategy
        Args:
          strategy: the strategy array to multiply the deviation matrix by.
        Returns:

        """
        return np.matmul(self.matrix_transform, strategy)
