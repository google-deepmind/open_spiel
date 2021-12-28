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

"""Q-values and reach probabilities computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

_CalculatorReturn = collections.namedtuple(
    "_CalculatorReturn",
    [
        # A list of size `num_players` of the root node value for each player.
        "root_node_values",
        # An array of shape `[len(info_states), game.num_distinct_actions()]`
        # giving the value of each action. Will be zero for invalid actions.
        "action_values",
        # The player's counterfactual reach probability of this infostate when
        # playing, as a list of shape [num_info_states].
        "counterfactual_reach_probs",
        # The reach probability of the current player at the infostates, as a
        # list of shape [num_info_states].
        # This is the product of the current player probs along *one* trajectory
        # leading to this info-state (this number should be the same along
        # any trajectory leading to this info-state because of perfect recall).
        "player_reach_probs",
        # A list of `len(info_states)` `[game.num_distinct_actions()]` numpy
        # array so that v[s_index][a] = \sum_{h \in x} cfr_reach(h) * Q(h, a)
        "sum_cfr_reach_by_action_value",
    ])


class TreeWalkCalculator(object):
  r"""Class to orchestrate the calculation.

  This performs a full history tree walk and computes several statistics,
  available as attributes.

  Attributes:
    weighted_action_values: A dictionary mapping (player,information state
      string) to a dictionary mapping each action to a vector of the sum of
      (reward * prob) reward taking that action for each player. To get the
      action-values, one will need to normalize by `info_state_prob`.
    info_state_prob:  A dictionary mapping (player,information state string) to
      the reach probability of this info_state.
    info_state_player_prob: Same as info_state_prob for the player reach
      probability.
    info_state_cf_prob: Same as info_state_prob for the counterfactual reach
      probability to get to that state, i.e. the sum over histories, of the
      product of the opponents probabilities of actions leading to the history.
    info_state_chance_prob: Same as above, for the chance probability to get
      into that state.
    info_state_cf_prob_by_q_sum: A dictionary mapping (player,information state
      string) to a vector of shape `[num_actions]`, that store for each action
      the cumulative \sum_{h \in x} cfr_reach(h) * Q(h, a)
    root_values: The values at the root node [for player 0, for player 1].
  """

  def __init__(self, game):
    if not game.get_type().provides_information_state_string:
      raise ValueError("Only game which provide the information_state_string "
                       "are supported, as this is being used in the key to "
                       "identify states.")

    self._game = game
    self._num_players = game.num_players()
    self._num_actions = game.num_distinct_actions()

    self.weighted_action_values = None
    self.info_state_prob = None
    self.info_state_player_prob = None
    self.info_state_cf_prob = None
    self.info_state_chance_prob = None
    self.info_state_cf_prob_by_q_sum = None
    self.root_values = None

  def _get_action_values(self, state, policies, reach_probabilities):
    """Computes the value of the state given the policies for both players.

    Args:
      state: The state to start analysis from.
      policies: List of `policy.Policy` objects, one per player.
      reach_probabilities: A numpy array of shape `[num_players + 1]`.
        reach_probabilities[i] is the product of the player i action
        probabilities along the current trajectory. Note that
        reach_probabilities[-1] corresponds to the chance player. Initially, it
        should be called with np.ones(self._num_players + 1) at the root node.

    Returns:
      The value of the root state to each player.

    Side-effects - populates:
      `self.weighted_action_values[(player, infostate)][action]`.
      `self.info_state_prob[(player, infostate)]`.
      `self.info_state_cf_prob[(player, infostate)]`.
      `self.info_state_chance_prob[(player, infostate)]`.

    We use `(player, infostate)` as a key in case the same infostate is shared
    by multiple players, e.g. in a simultaneous-move game.
    """
    if state.is_terminal():
      return np.array(state.returns())

    current_player = state.current_player()
    is_chance = state.is_chance_node()

    if not is_chance:
      key = (current_player, state.information_state_string())
      reach_prob = np.prod(reach_probabilities)

      # We exclude both the current and the chance players.
      opponent_probability = (
          np.prod(reach_probabilities[:current_player]) *
          np.prod(reach_probabilities[current_player + 1:-1]))
      self.info_state_cf_prob[key] += (
          reach_probabilities[-1] * opponent_probability)
      self.info_state_prob[key] += reach_prob
      self.info_state_chance_prob[key] += reach_probabilities[-1]
      # Mind that we have "=" here and not "+=", because we just need to use
      # the reach prob for the player for *any* of the histories leading to
      # the current info_state (they are all equal because of perfect recall).
      self.info_state_player_prob[key] = reach_probabilities[current_player]

    value = np.zeros(len(policies))
    if is_chance:
      action_to_prob = dict(state.chance_outcomes())
    else:
      action_to_prob = policies[current_player].action_probabilities(state)
    for action in state.legal_actions():
      prob = action_to_prob.get(action, 0)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= prob

      child = state.child(action)
      child_value = self._get_action_values(
          child, policies, reach_probabilities=new_reach_probabilities)
      if not is_chance:
        self.weighted_action_values[key][action] += child_value * reach_prob
        self.info_state_cf_prob_by_q_sum[key][action] += (
            child_value[current_player] * opponent_probability *
            reach_probabilities[-1])
      value += child_value * prob
    return value

  def compute_all_states_action_values(self, policies):
    """Computes action values per state for the player.

    The internal state is fully re-created when calling this method, thus it's
    safe to use one object to perform several tree-walks using different
    policies, and to extract the results using for example
    `calculator.infor_state_prob` to take ownership of the dictionary.

    Args:
      policies: List of `policy.Policy` objects, one per player. As the policy
        will be accessed using `policies[i]`, it can also be a dictionary
        mapping player_id to a `policy.Policy` object.
    """
    assert len(policies) == self._num_players

    # Compute action values
    self.weighted_action_values = collections.defaultdict(
        lambda: collections.defaultdict(lambda: np.zeros(self._num_players)))
    self.info_state_prob = collections.defaultdict(float)
    self.info_state_player_prob = collections.defaultdict(float)
    self.info_state_cf_prob = collections.defaultdict(float)
    self.info_state_chance_prob = collections.defaultdict(float)
    self.info_state_cf_prob_by_q_sum = collections.defaultdict(
        lambda: np.zeros(self._num_actions))

    self.root_values = self._get_action_values(
        self._game.new_initial_state(),
        policies,
        reach_probabilities=np.ones(self._num_players + 1))

  def _get_tabular_statistics(self, keys):
    """Returns tabular numpy arrays of the resulting stastistics.

    Args:
      keys: A list of the (player, info_state_str) keys to use to return the
        tabular numpy array of results.
    """
    # Collect normalized action values for each information state
    action_values = []
    cfrp = []  # Counterfactual reach probabilities
    player_reach_probs = []
    sum_cfr_reach_by_action_value = []

    for key in keys:
      player = key[0]
      av = self.weighted_action_values[key]
      norm_prob = self.info_state_prob[key]
      action_values.append([(av[a][player] / norm_prob) if
                            (a in av and norm_prob > 0) else 0
                            for a in range(self._num_actions)])
      cfrp.append(self.info_state_cf_prob[key])
      player_reach_probs.append(self.info_state_player_prob[key])
      sum_cfr_reach_by_action_value.append(
          self.info_state_cf_prob_by_q_sum[key])

    # Return values
    return _CalculatorReturn(
        root_node_values=self.root_values,
        action_values=action_values,
        counterfactual_reach_probs=cfrp,
        player_reach_probs=player_reach_probs,
        sum_cfr_reach_by_action_value=sum_cfr_reach_by_action_value)

  def get_tabular_statistics(self, tabular_policy):
    """Returns tabular numpy arrays of the resulting stastistics.

    This function should be called after `compute_all_states_action_values`.
    Optionally, one can directly call the object to perform both actions.

    Args:
      tabular_policy: A `policy.TabularPolicy` object, used to get the ordering
        of the states in the tabular numpy array.
    """
    keys = []
    for player_id, player_states in enumerate(tabular_policy.states_per_player):
      keys += [(player_id, s) for s in player_states]
    return self._get_tabular_statistics(keys)

  def __call__(self, policies, tabular_policy):
    """Computes action values per state for the player.

    The internal state is fully re-created when calling this method, thus it's
    safe to use one object to perform several tree-walks using different
    policies, and to extract the results using for example
    `calculator.infor_state_prob` to take ownership of the dictionary.

    Args:
      policies: List of `policy.Policy` objects, one per player.
      tabular_policy: A `policy.TabularPolicy` object, used to get the ordering
        of the states in the tabular numpy array.

    Returns:
      A `_CalculatorReturn` namedtuple. See its docstring for the details.
    """
    self.compute_all_states_action_values(policies)
    return self.get_tabular_statistics(tabular_policy)

  def get_root_node_values(self, policies):
    """Gets root values only.

    This speeds up calculation in two ways:

    1. It only searches nodes with positive probability.
    2. It does not populate a large dictionary of meta information.

    Args:
      policies: List of `policy.Policy` objects, one per player.

    Returns:
      A numpy array of shape [num_players] of the root value.
    """
    return self._get_action_values_only(
        self._game.new_initial_state(),
        policies,
        reach_probabilities=np.ones(self._num_players + 1))

  def _get_action_values_only(self, state, policies, reach_probabilities):
    """Computes the value of the state given the policies for both players.

    Args:
      state: The state to start analysis from.
      policies: List of `policy.Policy` objects, one per player.
      reach_probabilities: A numpy array of shape `[num_players + 1]`.
        reach_probabilities[i] is the product of the player i action
        probabilities along the current trajectory. Note that
        reach_probabilities[-1] corresponds to the chance player. Initially, it
        should be called with np.ones(self._num_players + 1) at the root node.

    Returns:
      A numpy array of shape [num_players] of the root value.
    """
    if state.is_terminal():
      return np.array(state.returns())

    current_player = state.current_player()
    is_chance = state.is_chance_node()

    value = np.zeros(len(policies))
    if is_chance:
      action_to_prob = dict(state.chance_outcomes())
    else:
      action_to_prob = policies[current_player].action_probabilities(state)

    for action in state.legal_actions():
      prob = action_to_prob.get(action, 0)

      # Do not follow tree down if there is zero probability.
      if prob == 0.0:
        continue

      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= prob

      child = state.child(action)
      child_value = self._get_action_values_only(
          child, policies, reach_probabilities=new_reach_probabilities)
      value += child_value * prob
    return value
