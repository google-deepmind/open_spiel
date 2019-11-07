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

"""Compute the value of action given a policy vs a best responder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_utils
import pyspiel


def _transitions(state, policies):
  """Returns a list of (action, prob) pairs from the specified state."""
  if state.is_chance_node():
    return state.chance_outcomes()
  else:
    pl = state.current_player()
    return list(policies[pl].action_probabilities(state).items())


def _tuples_from_policy(policy_vector):
  return [
      (action, probability) for action, probability in enumerate(policy_vector)
  ]


_CalculatorReturn = collections.namedtuple(
    "_CalculatorReturn",
    [
        # The exploitability of the opponent strategy, i.e. the value of the
        # best-responder player BR.
        "exploitability",
        # An array of shape `[len(info_states), game.num_distinct_actions()]`
        # giving the value of each action vs the best response.
        # Will be zero for invalid actions.
        "values_vs_br",
        # The player's counterfactual reach probability of this infostate when
        # playing against the BR, as a list of shape [num_info_states].
        "counterfactual_reach_probs_vs_br",
        # The reach probability of the current player at the infostates when
        # playing against the BR, as list shape [num_info_states].
        # This is the product of the current player probs along *one* trajectory
        # leading to this info-state (this number should be the same along
        # any trajectory leading to this info-state because of perfect recall).
        "player_reach_probs_vs_br",
    ])


class Calculator(object):
  """Class to orchestrate the calculation."""

  def __init__(self, game):
    if game.num_players() != 2:
      raise ValueError("Only supports 2-player games.")
    self.game = game
    self._num_players = game.num_players()
    self.action_values = None
    self.num_actions = game.num_distinct_actions()
    self.info_state_prob = None
    self.info_state_cf_prob = None
    self.info_state_chance_prob = None
    self._best_responder = {0: None, 1: None}
    self._all_states = None

  def get_action_values(self, state, policies, reach_probabilities=None):
    """Computes the value of the state given the policies.

    Args:
      state: The state to start analysis from.
      policies: List of `policy.Policy` objects, one per player.
      reach_probabilities: A numpy array of shape `[num_players + 1]`.
        reach_probabilities[i] is the product of the player i action
        probabilities along the current trajectory. Note that
        reach_probabilities[-1] corresponds to the chance player.

    Returns:
      The value of the root state to each player.

    Side-effects - populates:
      `self.action_values[(player, infostate)][action]`.
      `self.info_state_prob[(player, infostate)]`.
      `self.info_state_player_prob[(player, infostate)]`:
      `self.info_state_cf_prob[(player, infostate)]`.
      `self.info_state_chance_prob[(player, infostate)]`.

    We use `(player, infostate)` as a key in case the same infostate is shared
    by multiple players, e.g. in a simultaneous-move game.
    """
    if reach_probabilities is None:
      reach_probabilities = np.ones(self._num_players + 1)

    if state.is_terminal():
      return np.array(state.returns())

    current_player = state.current_player()
    reach_prob = np.prod(reach_probabilities)

    is_chance = state.is_chance_node()
    if not is_chance:
      key = (state.current_player(), state.information_state())
      counterfactual_reach_prob = (
          np.prod(reach_probabilities[:current_player]) *
          np.prod(reach_probabilities[current_player + 1:]))
      self.info_state_prob[key] += reach_prob
      self.info_state_cf_prob[key] += counterfactual_reach_prob
      self.info_state_chance_prob[key] += reach_probabilities[-1]
      # Mind that we have "=" here and not "+=", because we just need to use
      # the reach prob for the player for *any* of the histories leading to
      # the current info_state (they are all equal because of perfect recall).
      self.info_state_player_prob[key] = reach_probabilities[current_player]

    value = np.zeros(len(policies))
    for action, prob in _transitions(state, policies):
      child = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= prob
      child_value = self.get_action_values(
          child, policies, reach_probabilities=new_reach_probabilities)
      if not is_chance:
        self.action_values[key][action] += child_value * reach_prob
      value += child_value * prob
    return value

  def __call__(self, player, player_policy, info_states):
    """Computes action values per state for the player.

    Args:
      player: The id of the player 0 <= player < game.num_players().
      player_policy: A `policy.Policy` object.
      info_states: A list of info state strings.

    Returns:
      A `_CalculatorReturn` nametuple. See its docstring for the documentation.
    """
    self.player = player
    opponent = 1 - player

    def best_response_policy(state):
      infostate = state.information_state(opponent)
      action = best_response_actions[infostate]
      return [(action, 1.0)]

    # If the policy is a TabularPolicy, we can directly copy the infostate
    # strings & values from the class. This is significantly faster than having
    # to create the infostate strings.
    if isinstance(player_policy, policy.TabularPolicy):
      tabular_policy = {
          key: _tuples_from_policy(player_policy.policy_for_key(key))
          for key in player_policy.state_lookup
      }
    # Otherwise, we have to calculate all the infostate strings everytime. This
    # is ~2x slower.
    else:
      # We cache these as they are expensive to compute & do not change.
      if self._all_states is None:
        self._all_states = get_all_states.get_all_states(
            self.game,
            depth_limit=-1,
            include_terminals=False,
            include_chance_states=False)
        self._state_to_information_state = {
            state: self._all_states[state].information_state()
            for state in self._all_states
        }
      tabular_policy = policy_utils.policy_to_dict(
          player_policy, self.game, self._all_states,
          self._state_to_information_state)

    # When constructed, TabularBestResponse does a lot of work; we can save that
    # work by caching it.
    if self._best_responder[player] is None:
      self._best_responder[player] = pyspiel.TabularBestResponse(
          self.game, opponent, tabular_policy)
    else:
      self._best_responder[player].set_policy(tabular_policy)

    # Computing the value at the root calculates best responses everywhere.
    history = str(self.game.new_initial_state())
    best_response_value = self._best_responder[player].value(history)
    best_response_actions = self._best_responder[
        player].get_best_response_actions()

    # Compute action values
    self.action_values = collections.defaultdict(
        lambda: collections.defaultdict(lambda: np.zeros(2)))
    self.info_state_prob = collections.defaultdict(float)
    self.info_state_player_prob = collections.defaultdict(float)
    self.info_state_cf_prob = collections.defaultdict(float)
    self.info_state_chance_prob = collections.defaultdict(float)
    self.get_action_values(
        self.game.new_initial_state(), {
            player:
                player_policy,
            opponent:
                policy.PolicyFromCallable(self.game, best_response_policy),
        })

    # Collect normalized action values for each information state
    rv = []
    cfrp = []
    player_reach_probs_vs_br = []
    for info_state in info_states:
      key = (player, info_state)
      av = self.action_values[key]
      norm_prob = self.info_state_prob[key]
      rv.append([(av[a][player] / norm_prob) if
                 (a in av and norm_prob > 0) else 0
                 for a in range(self.num_actions)])
      cfrp.append(self.info_state_cf_prob[key])
      player_reach_probs_vs_br.append(self.info_state_player_prob[key])

    # Return values
    return _CalculatorReturn(
        exploitability=best_response_value,
        values_vs_br=rv,
        counterfactual_reach_probs_vs_br=cfrp,
        player_reach_probs_vs_br=player_reach_probs_vs_br)
