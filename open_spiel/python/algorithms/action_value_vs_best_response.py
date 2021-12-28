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

"""Compute the value of action given a policy vs a best responder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from open_spiel.python import policy
from open_spiel.python.algorithms import action_value
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
    self._num_actions = game.num_distinct_actions()

    self._action_value_calculator = action_value.TreeWalkCalculator(game)
    # best_responder[i] is a best response to the provided policy for player i.
    # It is therefore a policy for player (1-i).
    self._best_responder = {0: None, 1: None}
    self._all_states = None

  def __call__(self, player, player_policy, info_states):
    """Computes action values per state for the player.

    Args:
      player: The id of the player (0 <= player < game.num_players()). This
        player will play `player_policy`, while the opponent will play a best
        response.
      player_policy: A `policy.Policy` object.
      info_states: A list of info state strings.

    Returns:
      A `_CalculatorReturn` nametuple. See its docstring for the documentation.
    """
    self.player = player
    opponent = 1 - player

    def best_response_policy(state):
      infostate = state.information_state_string(opponent)
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
            state: self._all_states[state].information_state_string()
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
    best_response_value = self._best_responder[player].value_from_state(
        self.game.new_initial_state())
    best_response_actions = self._best_responder[
        player].get_best_response_actions()

    # Compute action values
    self._action_value_calculator.compute_all_states_action_values({
        player:
            player_policy,
        opponent:
            policy.tabular_policy_from_callable(
                self.game, best_response_policy, [opponent]),
    })
    obj = self._action_value_calculator._get_tabular_statistics(  # pylint: disable=protected-access
        ((player, s) for s in info_states))

    # Return values
    return _CalculatorReturn(
        exploitability=best_response_value,
        values_vs_br=obj.action_values,
        counterfactual_reach_probs_vs_br=obj.counterfactual_reach_probs,
        player_reach_probs_vs_br=obj.player_reach_probs)
