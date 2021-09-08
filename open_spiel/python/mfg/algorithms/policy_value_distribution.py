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

"""Does a backward pass to output the value distributions of a policy."""
from typing import Dict

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value as value_module
import pyspiel


class PolicyValueDistribution(value_module.ValueFunction):
  """Computes the value of a specified strategy.

  Attributes:
    _distribution:
    _policy:
    _root_states:
    _state_value:
  """
  _state_value: Dict[str, Dict[float, float]]

  def __init__(self,
               game,
               distribution: distribution_std.Distribution,
               policy: policy_std.Policy,
               root_state=None):
    """Initializes the value calculation.

    Args:
      game: The game to analyze.
      distribution: A `distribution.Distribution` object.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    super().__init__(game)
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distribution = distribution
    self._policy = policy

    self._state_value = {}

    self.evaluate()

  def eval_state(self, state) -> Dict[float, float]:
    """Evaluate the value of a state."""
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if state_str in self._state_value:
      return self._state_value[state_str]
    self._state_value[state_str] = {}
    if state.is_terminal():
      self._state_value[state_str] = {state.rewards()[
          state.mean_field_population()]: 1.0}
      return self._state_value[state_str]
    if state.current_player() == pyspiel.PlayerId.CHANCE:
      for action, prob in state.chance_outcomes():
        new_state = state.child(action)
        outcome_prob_dict = self.eval_state(new_state)
        for outcome, prob_outcome in outcome_prob_dict.items():
          if outcome not in self._state_value[state_str]:
            self._state_value[state_str][outcome] = 0
          self._state_value[state_str][outcome] += prob * prob_outcome
      return self._state_value[state_str]
    if state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = state.distribution_support()
      dist = [
          self._distribution.value_str(str_state, 0.)
          for str_state in dist_to_register
      ]
      new_state = state.clone()
      new_state.update_distribution(dist)
      outcome_prob_dict = self.eval_state(new_state)
      for outcome_new_state, prob_outcome in outcome_prob_dict.items():
        outcome = (state.rewards()[state.mean_field_population()] +
                   outcome_new_state)
        if outcome not in self._state_value[state_str]:
          self._state_value[state_str][outcome] = 0
        self._state_value[state_str][outcome] += prob_outcome
      return self._state_value[state_str]
    assert int(state.current_player()) >= 0, "The player id should be >= 0"
    for action, prob in self._policy.action_probabilities(state).items():
      new_state = state.child(action)
      outcome_prob_dict = self.eval_state(new_state)
      for outcome_new_state, prob_outcome in outcome_prob_dict.items():
        outcome = (outcome_new_state +
                   state.rewards()[state.mean_field_population()])
        if outcome not in self._state_value[state_str]:
          self._state_value[state_str][outcome] = 0
        self._state_value[state_str][outcome] += prob * prob_outcome
    return self._state_value[state_str]

  def evaluate(self):
    """Evaluate the value over states of self._policy."""
    for state in self._root_states:
      self.eval_state(state)

  def value_distribution(self, state, action=None):
    """Returns a distribution of possible outcomes.

    Args:
      state: A `pyspiel.State` object.
      action: may be None or a legal action

    Returns:
      A probability distribution of over the possible outcomes as a dictionary
        of float to float. The key of the dictionary are the possible values of
        the state given the action and the values are the probability to have
        these values.
    """
    if action is None:
      return self._state_value[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    new_state = state.child(action)
    outcome_prob_dict = self._state_value[
        new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    returned_outcome_prob_dict = {}
    for outcome_new_state, prob_outcome in outcome_prob_dict.items():
      outcome = outcome_new_state + state.rewards()[0]
      if outcome not in returned_outcome_prob_dict:
        returned_outcome_prob_dict[outcome] = 0
      returned_outcome_prob_dict[outcome] += prob_outcome
    return returned_outcome_prob_dict

  def value(self, state, action=None):
    """Returns a float representing a value.

    Args:
      state: A `pyspiel.State` object.
      action: may be None or a legal action

    Returns:
      A value for the state (and eventuallu state action pair).
    """
    value = 0
    for outcome, prob in self.value_distribution(state, action):
      value += outcome * prob
    return value
