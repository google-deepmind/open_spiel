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

"""Does a backward pass to output a value of a best response policy."""
import collections

from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
import pyspiel


class BestResponse(value.ValueFunction):
  """Computes a best response value."""

  def __init__(self,
               game,
               distribution: distribution_std.Distribution,
               root_state=None):
    """Initializes the best response calculation.

    Args:
      game: The game to analyze.
      distribution: A `distribution_std.Distribution` object.
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    super(BestResponse, self).__init__(game)
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distribution = distribution
    # Maps states (in string format) to the value of the optimal policy given
    # 'self._distribution'.
    self._state_value = collections.defaultdict(float)

    self.evaluate()

  def eval_state(self, state):
    """Evaluate the value of a state.

    Args:
      state: a game state.

    Returns:
      the optimal value of the state

    Recursively computes the value of the optimal policy given the fixed state
    distribution. `self._state_value` is used as a cache for pre-computed
    values.
    """
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if state_str in self._state_value:
      return self._state_value[state_str]
    elif state.is_terminal():
      self._state_value[state_str] = state.rewards()[
          state.mean_field_population()]
      return self._state_value[state_str]
    elif state.current_player() == pyspiel.PlayerId.CHANCE:
      self._state_value[state_str] = 0.0
      for action, prob in state.chance_outcomes():
        new_state = state.child(action)
        self._state_value[state_str] += prob * self.eval_state(new_state)
      return self._state_value[state_str]
    elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = state.distribution_support()
      dist = [
          self._distribution.value_str(str_state)
          for str_state in dist_to_register
      ]
      new_state = state.clone()
      new_state.update_distribution(dist)
      self._state_value[state_str] = (
          state.rewards()[state.mean_field_population()] +
          self.eval_state(new_state))
      return self._state_value[state_str]
    else:
      assert int(state.current_player()) >= 0, "The player id should be >= 0"
      state_legal_actions = state.legal_actions()
      if state_legal_actions:
        max_q = max(
            self.eval_state(state.child(action))
            for action in state.legal_actions())
      else:
        # If no legal action 0 should be played.
        max_q = self.eval_state(state.child(0))
      self._state_value[state_str] = state.rewards()[
          state.mean_field_population()] + max_q
      return self._state_value[state_str]

  def evaluate(self):
    """Evaluate the best response value on all states."""
    for s in self._root_states:
      self.eval_state(s)

  def value(self, state, action=None):
    if action is None:
      return self._state_value[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    else:
      new_state = state.child(action)
      return state.rewards()[state.mean_field_population()] + self._state_value[
          new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
