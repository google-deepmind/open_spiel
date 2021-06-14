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

"""Does a backward pass to output the value of a policy."""
import collections

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
import pyspiel


class PolicyValue(value.ValueFunction):
  """Computes the value of a specified strategy."""

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
    super(PolicyValue, self).__init__(game)
    if root_state is None:
      root_state = game.new_initial_state()
    self._root_state = root_state
    self._distribution = distribution
    self._policy = policy

    self._state_value = collections.defaultdict(float)

    self.evaluate()

  def eval_state(self, state):
    """Evaluate the value of a state."""
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if state_str in self._state_value:
      return self._state_value[state_str]
    elif state.is_terminal():
      self._state_value[state_str] = state.rewards()[0]
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
          state.rewards()[0] + self.eval_state(new_state))
      return self._state_value[state_str]
    else:
      assert state.current_player() == 0, "The player id should be 0"
      v = 0.0
      for action, prob in self._policy.action_probabilities(state).items():
        new_state = state.child(action)
        v += prob * self.eval_state(new_state)
      self._state_value[state_str] = state.rewards()[0] + v
      return self._state_value[state_str]

  def evaluate(self):
    """Evaluate the value over states of self._policy."""
    _ = self.eval_state(self._root_state)

  def value(self, state, action=None):
    if action is None:
      return self._state_value[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    else:
      new_state = state.child(action)
      return state.rewards()[0] + self._state_value[
          new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
