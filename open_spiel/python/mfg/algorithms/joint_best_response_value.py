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

"""Outputs value of best response policy against set of distributions."""
import collections
from typing import List
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
import pyspiel


class JointBestResponse(value.ValueFunction):
  """Computes a best response value."""

  def __init__(
      self,
      game,
      distributions: List[distribution_std.Distribution],
      weights,
      root_state=None,
  ):
    """Initializes the joint best response computation.

    The joint best response is computed under the following premisse : the
    player does not know which distribution it is playing against. It only knows
    their probabilities, and thus tries to find a best response against their
    mixture.

    This is accomplished by recursively computing the action that maximizes the
    marginalized value of each node over each distribution.

    Warning : This version only works on games whose observation space &
    dynamics do NOT depend on state distribution.

    Args:
      game: The game to analyze.
      distributions: A list of `distribution_std.Distribution`.
      weights: A list of floats the same length as `distributions`. Represents
        the mixture weight of each member of `distributions`.
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    super().__init__(game)
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distributions = distributions
    self._weights = weights
    # Maps states (in string format) to the value of the optimal policy given
    # 'self._distribution'.
    self._state_value = collections.defaultdict(float)
    self.evaluate()

  def get_state_rewards(self, mu_states):
    return sum([
        weight * mu_state.rewards()[mu_state.mean_field_population()]
        for weight, mu_state in zip(self._weights, mu_states)
    ])

  def get_new_mu_states(self, mu_states):
    new_mu_states = []
    for mu_ind, mu_state in enumerate(mu_states):
      dist = [
          self._distributions[mu_ind].value_str(str_state, 0.0)
          for str_state in mu_state.distribution_support()
      ]
      new_mu_state = mu_state.clone()
      new_mu_state.update_distribution(dist)
      new_mu_states.append(new_mu_state)
    return new_mu_states

  def eval_state(self, mu_states):
    """Evaluate the value of a state.

    Args:
      mu_states: A list of game states, one for each `distributions` member.

    Returns:
      The optimal value of the state.

    Recursively computes the value of the optimal policy given the fixed state
    distributions. `self._state_value` is used as a cache for pre-computed
    values.
    """
    state = mu_states[0]
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if state_str in self._state_value:
      return self._state_value[state_str]
    if state.is_terminal():
      self._state_value[state_str] = self.get_state_rewards(mu_states)
      return self._state_value[state_str]
    if state.current_player() == pyspiel.PlayerId.CHANCE:
      self._state_value[state_str] = 0.0
      for action, prob in state.chance_outcomes():
        new_mu_states = [mu_state.child(action) for mu_state in mu_states]
        self._state_value[state_str] += prob * self.eval_state(new_mu_states)
      return self._state_value[state_str]
    if state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      new_mu_states = self.get_new_mu_states(mu_states)
      self._state_value[state_str] = self.get_state_rewards(
          mu_states
      ) + self.eval_state(new_mu_states)
      return self._state_value[state_str]
    else:
      assert int(state.current_player()) >= 0, "The player id should be >= 0"
      max_q = max(
          self.eval_state([mu_state.child(action) for mu_state in mu_states])
          for action in state.legal_actions()
      )
      self._state_value[state_str] = self.get_state_rewards(mu_states) + max_q
      return self._state_value[state_str]

  def evaluate(self):
    """Evaluate the best response value on all states."""
    for state in self._root_states:
      self.eval_state([state.clone() for _ in self._distributions])

  def value(self, state, action=None):
    if action is None:
      return self._state_value[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    new_state = state.child(action)
    return state.rewards()[state.mean_field_population()] + self._state_value[
        new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
