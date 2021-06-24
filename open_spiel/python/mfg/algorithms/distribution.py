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

"""Computes the distribution of a policy."""
import collections

from open_spiel.python.mfg import distribution
import pyspiel


def type_from_states(states):
  """Get node type of a list of states and assert they are the same."""
  types = [state.get_type() for state in states]
  for state_type in types:
    assert types[0] == state_type, f'types: {types}'
  return types[0]


class DistributionPolicy(distribution.Distribution):
  """Computes the distribution of a specified strategy."""

  def __init__(self, game, policy, root_state=None):
    """Initializes the distribution calculation.

    Args:
      game: The game to analyze.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
    """
    super().__init__(game)
    self._policy = policy
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self.evaluate()

  def evaluate(self):
    """Evaluate the distribution over states of self._policy."""
    # List of all game states that have a non-zero probability at the current
    # timestep and player ID.
    listing_states = self._root_states.copy()
    # Maps state strings to floats. For each group of states at a
    # given timestep, given player ID and given population, these
    # floats represent a probability distribution.
    self.distribution = collections.defaultdict(float)
    for state in listing_states:
      self.distribution[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)] = 1.

    while type_from_states(listing_states) != pyspiel.StateType.TERMINAL:
      new_listing_states = []

      if type_from_states(listing_states) == pyspiel.StateType.CHANCE:
        for mfg_state in listing_states:
          for action, prob in mfg_state.chance_outcomes():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            if new_mfg_state_str not in self.distribution:
              new_listing_states.append(new_mfg_state)
            self.distribution[new_mfg_state_str] += prob * self.distribution[
                mfg_state.observation_string(
                    pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

      elif (type_from_states(listing_states) ==
            pyspiel.StateType.MEAN_FIELD):
        for mfg_state in listing_states:
          dist_to_register = mfg_state.distribution_support()
          dist = [
              self.distribution[str_state] for str_state in dist_to_register
          ]
          new_mfg_state = mfg_state.clone()
          new_mfg_state.update_distribution(dist)
          new_listing_states.append(new_mfg_state)
          self.distribution[new_mfg_state.observation_string(
              pyspiel.PlayerId.DEFAULT_PLAYER_ID)] = self.distribution[
                  mfg_state.observation_string(
                      pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

      else:
        assert type_from_states(
            listing_states) == pyspiel.StateType.DECISION
        for mfg_state in listing_states:
          for action, prob in self._policy.action_probabilities(
              mfg_state).items():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            if new_mfg_state_str not in self.distribution:
              new_listing_states.append(new_mfg_state)
            self.distribution[new_mfg_state_str] += prob * self.distribution[
                mfg_state.observation_string(
                    pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
      listing_states = new_listing_states

  def value(self, state):
    return self.value_str(
        state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))

  def value_str(self, state_str):
    state_probability = self.distribution.get(state_str)
    if state_probability is None:
      # Check this because self.distribution is a default dict.
      raise ValueError(f'Distribution not computed for state {state_str}')
    return state_probability
