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
    super(DistributionPolicy, self).__init__(game)
    self._policy = policy
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self.evaluate()

  def type_from_states(self, states):
    """Get node type of a list of states and assert they are the same."""
    types = [state.get_type() for state in states]
    for t in types:
      assert types[0] == t, f'types: {types}'
    return types[0]

  def is_terminal_from_states(self, states):
    """Get is_terminal of a list of states and assert they are the same."""
    is_terminal_from_states_ = [state.is_terminal() for state in states]
    for is_terminal in is_terminal_from_states_:
      assert is_terminal_from_states_[0] == is_terminal
    return is_terminal_from_states_[0]

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

    while not self.is_terminal_from_states(listing_states):
      new_listing_states = []
      new_distribution = collections.defaultdict(float)

      if self.type_from_states(listing_states) == pyspiel.StateType.CHANCE:
        for mfg_state in listing_states:
          for action, prob in mfg_state.chance_outcomes():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            # As a state can be the child of two different parent states, we do
            # not add it in the new states if it has already been seen in this
            # iteration.
            if new_mfg_state_str not in new_distribution:
              new_listing_states.append(new_mfg_state)
            new_distribution[new_mfg_state_str] += prob * self.value(mfg_state)

      elif (self.type_from_states(listing_states) ==
            pyspiel.StateType.MEAN_FIELD):
        for mfg_state in listing_states:
          dist_to_register = mfg_state.distribution_support()
          dist = list(map(self.value_str, dist_to_register))
          assert (sum(dist) - len(self._root_states)) < 1e-4, (
            "Sum of probabilities of all possible states should be the number of "
            f"population, it is {sum(dist)}.")
          new_mfg_state = mfg_state.clone()
          new_mfg_state.update_distribution(dist)
          new_listing_states.append(new_mfg_state)
          new_distribution[new_mfg_state.observation_string(
              pyspiel.PlayerId.DEFAULT_PLAYER_ID)] = self.value(mfg_state)

      else:
        assert self.type_from_states(
            listing_states) == pyspiel.StateType.DECISION
        for mfg_state in listing_states:
          for action, prob in self._policy.action_probabilities(
              mfg_state).items():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            # As a state can be the child of two different parent states, we do
            # not add it in the new states if it has already been seen in this
            # iteration.
            if new_mfg_state_str not in new_distribution:
              new_listing_states.append(new_mfg_state)
            new_distribution[new_mfg_state_str] += prob * self.value(mfg_state)

      assert all(state_str not in self.distribution
            for state_str in new_distribution), (
          "Some new states have the same string representation of old states.")
      self.distribution.update(new_distribution)
      sum_state_probabilities = sum(map(self.value, new_listing_states))
      assert abs(sum_state_probabilities - len(self._root_states)) < 1e-4, (
        "Sum of probabilities of all possible states should be the number of "
        f"population, it is {sum_state_probabilities}.")
      listing_states = new_listing_states

  def value(self, mfg_state):
    return self.value_str(
        mfg_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))

  def value_str(self, mfg_state_str):
    v = self.distribution.get(mfg_state_str)
    if v is None:
      # Check this because self.distribution is a default dict.
      raise ValueError(f'Distribution not computed for state {mfg_state_str}')
    return v
