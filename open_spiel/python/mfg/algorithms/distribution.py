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
      root_state = game.new_initial_state()
    self._root_state = root_state
    self.evaluate()

  def player_id_from_states(self, states):
    """Get the current player of a list of states and assert they are the same."""
    player_id_from_states_ = [state.current_player() for state in states]
    for player_id in player_id_from_states_:
      assert player_id_from_states_[0] == player_id
    return player_id_from_states_[0]

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
    listing_states = [self._root_state]
    # Maps state strings to floats. For each group of states at a given timestep
    # and given player ID, these floats represent a probability distribution.
    self.distribution_over_states = collections.defaultdict(float)
    self.distribution_over_states[self._root_state.observation_string(
        pyspiel.PlayerId.DEFAULT_PLAYER_ID)] = 1.0

    while not self.is_terminal_from_states(listing_states):
      new_listing_states = []
      new_distribution_over_states = collections.defaultdict(float)

      if self.player_id_from_states(listing_states) == pyspiel.PlayerId.CHANCE:
        for mfg_state in listing_states:
          for action, prob in mfg_state.chance_outcomes():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            # as a state can be the child of two different parent states, we do
            # not add it in the new states if it has already been seen in this
            # iteration.
            if new_mfg_state_str not in new_distribution_over_states:
              new_listing_states.append(new_mfg_state)
            new_distribution_over_states[
                new_mfg_state_str] += prob * self.distribution_over_states[
                    mfg_state.observation_string(
                        pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

      elif self.player_id_from_states(
          listing_states) == pyspiel.PlayerId.MEAN_FIELD:
        for mfg_state in listing_states:
          dist_to_register = mfg_state.distribution_support()
          dist = [
              self.distribution_over_states[str_state]
              for str_state in dist_to_register
          ]
          new_mfg_state = mfg_state.clone()
          new_mfg_state.update_distribution(dist)
          new_listing_states.append(new_mfg_state)
          new_distribution_over_states[new_mfg_state.observation_string(
              pyspiel.PlayerId.DEFAULT_PLAYER_ID
          )] = self.distribution_over_states[mfg_state.observation_string(
              pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

      else:
        assert self.player_id_from_states(
            listing_states) == pyspiel.PlayerId.DEFAULT_PLAYER_ID, (
          f"The player id should be {pyspiel.PlayerId.DEFAULT_PLAYER_ID}")
        for mfg_state in listing_states:
          for action, prob in self._policy.action_probabilities(
              mfg_state).items():
            new_mfg_state = mfg_state.child(action)
            new_mfg_state_str = new_mfg_state.observation_string(
                pyspiel.PlayerId.DEFAULT_PLAYER_ID)
            # as a state can be the child of two different parent states, we do
            # not add it in the new states if it has already been seen in this
            # iteration.
            if new_mfg_state_str not in new_distribution_over_states:
              new_listing_states.append(new_mfg_state)
            new_distribution_over_states[
                new_mfg_state_str] += prob * self.distribution_over_states[
                    mfg_state.observation_string(
                        pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

      assert all(state_str not in self.distribution_over_states
                 for state_str in new_distribution_over_states), (
        "Some new states have the same string representation of old states.")
      self.distribution_over_states.update(new_distribution_over_states)
      sum_state_probabilities = sum(map(self.value, new_listing_states))
      assert abs(sum_state_probabilities - 1) < 1e-4, (
        "Sum of probabilities off all possible states should be 1, it is "
        f"{sum_state_probabilities}.")
      listing_states = new_listing_states

  def value(self, mfg_state):
    return self.distribution_over_states[mfg_state.observation_string(
        pyspiel.PlayerId.DEFAULT_PLAYER_ID)]

  def value_str(self, mfg_state_str):
    return self.distribution_over_states[mfg_state_str]
