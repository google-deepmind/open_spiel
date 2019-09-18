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

"""Computes a Best-Response policy.

The goal if this file is to be the main entry-point for BR APIs in Python.

TODO(author2): Also include computation using the more efficient C++
`TabularBestResponse` implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from open_spiel.python import policy as pyspiel_policy


def _memoize_method(method):
  """Memoize a single-arg instance method using an on-object cache."""
  cache_name = "cache_" + method.__name__

  def wrap(self, arg):
    key = str(arg)
    cache = vars(self).setdefault(cache_name, {})
    if key not in cache:
      cache[key] = method(self, arg)
    return cache[key]

  return wrap


class BestResponsePolicy(pyspiel_policy.Policy):
  """Computes the best response to a specified strategy."""

  def __init__(self, game, player_id, policy, root_state=None):
    """Initializes the best-response calculation.

    Args:
      game: The game to analyze.
      player_id: The player id of the best-responder.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
    """
    self._num_players = game.num_players()
    self._player_id = player_id
    self._policy = policy
    if root_state is None:
      root_state = game.new_initial_state()
    self._root_state = root_state
    self.infosets = self.info_sets(root_state)

  def info_sets(self, state):
    """Returns a dict of infostatekey to list of (state, cf_probability)."""
    infosets = collections.defaultdict(list)
    for s, p in self.decision_nodes(state):
      infosets[s.information_state(self._player_id)].append((s, p))
    return dict(infosets)

  def decision_nodes(self, parent_state):
    """Yields a (state, cf_prob) pair for each descendant decision node."""
    if not parent_state.is_terminal():
      if parent_state.current_player() == self._player_id:
        yield (parent_state, 1.0)
      for action, p_action in self.transitions(parent_state):
        for state, p_state in self.decision_nodes(parent_state.child(action)):
          yield (state, p_state * p_action)

  def transitions(self, state):
    """Returns a list of (action, cf_prob) pairs from the specifed state."""
    if state.current_player() == self._player_id:
      # Counterfactual reach probabilities exclude the best-responder's actions,
      # hence return probability 1.0 for every action.
      return [(action, 1.0) for action in state.legal_actions()]
    elif state.is_chance_node():
      return state.chance_outcomes()
    else:
      return list(self._policy.action_probabilities(state).items())

  @_memoize_method
  def value(self, state):
    """Returns the value of the specified state to the best-responder."""
    if state.is_terminal():
      return state.player_return(self._player_id)
    elif state.current_player() == self._player_id:
      action = self.best_response_action(
          state.information_state(self._player_id))
      return self.q_value(state, action)
    else:
      return sum(p * self.q_value(state, a) for a, p in self.transitions(state))

  def q_value(self, state, action):
    """Returns the value of the (state, action) to the best-responder."""
    return self.value(state.child(action))

  @_memoize_method
  def best_response_action(self, infostate):
    """Returns the best response for this information state."""
    infoset = self.infosets[infostate]
    # Get actions from the first (state, cf_prob) pair in the infoset list.
    # Return the best action by counterfactual-reach-weighted state-value.
    return max(
        infoset[0][0].legal_actions(),
        key=lambda a: sum(cf_p * self.q_value(s, a) for s, cf_p in infoset))

  def action_probabilities(self, state, player_id=None):
    """Returns the policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
    if player_id is None:
      player_id = state.current_player()
    return {self.best_response_action(state.information_state(player_id)): 1}
