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

"""Policy aggregator.

Turns a weighted sum of N policies into a realization-equivalent single
policy by sweeping over the state space.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from open_spiel.python import policy
import pyspiel


class PolicyFunction(policy.Policy):
  """A callable policy class."""

  def __init__(self, pids, policies, game):
    """Construct a policy function.

    Arguments:
      pids: spiel player id of players these policies belong to.
      policies: a list of dictionaries of keys (stringified binary observations)
        to a list of probabilities for each move uid (between 0 and max_moves -
        1).
      game: OpenSpiel game.
    """
    super().__init__(game, pids)
    self._policies = policies
    self._game_type = game.get_type()

  def _state_key(self, state, player_id=None):
    """Returns the key to use to look up this (state, player_id) pair."""
    if self._game_type.provides_information_state_string:
      if player_id is None:
        return state.information_state_string()
      else:
        return state.information_state_string(player_id)
    elif self._game_type.provides_observation_tensor:
      if player_id is None:
        return state.observation_tensor()
      else:
        return state.observation_tensor(player_id)
    else:
      return str(state)

  @property
  def policy(self):
    return self._policies

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
    state_key = self._state_key(state, player_id=player_id)
    if state.is_simultaneous_node():
      # Policy aggregator doesn't yet support simultaneous moves nodes.
      # The below lines are one step towards that direction.
      result = []
      for player_pol in self._policies:
        result.append(player_pol[state_key])
      return result
    if player_id is None:
      player_id = state.current_player()
    return self._policies[player_id][state_key]


class PolicyPool(object):
  """Transforms a list of list of policies (One list per player) to callable."""

  def __init__(self, policies):
    """Transforms a list of list of policies (One list per player) to callable.

    Args:
      policies: List of list of policies.
    """
    self._policies = policies

  def __call__(self, state, player):
    return [
        a.action_probabilities(state, player_id=player)
        for a in self._policies[player]
    ]


class PolicyAggregator(object):
  """Main aggregator object."""

  def __init__(self, game, epsilon=1e-40):
    self._game = game
    self._game_type = game.get_type()
    self._num_players = self._game.num_players()
    self._policy_pool = None
    self._weights = None
    self._policy = {}
    self._epsilon = epsilon

  def _state_key(self, state, player_id=None):
    """Returns the key to use to look up this (state, player) pair."""
    # TODO(somidshafiei): fuse this with the identical PolicyFunction._state_key
    if self._game_type.provides_information_state_string:
      if player_id is None:
        return state.information_state_string()
      else:
        return state.information_state_string(player_id)
    elif self._game_type.provides_observation_string:
      if player_id is None:
        return state.observation_string()
      else:
        return state.observation_string(player_id)
    else:
      return str(state)

  def aggregate(self, pids, policies, weights):
    """Aggregate the list of policies for each player.

    Arguments:
      pids: the spiel player ids of the players the strategies belong to.
      policies: List of list of policies (One list per player)
      weights: the list of weights to attach to each policy.

    Returns:
      A PolicyFunction, a callable object representing the policy.
    """
    aggr_policies = []

    for pid in pids:
      aggr_policies.append(self._sub_aggregate(pid, policies, weights))
    return PolicyFunction(pids, aggr_policies, self._game)

  def _sub_aggregate(self, pid, policies, weights):
    """Aggregate the list of policies for one player.

    Arguments:
      pid: the spiel player id of the player the strategies belong to.
      policies: List of list of policies (One list per player)
      weights: the list of weights to attach to each policy.

    Returns:
      A PolicyFunction, a callable object representing the policy.
    """
    self._policy_pool = PolicyPool(policies)
    # ipdb.set_trace()

    assert self._policy_pool is not None
    self._weights = weights
    # string of state -> probs list
    self._policy = {}

    state = self._game.new_initial_state()
    my_reaches = weights[:]
    self._rec_aggregate(pid, state, my_reaches)

    # Now normalize
    for key in self._policy:
      actions, probabilities = zip(*self._policy[key].items())
      # Add some small proba mass to avoid divide by zero, which happens for
      # games with low reach probabilities for certain states (keys)
      new_probs = [prob + self._epsilon for prob in probabilities]
      denom = sum(new_probs)
      for i in range(len(actions)):
        self._policy[key][actions[i]] = new_probs[i] / denom
    return self._policy

  def _rec_aggregate(self, pid, state, my_reaches):
    """Recursively traverse game tree to compute aggregate policy."""

    if state.is_terminal():
      return
    elif state.is_simultaneous_node():
      # TODO(author10): this is assuming that if there is a sim.-move state, it is
      #               the only state, i.e., the game is a normal-form game
      def assert_type(cond, msg):
        assert cond, msg
      assert_type(self._game_type.dynamics ==
                  pyspiel.GameType.Dynamics.SIMULTANEOUS,
                  "Game must be simultaneous-move")
      assert_type(self._game_type.chance_mode ==
                  pyspiel.GameType.ChanceMode.DETERMINISTIC,
                  "Chance nodes not supported")
      assert_type(self._game_type.information ==
                  pyspiel.GameType.Information.ONE_SHOT,
                  "Only one-shot NFGs supported")
      policies = self._policy_pool(state, pid)
      state_key = self._state_key(state, pid)
      self._policy[state_key] = {}
      for player_policy, weight in zip(policies, my_reaches[pid]):
        for action in player_policy.keys():
          if action in self._policy[state_key]:
            self._policy[state_key][action] += weight * player_policy[action]
          else:
            self._policy[state_key][action] = weight * player_policy[action]
      return
    elif state.is_chance_node():
      # do not factor in opponent reaches
      outcomes, _ = zip(*state.chance_outcomes())
      for i in range(0, len(outcomes)):
        outcome = outcomes[i]
        new_state = state.clone()
        new_state.apply_action(outcome)
        self._rec_aggregate(pid, new_state, my_reaches)
      return
    else:
      turn_player = state.current_player()

      state_key = self._state_key(state, turn_player)
      legal_policies = self._policy_pool(state, turn_player)
      if pid == turn_player:
        # update the current node
        # will need the observation to query the policies
        if state not in self._policy:
          self._policy[state_key] = {}

      used_moves = []
      for k in range(len(legal_policies)):
        used_moves += [a[0] for a in legal_policies[k].items()]
      used_moves = np.unique(used_moves)

      for uid in used_moves:
        new_reaches = np.copy(my_reaches)
        if pid == turn_player:
          for i in range(len(legal_policies)):
            # compute the new reach for each policy for this action
            new_reaches[turn_player][i] *= legal_policies[i].get(uid, 0)
            # add reach * prob(a) for this policy to the computed policy
            if uid in self._policy[state_key].keys():
              self._policy[state_key][uid] += new_reaches[turn_player][i]
            else:
              self._policy[state_key][uid] = new_reaches[turn_player][i]

        # recurse
        new_state = state.clone()
        new_state.apply_action(uid)
        self._rec_aggregate(pid, new_state, new_reaches)
