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

A joint policy is a list of `num_players` policies.
This files enables to compute mixtures of such joint-policies to get a new
policy.
"""

import numpy as np
from open_spiel.python import policy
import pyspiel


def _aggregate_at_state(joint_policies, state, player):
  """Returns {action: prob} for `player` in `state` for all joint policies.

  Args:
    joint_policies: List of joint policies.
    state: Openspiel State
    player: Current Player

  Returns:
    {action: prob} for `player` in `state` for all joint policies.
  """
  return [
      joint_policy[player].action_probabilities(state, player_id=player)
      for joint_policy in joint_policies
  ]


class _DictPolicy(policy.Policy):
  """A callable policy class."""

  def __init__(self, game, policies_as_dict):
    """Constructs a policy function.

    Arguments:
      game: OpenSpiel game.
      policies_as_dict: A list of `num_players` policy objects {action: prob}.
    """
    self._game = game
    self._game_type = game.get_type()
    self._policies_as_dict = policies_as_dict

  def _state_key(self, state, player_id=None):
    """Returns the key to use to look up this (state, player_id) pair."""
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

  @property
  def policies(self):
    return self._policies_as_dict

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
    if player_id is None:
      player_id = state.current_player()
    return self._policies_as_dict[player_id][state_key]


class JointPolicyAggregator(object):
  """Main aggregator object."""

  def __init__(self, game, epsilon=1e-40):
    self._game = game
    self._game_type = game.get_type()
    self._num_players = self._game.num_players()
    self._joint_policies = None
    self._policy = {}  # A Dict from info-state to {action: prob}
    self._epsilon = epsilon

  def _state_key(self, state, player_id=None):
    """Returns the key to use to look up this (state, player) pair."""
    if self._game_type.provides_information_state_string:
      if player_id is None:
        return state.information_state_string()
      else:
        return state.information_state_string(player_id)
    elif self._game_type.provides_observation_string:
      if player_id is None:
        return state.observation()
      else:
        return state.observation(player_id)
    else:
      return str(state)

  def aggregate(self, pids, joint_policies, weights):
    r"""Computes the weighted-mixture of the joint policies.

    Let P of shape [num_players] be the joint policy, and W some weights.
    Let N be the number of policies (i.e. len(policies)).
    We return the policy P' such that for all state `s`:

    P[s] ~ \sum_{i=0}^{N-1} (policies[i][player(s)](s) * weights[i] *
                             reach_prob(s, policies[i]))

    Arguments:
      pids: Spiel player ids of the players the strategies belong to.
      joint_policies: List of list of policies (One list per joint strategy)
      weights: List of weights to attach to each joint strategy.

    Returns:
      A _DictPolicy, a callable object representing the policy.
    """
    aggr_policies = []
    self._joint_policies = joint_policies

    # To do(pmuller): We should be able to do a single recursion.
    for pid in pids:
      aggr_policies.append(self._sub_aggregate(pid, weights))
    return _DictPolicy(self._game, aggr_policies)

  def _sub_aggregate(self, pid, weights):
    """Aggregate the list of policies for one player.

    Arguments:
      pid: Spiel player id of the player the strategies belong to.
      weights: List of weights to attach to each joint strategy.

    Returns:
      A _DictPolicy, a callable object representing the policy.
    """

    # string of state -> probs list
    self._policy = {}

    state = self._game.new_initial_state()
    self._rec_aggregate(pid, state, weights.copy())

    # Now normalize
    for key in self._policy:
      actions, probabilities = zip(*self._policy[key].items())
      new_probs = [prob + self._epsilon for prob in probabilities]
      denom = sum(new_probs)
      for i in range(len(actions)):
        self._policy[key][actions[i]] = new_probs[i] / denom
    return self._policy

  def _rec_aggregate(self, pid, state, my_reaches):
    """Recursively traverse game tree to compute aggregate policy."""
    if state.is_terminal():
      return

    if state.is_simultaneous_node():
      assert (self._game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS
             ), "Game must be simultaneous-move"
      assert (self._game_type.chance_mode == pyspiel.GameType.ChanceMode
              .DETERMINISTIC), "Chance nodes not supported"
      assert (self._game_type.information == pyspiel.GameType.Information
              .ONE_SHOT), "Only one-shot NFGs supported"
      policies = _aggregate_at_state(self._joint_policies, state, pid)
      state_key = self._state_key(state, pid)

      self._policy[state_key] = {}

      for player_policies, weight in zip(policies, my_reaches):
        player_policy = player_policies[pid]
        for action in player_policy.keys():
          if action in self._policy[state_key]:
            self._policy[state_key][action] += weight * player_policy[action]
          else:
            self._policy[state_key][action] = weight * player_policy[action]
      # No recursion because we only support one shot simultaneous games.
      return

    if state.is_chance_node():
      for action in state.legal_actions():
        new_state = state.child(action)
        self._rec_aggregate(pid, new_state, my_reaches)
      return

    current_player = state.current_player()

    state_key = self._state_key(state, current_player)
    action_probabilities_list = _aggregate_at_state(self._joint_policies, state,
                                                    current_player)
    if pid == current_player:
      # update the current node
      # will need the observation to query the policies
      if state not in self._policy:
        self._policy[state_key] = {}

    for action in state.legal_actions():
      new_reaches = np.copy(my_reaches)
      if pid == current_player:
        for idx, state_action_probs in enumerate(action_probabilities_list):
          # compute the new reach for each policy for this action
          new_reaches[idx] *= state_action_probs.get(action, 0)
          # add reach * prob(a) for this policy to the computed policy
          if action in self._policy[state_key].keys():
            self._policy[state_key][action] += new_reaches[idx]
          else:
            self._policy[state_key][action] = new_reaches[idx]

      # recurse
      self._rec_aggregate(pid, state.child(action), new_reaches)
