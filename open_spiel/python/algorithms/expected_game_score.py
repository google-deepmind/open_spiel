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

"""Computes the value of a given policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from typing import List, Union

import numpy as np

from open_spiel.python import policy


PROBABILITY_THRESHOLD = 0


def policy_value(state, policies: Union[List[policy.Policy], policy.Policy]):
  """Returns the expected values for the state for players following `policies`.

  Computes the expected value of the`state` for each player, assuming player `i`
  follows the policy given in `policies[i]`.

  Args:
    state: A `pyspiel.State`.
    policies: A `list` of `policy.Policy` objects, one per player for sequential
      games, one policy for simulatenous games.

  Returns:
    A `numpy.array` containing the expected value for each player.
  """
  num_players = state.get_game().num_players()

  if state.is_terminal():
    values = np.array(state.returns())
  elif state.is_chance_node():
    values = np.zeros(shape=num_players)
    for action, action_prob in state.chance_outcomes():
      child = state.child(action)
      values += action_prob * policy_value(child, policies)
  elif state.is_simultaneous_node():
    values = np.zeros(shape=num_players)
    action_prob_dicts = []
    for player in range(num_players):
      action_prob_dicts.append(
        policies.action_probabilities(state, player))
    actions = itertools.product(*action_prob_dicts)
    probabilities = itertools.product(
      *[action_prob.values() for action_prob in action_prob_dicts])
    for action_tuple, probability_tuple in zip(actions, probabilities):
      probability = np.prod(probability_tuple)
      action_list = list(action_tuple)
      if probability > PROBABILITY_THRESHOLD:
        child = state.clone()
        child.apply_actions(action_list)
        values += probability * policy_value(child, policies)
  else:
    player = state.current_player()
    values = np.zeros(shape=num_players)
    for action, probability in policies[player].action_probabilities(
        state).items():
      if probability > PROBABILITY_THRESHOLD:
        child = state.child(action)
        values += probability * policy_value(child, policies)
  return values
