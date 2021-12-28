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

"""Computes the value of a given policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Union

import numpy as np

from open_spiel.python import policy


def _transitions(state, policies):
  """Returns iterator over (action, prob) from the given state."""
  if state.is_chance_node():
    return state.chance_outcomes()
  elif state.is_simultaneous_node():
    return policy.joint_action_probabilities(state, policies)
  else:
    player = state.current_player()
    return policies[player].action_probabilities(state).items()


def policy_value(state,
                 policies: Union[List[policy.Policy], policy.Policy],
                 probability_threshold: float = 0):
  """Returns the expected values for the state for players following `policies`.

  Computes the expected value of the`state` for each player, assuming player `i`
  follows the policy given in `policies[i]`.

  Args:
    state: A `pyspiel.State`.
    policies: A `list` of `policy.Policy` objects, one per player for sequential
      games, one policy for simulatenous games.
    probability_threshold: only sum over entries with prob greater than this
      (default: 0).

  Returns:
    A `numpy.array` containing the expected value for each player.
  """
  if state.is_terminal():
    return np.array(state.returns())
  else:
    return sum(prob * policy_value(policy.child(state, action), policies)
               for action, prob in _transitions(state, policies)
               if prob > probability_threshold)
