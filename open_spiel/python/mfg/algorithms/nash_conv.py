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

"""Implementation of Nash Conv metric for a policy.

In the context of mean field games, the Nash Conv is the difference between:
- the value of a policy against the distribution of that policy,
- and the best response against the distribution of the policy.
"""

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value


class NashConv(object):
  """Computes the Nash Conv of a policy."""

  def __init__(self, game, policy: policy_std.Policy, root_state=None):
    """Initializes the nash conv.

    Args:
      game: The game to analyze.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    self._game = game
    self._policy = policy
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distrib = distribution.DistributionPolicy(
        self._game, self._policy, root_state=root_state)
    self._pi_value = policy_value.PolicyValue(
        self._game,
        self._distrib,
        self._policy,
        value.TabularValueFunction(self._game),
        root_state=root_state)
    self._br_value = best_response_value.BestResponse(
        self._game,
        self._distrib,
        value.TabularValueFunction(self._game),
        root_state=root_state)

  def nash_conv(self):
    """Returns the nash conv.

    Returns:
      A float representing the nash conv for the policy.
    """
    return sum([
        self._br_value.eval_state(state) - self._pi_value.eval_state(state)
        for state in self._root_states
    ])

  def br_values(self):
    """Returns the best response values to the policy distribution.

    Returns:
      A List[float] representing the best response values for a policy
        distribution.
    """
    return [self._br_value.eval_state(state) for state in self._root_states]
