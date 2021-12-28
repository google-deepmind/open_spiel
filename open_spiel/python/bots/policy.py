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

"""A bot that samples from legal actions based on a policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel


class PolicyBot(pyspiel.Bot):
  """Samples an action from action probabilities based on a policy.

  This bot plays actions as specified by the underlying Policy. Problems may
  occur if the policy assigns non-zero probability to invalid actions, or if the
  policy is not complete, or if probabilities don't sum to 1.
  """

  def __init__(self, player_id, rng, policy):
    """Initializes a policy bot.

    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting
        as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
      policy: A policy to get action distributions
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng
    self._policy = policy

  def player_id(self):
    return self._player_id

  def restart_at(self, state):
    pass

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.

    Args:
      state: The current state of the game.

    Returns:
      A `(policy, action)` pair, where policy is a `list` of
      `(action, probability)` pairs for each legal action, with
      `probability` defined by the policy action probabilities.
      The `action` is sampled from the distribution,
      or `pyspiel.INVALID_ACTION` if there are no actions available.
    """
    policy = self._policy.action_probabilities(state, self._player_id)
    action_list = list(policy.keys())
    if not any(action_list):
      return [], pyspiel.INVALID_ACTION

    action = self._rng.choice(action_list, p=list(policy.values()))
    return list(policy.items()), action

  def step(self, state):
    return self.step_with_policy(state)[1]
