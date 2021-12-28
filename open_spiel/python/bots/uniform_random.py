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

"""A bot that chooses uniformly at random from legal actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel


class UniformRandomBot(pyspiel.Bot):
  """Chooses uniformly at random from the available legal actions."""

  def __init__(self, player_id, rng):
    """Initializes a uniform-random bot.

    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting
        as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.

    Args:
      state: The current state of the game.

    Returns:
      A `(policy, action)` pair, where policy is a `list` of
      `(action, probability)` pairs for each legal action, with
      `probability = 1/num_actions`
      The `action` is selected uniformly at random from the legal actions,
      or `pyspiel.INVALID_ACTION` if there are no legal actions available.
    """
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]
    action = self._rng.choice(legal_actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]
