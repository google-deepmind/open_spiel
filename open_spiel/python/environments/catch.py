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

"""Catch reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import rl_environment

# Actions
NOOP = 0
LEFT = 1
RIGHT = 2

_Point = collections.namedtuple("Point", ["x", "y"])


class Environment(object):
  """A catch reinforcement learning environment.

  The implementation considers illegal actions: trying to move the paddle in the
  wall direction when next to a wall will incur in an invalid action and an
  error will be purposely raised.
  """

  def __init__(self, discount=1.0, width=5, height=10, seed=None):
    self._rng = np.random.RandomState(seed)
    self._width = width
    self._height = height
    self._should_reset = True
    self._num_actions = 3

    # Discount returned at non-initial steps.
    self._discounts = [discount] * self.num_players

  def reset(self):
    """Resets the environment."""
    self._should_reset = False
    self._ball_pos = _Point(x=self._rng.randint(0, self._width - 1), y=0)
    self._paddle_pos = _Point(
        x=self._rng.randint(0, self._width - 1), y=self._height - 1)

    legal_actions = [NOOP]
    if self._paddle_pos.x > 0:
      legal_actions.append(LEFT)
    if self._paddle_pos.x < self._width - 1:
      legal_actions.append(RIGHT)

    observations = {
        "info_state": [self._get_observation()],
        "legal_actions": [legal_actions],
        "current_player": 0,
    }

    return rl_environment.TimeStep(
        observations=observations,
        rewards=None,
        discounts=None,
        step_type=rl_environment.StepType.FIRST)

  def step(self, actions):
    """Updates the environment according to `actions` and returns a `TimeStep`.

    Args:
      actions: A singleton list with an integer, or an integer, representing the
        action the agent took.

    Returns:
      A `rl_environment.TimeStep` namedtuple containing:
        observation: singleton list of dicts containing player observations,
            each corresponding to `observation_spec()`.
        reward: singleton list containing the reward at this timestep, or None
            if step_type is `rl_environment.StepType.FIRST`.
        discount: singleton list containing the discount in the range [0, 1], or
            None if step_type is `rl_environment.StepType.FIRST`.
        step_type: A `rl_environment.StepType` value.
    """
    if self._should_reset:
      return self.reset()

    if isinstance(actions, list):
      action = actions[0]
    elif isinstance(actions, int):
      action = actions
    else:
      raise ValueError("Action not supported.", actions)

    # Update paddle position
    x, y = self._paddle_pos.x, self._paddle_pos.y
    if action == LEFT:
      x -= 1
    elif action == RIGHT:
      x += 1
    elif action != NOOP:
      raise ValueError("unrecognized action ", action)

    assert 0 <= x < self._width, (
        "Illegal action detected ({}), new state: ({},{})".format(action, x, y))
    self._paddle_pos = _Point(x, y)

    # Update ball position
    x, y = self._ball_pos.x, self._ball_pos.y
    if y == self._height - 1:
      done = True
      reward = 1.0 if x == self._paddle_pos.x else -1.0
    else:
      done = False
      y += 1
      reward = 0.0
      self._ball_pos = _Point(x, y)

    # Return observation
    step_type = (
        rl_environment.StepType.LAST if done else rl_environment.StepType.MID)
    self._should_reset = step_type == rl_environment.StepType.LAST

    legal_actions = [NOOP]
    if self._paddle_pos.x > 0:
      legal_actions.append(LEFT)
    if self._paddle_pos.x < self._width - 1:
      legal_actions.append(RIGHT)

    observations = {
        "info_state": [self._get_observation()],
        "legal_actions": [legal_actions],
        "current_player": 0,
    }

    return rl_environment.TimeStep(
        observations=observations,
        rewards=[reward],
        discounts=self._discounts,
        step_type=step_type)

  def _get_observation(self):
    board = np.zeros((self._height, self._width), dtype=np.float32)
    board[self._ball_pos.y, self._ball_pos.x] = 1.0
    board[self._paddle_pos.y, self._paddle_pos.x] = 1.0
    return board.flatten()

  def observation_spec(self):
    """Defines the observation provided by the environment.

    Each dict member will contain its expected structure and shape.

    Returns:
      A specification dict describing the observation fields and shapes.
    """
    return dict(
        info_state=tuple([self._height * self._width]),
        legal_actions=(self._num_actions,),
        current_player=(),
    )

  def action_spec(self):
    """Defines action specifications.

    Specifications include action boundaries and their data type.

    Returns:
      A specification dict containing action properties.
    """
    return dict(num_actions=self._num_actions, min=0, max=2, dtype=int)

  @property
  def num_players(self):
    return 1

  @property
  def is_turn_based(self):
    return False
