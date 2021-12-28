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

"""A cliff walking single agent reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from open_spiel.python import rl_environment

# Actions
RIGHT, UP, LEFT, DOWN = range(4)


class Environment(object):
  r"""A cliff walking reinforcement learning environment.

  This is a deterministic environment that can be used to test RL algorithms.
  Note there are *no illegal moves* in this environment--if the agent is on the
  edge of the cliff and takes an action which would yield an invalid position,
  the action is ignored (as if there were walls surrounding the cliff).

  Cliff example for height=3 and width=5:

                |   |   |   |   |   |
                |   |   |   |   |   |
                | S | x | x | x | G |

  where `S` is always the starting position, `G` is always the goal and `x`
  represents the zone of high negative reward to be avoided. For this instance,
  the optimum policy is depicted as follows:

                |   |   |   |   |   |
                |-->|-->|-->|-->|\|/|
                |/|\| x | x | x | G |

  yielding a reward of -6 (minus 1 per time step).

  See pages 132 of Rich Sutton's book for details:
  http://www.incompleteideas.net/book/bookdraft2018mar21.pdf
  """

  def __init__(self, height=4, width=8, discount=1.0, max_t=100):
    if height < 2 or width < 3:
      raise ValueError("height must be >= 2 and width >= 3.")
    self._height = height
    self._width = width
    self._legal_actions = [RIGHT, UP, LEFT, DOWN]
    self._should_reset = True
    self._max_t = max_t

    # Discount returned at non-initial steps.
    self._discounts = [discount] * self.num_players

  def reset(self):
    """Resets the environment."""
    self._should_reset = False
    self._time_counter = 0
    self._state = np.array([self._height - 1, 0])

    observations = {
        "info_state": [self._state.copy()],
        "legal_actions": [self._legal_actions],
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
    self._time_counter += 1

    if isinstance(actions, list):
      action = actions[0]
    elif isinstance(actions, int):
      action = actions
    else:
      raise ValueError("Action not supported.", actions)

    dx = 0
    dy = 0
    if action == LEFT:
      dx -= 1
    elif action == RIGHT:
      dx += 1

    if action == UP:
      dy -= 1
    elif action == DOWN:
      dy += 1

    self._state += np.array([dy, dx])
    self._state = self._state.clip(0, [self._height - 1, self._width - 1])

    done = self._is_pit(self._state) or self._is_goal(self._state)
    done = done or self._time_counter >= self._max_t
    # Return observation
    step_type = (
        rl_environment.StepType.LAST if done else rl_environment.StepType.MID)
    self._should_reset = step_type == rl_environment.StepType.LAST

    observations = {
        "info_state": [self._state.copy()],
        "legal_actions": [self._legal_actions],
        "current_player": 0,
    }

    return rl_environment.TimeStep(
        observations=observations,
        rewards=[self._get_reward(self._state)],
        discounts=self._discounts,
        step_type=step_type)

  def _is_goal(self, pos):
    """Check if position is bottom right corner of grid."""
    return pos[0] == self._height - 1 and pos[1] == self._width - 1

  def _is_pit(self, pos):
    """Check if position is in bottom row between start and goal."""
    return (pos[1] > 0 and pos[1] < self._width - 1 and
            pos[0] == self._height - 1)

  def _get_reward(self, pos):
    if self._is_pit(pos):
      return -100.0
    else:
      return -1.0

  def observation_spec(self):
    """Defines the observation provided by the environment.

    Each dict member will contain its expected structure and shape.

    Returns:
      A specification dict describing the observation fields and shapes.
    """
    return dict(
        info_state=tuple([2]),
        legal_actions=(len(self._legal_actions),),
        current_player=(),
    )

  def action_spec(self):
    """Defines action specifications.

    Specifications include action boundaries and their data type.

    Returns:
      A specification dict containing action properties.
    """
    return dict(
        num_actions=len(self._legal_actions),
        min=min(self._legal_actions),
        max=max(self._legal_actions),
        dtype=int,
    )

  @property
  def num_players(self):
    return 1

  @property
  def is_turn_based(self):
    return False
