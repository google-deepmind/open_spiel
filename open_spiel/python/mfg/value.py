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

# Lint as python3
"""Representation of a value for a game.

This is a standard representation for passing value functions into algorithms,
with currently the following implementations:

The main way of using a value is to call `value(state)`
or `value(state, action)`.

We will prevent calling a value on a state action on a MEAN_FIELD state.
"""


class ValueFunction(object):
  """Base class for values.

  A ValueFunction is something that returns a value given
  a state of the world or a state and an action.

  Attributes:
    game: the game for which this ValueFunction derives
  """

  def __init__(self, game):
    """Initializes a value.

    Args:
      game: the game for which this value derives
    """
    self.game = game

  def value(self, state, action=None):
    """Returns a float representing a value.

    Args:
      state: A `pyspiel.State` object.
      action: may be None or a legal action

    Returns:
      A value for the state (and eventuallu state action pair).
    """
    raise NotImplementedError()

  def __call__(self, state, action=None):
    """Turns the value into a callable.

    Args:
      state: The current state of the game.
      action: may be None or a legal action

    Returns:
      Float: the value of the state or the state action pair.
    """
    return self.value(state, action=action)

