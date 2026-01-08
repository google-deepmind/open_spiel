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

# Lint as python3
"""Representation of a value for a game.

This is a standard representation for passing value functions into algorithms,
with currently the following implementations:

The main way of using a value is to call `value(state)`
or `value(state, action)`.

We will prevent calling a value on a state action on a MEAN_FIELD state.

The state can be a pyspiel.State object or its string representation. For a
particular ValueFunction instance, you should use only one or the other. The
behavior may be undefined for mixed usage depending on the implementation.
"""

import collections
from typing import Union

import pyspiel

ValueFunctionState = Union[pyspiel.State, str]


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

  def value(self, state: ValueFunctionState, action=None) -> float:
    """Returns a float representing a value.

    Args:
      state: A `pyspiel.State` object or its string representation.
      action: may be None or a legal action

    Returns:
      A value for the state (and eventuallu state action pair).
    """
    raise NotImplementedError()

  def __call__(self, state: ValueFunctionState, action=None) -> float:
    """Turns the value into a callable.

    Args:
      state: A `pyspiel.State` object or its string representation.
      action: may be None or a legal action

    Returns:
      Float: the value of the state or the state action pair.
    """
    return self.value(state, action=action)

  def set_value(self, state: ValueFunctionState, value: float, action=None):
    """Sets the value of the state.

    Args:
      state: A `pyspiel.State` object or its string representation.
      value: Value of the state.
      action: may be None or a legal action
    """
    raise NotImplementedError()

  def has(self, state: ValueFunctionState, action=None) -> bool:
    """Returns true if state(-action) has an explicit value.

    Args:
      state: A `pyspiel.State` object or its string representation.
      action: may be None or a legal action

    Returns:
      True if there is an explicitly specified value.
    """
    raise NotImplementedError()

  def add_value(self, state, value: float, action=None):
    """Adds the value to the current value of the state.

    Args:
      state: A `pyspiel.State` object or its string representation.
      value: Value to add.
      action: may be None or a legal action
    """
    self.set_value(
        state, self.value(state, action=action) + value, action=action)


class TabularValueFunction(ValueFunction):
  """Tabular value function backed by a dictionary."""

  def __init__(self, game):
    super().__init__(game)
    self._values = collections.defaultdict(float)

  def value(self, state: ValueFunctionState, action=None):
    return self._values[(state, action)]

  def set_value(self, state: ValueFunctionState, value: float, action=None):
    self._values[(state, action)] = value

  def has(self, state: ValueFunctionState, action=None):
    return (state, action) in self._values
