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
"""Representation of a distribution for a game.

This is a standard representation for passing distributions into algorithms,
with currently the following implementations:

The main way of using a distribution is to call `value(state)`.
"""


class Distribution(object):
  """Base class for distributions.

  This represents a probability distribution over the states of a game.

  Attributes:
    game: the game for which this distribution is derives
  """

  def __init__(self, game):
    """Initializes a distribution.

    Args:
      game: the game for which this distribution is derives
    """
    self.game = game

  def value(self, state):
    """Returns the probability of the distribution on the state.

    Args:
      state: A `pyspiel.State` object.

    Returns:
      A `float`.
    """
    raise NotImplementedError()

  def value_str(self, state_str, default_value=None):
    """Returns the probability of the distribution on the state string given.

    Args:
      state_str: A string.
      default_value: If not None, return this value if the state is not in the
        support of the distribution.

    Returns:
      A `float`.
    """
    raise NotImplementedError()

  def __call__(self, state):
    """Turns the distribution into a callable.

    Args:
      state: The current state of the game.

    Returns:
      Float: probability.
    """
    return self.value(state)
