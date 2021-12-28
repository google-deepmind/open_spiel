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

"""Reinforcement Learning (RL) tools Open Spiel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ValueSchedule(metaclass=abc.ABCMeta):
  """Abstract base class changing (decaying) values."""

  @abc.abstractmethod
  def __init__(self):
    """Initialize the value schedule."""

  @abc.abstractmethod
  def step(self):
    """Apply a potential change in the value.

    This method should be called every time the agent takes a training step.

    Returns:
      the value after the step.
    """

  @property
  @abc.abstractmethod
  def value(self):
    """Return the current value."""


class ConstantSchedule(ValueSchedule):
  """A schedule that keeps the value constant."""

  def __init__(self, value):
    super(ConstantSchedule, self).__init__()
    self._value = value

  def step(self):
    return self._value

  @property
  def value(self):
    return self._value


class LinearSchedule(ValueSchedule):
  """A simple linear schedule."""

  def __init__(self, init_val, final_val, num_steps):
    """A simple linear schedule.

    Once the the number of steps is reached, value is always equal to the final
    value.

    Arguments:
      init_val: the initial value.
      final_val: the final_value
      num_steps: the number of steps to get from the initial to final value.
    """
    super(LinearSchedule, self).__init__()
    self._value = init_val
    self._final_value = final_val
    assert isinstance(num_steps, int)
    self._num_steps = num_steps
    self._steps_taken = 0
    self._increment = (final_val - init_val) / num_steps

  def step(self):
    self._steps_taken += 1
    if self._steps_taken < self._num_steps:
      self._value += self._increment
    elif self._steps_taken == self._num_steps:
      self._value = self._final_value
    return self._value

  @property
  def value(self):
    return self._value
