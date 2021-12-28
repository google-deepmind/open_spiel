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

# Lint as: python3
"""Some basic stats classes."""

import math
from typing import List


class BasicStats(object):
  """A set of statistics about a single value series."""
  __slots__ = ("_num", "_min", "_max", "_sum", "_sum_sq")

  def __init__(self):
    self.reset()

  def reset(self):
    self._num = 0
    self._min = float("inf")
    self._max = float("-inf")
    self._sum = 0
    self._sum_sq = 0

  def add(self, val: float):
    self._num += 1
    if self._min > val:
      self._min = val
    if self._max < val:
      self._max = val
    self._sum += val
    self._sum_sq += val**2

  @property
  def num(self):
    return self._num

  @property
  def min(self):
    return 0 if self._num == 0 else self._min

  @property
  def max(self):
    return 0 if self._num == 0 else self._max

  @property
  def avg(self):
    return 0 if self._num == 0 else self._sum / self._num

  @property
  def std_dev(self):
    """Standard deviation."""
    if self._num == 0:
      return 0
    return math.sqrt(
        max(0, self._sum_sq / self._num - (self._sum / self._num)**2))

  def merge(self, other: "BasicStats"):
    # pylint: disable=protected-access
    self._num += other._num
    self._min = min(self._min, other._min)
    self._max = max(self._max, other._max)
    self._sum += other._sum
    self._sum_sq += other._sum_sq
    # pylint: enable=protected-access

  @property
  def as_dict(self):
    return {
        "num": self.num,
        "min": self.min,
        "max": self.max,
        "avg": self.avg,
        "std_dev": self.std_dev,
    }

  def __str__(self):
    if self.num == 0:
      return "num=0"
    return "sum: %.4f, avg: %.4f, dev: %.4f, min: %.4f, max: %.4f, num: %d" % (
        self.sum, self.avg, self.dev, self.min, self.max, self.num)


class HistogramNumbered:
  """Track a histogram of occurences for `count` buckets.

  You need to decide how to map your data into the buckets. Mainly useful for
  scalar values.
  """

  def __init__(self, num_buckets: int):
    self._counts = [0] * num_buckets

  def reset(self):
    self._counts = [0] * len(self._counts)

  def add(self, bucket_id: int):
    self._counts[bucket_id] += 1

  @property
  def data(self):
    return self._counts


class HistogramNamed:
  """Track a histogram of occurences for named buckets.

  Same as HistogramNumbered, but each bucket has a name associated with it.
  Mainly useful for categorical values.
  """

  def __init__(self, bucket_names: List[str]):
    self._names = bucket_names
    self.reset()

  def reset(self):
    self._counts = [0] * len(self._names)

  def add(self, bucket_id: int):
    self._counts[bucket_id] += 1

  @property
  def data(self):
    return {
        "counts": self._counts,
        "names": self._names,
    }
