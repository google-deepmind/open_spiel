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
        "min": float(self.min),
        "max": float(self.max),
        "avg": float(self.avg),
        "std_dev": self.std_dev,
    }

  def __str__(self):
    if self.num == 0:
      return "num=0"
    return "sum: %.4f, avg: %.4f, dev: %.4f, min: %.4f, max: %.4f, num: %d" % (
        self._sum, self.avg, self.std_dev, self.min, self.max, self.num)


class SlidingWindowAccumulator(object):
  """A utility object to compute the mean of a sliding window of values."""

  def __init__(self, max_window_size: int):
    self._max_window_size = max_window_size
    self._index = -1
    self._values = []

  def add(self, value: float):
    if len(self._values) < self._max_window_size:
      self._values.append(value)
      self._index += 1
    else:
      self._values[self._index] = value
      self._index += 1
      if self._index >= self._max_window_size:
        self._index = 0

  def mean(self):
    return sum(self._values) / len(self._values)


class StatCounter:
  """An object for incrementally counting statistics.

  Uses Welford's online algorithm for computing variance.
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

  Note: everything returns 0 if there are no data points. While technically
  incorrect, this makes workin with the StatCounter objects easier (i.e. they
  can print values even with zero data).
  """

  def __init__(self):
    self._sum = 0
    self._m2 = 0
    self._mean = 0
    self._n = 0
    self._max = -math.inf
    self._min = math.inf

  def add(self, value: float):
    self.sum = self._sum + value
    self._n += 1

    delta = value - self._mean
    self._mean = self._sum / self._n
    self._m2 = self._m2 + delta*(value - self._mean)

    self._min = min(self._min, value)
    self._max = max(self._max, value)

  def variance(self):
    if self._n == 0: return 0   # technically wrong but easier to work with
    return self._m2 / self._n

  def sample_variance(self):
    if self._n < 2: return 0
    return self._m2 / (self._n - 1)

  def stddev(self):
    return math.sqrt(self.variance())

  def mean(self):
    if self._n == 0: return 0
    return self._mean

  @property
  def max(self):
    return self._max

  @property
  def min(self):
    return self._min

  @property
  def n(self):
    return self.n

  def ci95(self):
    if self._n == 0: return 0
    return 1.96 * self.stddev() / math.sqrt(self._n)


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
