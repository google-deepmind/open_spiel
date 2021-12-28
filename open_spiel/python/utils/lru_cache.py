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
"""A Least Recently Used cache."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class CacheInfo(collections.namedtuple("CacheInfo", [
    "hits", "misses", "size", "max_size"])):
  """Info for LRUCache."""

  @property
  def usage(self):
    return self.size / self.max_size if self.max_size else 0

  @property
  def total(self):
    return self.hits + self.misses

  @property
  def hit_rate(self):
    return self.hits / self.total if self.total else 0


class LRUCache(object):
  """A Least Recently Used cache.

  This is more general than functools.lru_cache since that one requires the
  key to also be the input to the function to generate the value, which
  isn't possible when the input is not hashable, eg a numpy.ndarray.
  """

  def __init__(self, max_size):
    self._max_size = max_size
    self._data = collections.OrderedDict()
    self._hits = 0
    self._misses = 0

  def clear(self):
    self._data.clear()
    self._hits = 0
    self._misses = 0

  def make(self, key, fn):
    """Return the value, either from cache, or make it and save it."""
    try:
      val = self._data.pop(key)  # Take it out.
      self._hits += 1
    except KeyError:
      self._misses += 1
      val = fn()
      if len(self._data) >= self._max_size:
        self._data.popitem(False)
    self._data[key] = val  # Insert/reinsert it at the back.
    return val

  def get(self, key):
    """Get the value and move it to the back, or return None on a miss."""
    try:
      val = self._data.pop(key)  # Take it out.
      self._data[key] = val  # Reinsert it at the back.
      self._hits += 1
      return val
    except KeyError:
      self._misses += 1
      return None

  def set(self, key, val):
    """Set the value."""
    self._data.pop(key, None)  # Take it out if it existed.
    self._data[key] = val  # Insert/reinsert it at the back.
    if len(self._data) > self._max_size:
      self._data.popitem(False)
    return val

  def info(self):
    return CacheInfo(self._hits, self._misses, len(self._data), self._max_size)

  def __len__(self):
    return len(self._data)
