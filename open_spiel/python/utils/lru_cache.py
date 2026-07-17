"""A thread-safe Least Recently Used cache."""

import collections
import threading


class CacheInfo(
  collections.namedtuple("CacheInfo", ["hits", "misses", "size", "max_size"])
):
  """Info for LRUCache."""

  @property
  def usage(self):
    return self.size / self.max_size if self._max_size else 0

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
  Now it's thread-safe.
  """

  def __init__(self, max_size):
    self._max_size = max_size
    self._data = collections.OrderedDict()
    self._hits = 0
    self._misses = 0
    self._lock = threading.Lock()

  def clear(self):
    with self._lock:
      self._data.clear()
      self._hits = 0
      self._misses = 0

  def make(self, key, fn):
    """Return the value, either from cache, or make it and save it."""
    with self._lock:
      try:
        val = self._data.pop(key)
        self._hits += 1
      except KeyError:
        self._misses += 1
        # Release lock during factory call to avoid holding GIL during inference
        pass  # we'll handle below

    if "val" not in locals():
      # Cache miss — compute outside lock
      val = fn()
      with self._lock:
        if len(self._data) >= self._max_size:
          self._data.popitem(False)
        self._data[key] = val
      return val

    with self._lock:
      self._data[key] = val
    return val

  def get(self, key):
    """Get the value and move it to the back, or return None on a miss."""
    with self._lock:
      try:
        val = self._data.pop(key)
        self._data[key] = val
        self._hits += 1
        return val
      except KeyError:
        self._misses += 1
        return None

  def set(self, key, val):
    """Set the value."""
    with self._lock:
      self._data.pop(key, None)
      self._data[key] = val
      if len(self._data) > self._max_size:
        self._data.popitem(False)
      return val

  def info(self):
    with self._lock:
      return CacheInfo(
        self._hits, self._misses, len(self._data), self._max_size
      )

  def __len__(self):
    with self._lock:
      return len(self._data)
