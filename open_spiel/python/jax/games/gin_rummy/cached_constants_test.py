# Copyright 2026 DeepMind Technologies Limited
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

"""Gin rummy's cached constant tables are numpy arrays.

So filling a cache inside a trace never stores a tracer.
"""

import inspect

from absl.testing import absltest
import jax
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds

# Imported for its side effect: the game module builds the caches.
from open_spiel.python.jax.games.gin_rummy import game  # pylint: disable=unused-import
from open_spiel.python.jax.games.gin_rummy import layoffs
from open_spiel.python.jax.games.gin_rummy import lookup_tables

_AFFECTED_MODULES = (
    cards_and_melds,
    lookup_tables,
    layoffs,
)


def _cached_builders(module):
  for name, value in sorted(module.__dict__.items()):
    if name.startswith("__"):
      continue
    if getattr(value, "__module__", None) != module.__name__:
      continue
    if hasattr(value, "cache_info") and hasattr(value, "cache_clear"):
      yield name, value


def _assert_no_jax_array(value):
  if isinstance(value, tuple):
    for item in value:
      _assert_no_jax_array(item)
    return
  if isinstance(value, np.ndarray):
    return
  raise AssertionError(
      f"cached builder returned {type(value)!r}, expected np.ndarray"
  )


class CachedConstantsTest(absltest.TestCase):
  """The cached builders return numpy and never cache a tracer."""

  def test_cached_builders_return_numpy_arrays(self):
    """Every cached builder returns numpy arrays."""
    for module in _AFFECTED_MODULES:
      for name, builder in _cached_builders(module):
        with self.subTest(builder=f"{module.__name__}.{name}"):
          builder.cache_clear()
          _assert_no_jax_array(builder())

  def test_cached_builders_are_safe_inside_jit_after_concrete_cache_fill(self):
    """Cached builders are safe in jit after a concrete fill."""
    for module in _AFFECTED_MODULES:
      for name, builder in _cached_builders(module):
        with self.subTest(builder=f"{module.__name__}.{name}"):
          builder.cache_clear()
          builder()
          jax.jit(builder)()

  def test_cached_builders_do_not_leak_tracers_when_jit_fills_cache(self):
    """A cache filled inside jit holds no tracer."""
    for module in _AFFECTED_MODULES:
      for name, builder in _cached_builders(module):
        with self.subTest(builder=f"{module.__name__}.{name}"):
          builder.cache_clear()
          jax.jit(builder)()
          _assert_no_jax_array(builder())

  def test_game_import_warmed_affected_builders(self):
    """Importing the game warms the affected builders."""
    self.assertIsNotNone(game)  # Imported for module side effects.
    for module in _AFFECTED_MODULES:
      for name, builder in _cached_builders(module):
        with self.subTest(builder=f"{module.__name__}.{name}"):
          signature = inspect.signature(builder)
          self.assertEmpty(signature.parameters)
          self.assertGreaterEqual(builder.cache_info().currsize, 1)


if __name__ == "__main__":
  absltest.main()
