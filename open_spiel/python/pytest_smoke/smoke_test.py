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

"""Smoke tests of pytest_test targets, run by both CMake (CTest) and Bazel.

They use a fixture of conftest.py and import OpenSpiel modules by their full
name, from a package directory (with an __init__.py), like downstream tests.
"""

import pyspiel
from open_spiel.python.utils import lru_cache


def test_fixture_from_conftest(tic_tac_toe: pyspiel.Game) -> None:
  state = tic_tac_toe.new_initial_state()
  assert state.legal_actions() == list(range(9))


def test_imports_open_spiel_python_modules() -> None:
  cache = lru_cache.LRUCache(max_size=1)
  assert cache.make("key", lambda: 1) == 1


def test_deselected() -> None:
  """Deselected by the BUILD target: fails if the deselection is lost."""
  raise AssertionError("The BUILD target must deselect this test.")
