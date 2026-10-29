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

"""Helpers shared by the controller build tests.

builds_test.py and session_builds_test.py both use them. They live in this
module, not in a test, so that each test can depend on it.
"""

from collections.abc import Callable, Mapping
import threading
import time

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers

# Short, so the tests run fast; a request waits this long for a build.
_WAIT_SECONDS = 0.05

# The build budget of most tests: far longer than any build they make.
BUDGET_SECONDS = 30.0

# How long a slow build sleeps: far longer than a request waits for it.
SLEEP_SECONDS = 0.5

# How long a test waits for something that should happen at once.
PATIENCE_SECONDS = 20.0


class First:
  """Plays its seat's lowest legal action, and counts its moves."""

  def __init__(self) -> None:
    self.moves = 0

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the lowest of `seat`'s legal actions."""
    self.moves += 1
    return int(state.legal_actions(seat)[0])


def wait_for(check: Callable[[], bool], what: str) -> None:
  """Waits until `check` holds, failing loud after `PATIENCE_SECONDS`."""
  deadline = time.monotonic() + PATIENCE_SECONDS
  while not check():
    if time.monotonic() > deadline:
      raise AssertionError(f"Waited {PATIENCE_SECONDS:g} s for {what}.")
    time.sleep(0.01)


class BuildsTestCase(absltest.TestCase):
  """Sets a short build wait and a known budget for each test."""

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(builds.set_wait_seconds, builds.wait_seconds())
    self.addCleanup(builds.set_budget_seconds, builds.budget_seconds())
    builds.set_wait_seconds(_WAIT_SECONDS)
    builds.set_budget_seconds(BUDGET_SECONDS)

  def register(
      self, name: str, build: Callable[[], controllers.Controller]
  ) -> None:
    """Registers a controller `name` built by `build`, for this test."""

    def builder(
        game: pyspiel.Game,
        config: Mapping[str, contract.ParameterValue],
        seed: int,
    ) -> controllers.Controller:
      del game, config, seed
      return build()

    controllers.register(
        name, controllers.Kind(name.title(), (), False, builder)
    )
    self.addCleanup(controllers.unregister, name)

  def register_slow(
      self, name: str, *, seconds: float = SLEEP_SECONDS
  ) -> list[First]:
    """Registers `name`, built after sleeping; returns what it builds."""
    built: list[First] = []

    def build() -> controllers.Controller:
      time.sleep(seconds)
      built.append(First())
      return built[-1]

    self.register(name, build)
    return built

  def register_gated(self, name: str) -> tuple[threading.Event, list[First]]:
    """Registers `name`, built once the event is set; returns both."""
    gate = threading.Event()
    self.addCleanup(gate.set)
    built: list[First] = []

    def build() -> controllers.Controller:
      gate.wait()
      built.append(First())
      return built[-1]

    self.register(name, build)
    return gate, built
