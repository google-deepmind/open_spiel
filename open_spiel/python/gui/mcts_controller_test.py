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

"""Tests for the MCTS controller's limits."""

from collections.abc import Callable
from typing import cast
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import pyspiel

from open_spiel.python.games import tic_tac_toe  # pylint: disable=unused-import
from open_spiel.python.gui import cli
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health_report
from open_spiel.python.gui import mcts_controller
from open_spiel.python.gui import mcts_search

_OR_CLOCK = contract.MctsLimits.SIMULATIONS_OR_CLOCK
_ONLY = contract.MctsLimits.SIMULATIONS_ONLY

_MCTSBot = pyspiel.MCTSBot

# absl's decorator is not annotated; this gives it a type for strict mypy.
named_parameters = cast(
    Callable[..., Callable[[Callable[..., None]], Callable[..., None]]],
    parameterized.named_parameters,
)


class _Root:
  """A search tree's root, as the fake bots' searches return it."""

  def __init__(self, explore_count: int) -> None:
    self.explore_count = explore_count


class _HonouringBot:
  """A fake `pyspiel.MCTSBot` that stops at whichever limit comes first."""

  def __init__(self, *, max_simulations: int, **kwargs: object) -> None:
    del kwargs
    self.max_simulations = max_simulations

  def mcts_search(self, state: pyspiel.State) -> _Root:
    """Returns the root of a search that ran `max_simulations`."""
    del state
    return _Root(self.max_simulations)


class _IgnoringBot(_HonouringBot):
  """A fake `pyspiel.MCTSBot` whose wall-clock time replaces its count.

  It reports the many simulations such a search runs in the probe's time,
  without taking it.
  """

  def mcts_search(self, state: pyspiel.State) -> _Root:
    """Returns the root of a search that ran until its clock ran out."""
    del state
    return _Root(4000)


def _max_simulations() -> dict[str, contract.Json]:
  """Returns the schema of MCTS's `max_simulations`, as clients see it."""
  (mcts,) = [c for c in controllers.registered() if c.name == controllers.MCTS]
  properties = mcts.config_schema["properties"]
  assert isinstance(properties, dict)
  schema = properties["max_simulations"]
  assert isinstance(schema, dict)
  return schema


class LimitsTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.configure_mcts, controllers.mcts_limits())
    mcts_search.limits.cache_clear()
    self.addCleanup(mcts_search.limits.cache_clear)

  def test_mcts_starts_with_the_limits_that_fit_either_search(self) -> None:
    self.assertEqual(controllers.mcts_limits(), _ONLY)
    schema = _max_simulations()
    self.assertEqual((schema["default"], schema["maximum"]), (250, 250))

  @named_parameters(
      ("honoured", _HonouringBot, _OR_CLOCK, 1000, 10_000),
      ("replaced", _IgnoringBot, _ONLY, 250, 250),
  )
  def test_the_startup_probe_sets_the_limits(
      self,
      bot: type[_HonouringBot],
      limits: contract.MctsLimits,
      default: int,
      maximum: int,
  ) -> None:
    with (
        mock.patch.object(pyspiel, "MCTSBot", bot),
        mock.patch.object(logging, "info") as info,
    ):
      cli.configure_mcts()
    self.assertEqual(controllers.mcts_limits(), limits)
    self.assertEqual(health_report.build().capabilities.cpp_mcts_limits, limits)
    schema = _max_simulations()
    self.assertEqual((schema["default"], schema["maximum"]), (default, maximum))
    info.assert_called_once()
    self.assertEqual(info.call_args.args[1], limits.value)
    game = pyspiel.load_game("tic_tac_toe")
    controllers.create(
        controllers.Spec(controllers.MCTS, {"max_simulations": maximum}),
        game,
        seed=0,
    )
    with self.assertRaisesRegex(
        controllers.UnavailableError, f"lies in \\[1, {maximum}\\]"
    ):
      controllers.create(
          controllers.Spec(controllers.MCTS, {"max_simulations": maximum + 1}),
          game,
          seed=0,
      )

  def test_mcts_keeps_its_place_among_the_controllers(self) -> None:
    before = [c.name for c in controllers.registered()]
    controllers.configure_mcts(_OR_CLOCK)
    self.assertEqual([c.name for c in controllers.registered()], before)

  @named_parameters(
      ("or_clock", _OR_CLOCK, 9.0),
      ("only", _ONLY, None),
  )
  def test_a_cpp_move_has_a_backstop_only_when_the_count_is_honoured(
      self, limits: contract.MctsLimits, clock: float | None
  ) -> None:
    controllers.configure_mcts(limits)
    game = pyspiel.load_game("tic_tac_toe")
    bot = controllers.create(
        controllers.Spec(controllers.MCTS, {"max_simulations": 50}),
        game,
        seed=0,
    )
    state = game.new_initial_state()
    self.assertEqual(
        controllers.move_budget(), controllers.DEFAULT_MOVE_BUDGET_SECONDS
    )
    with (
        mock.patch.object(pyspiel, "MCTSBot", side_effect=_MCTSBot) as made,
        mock.patch.object(
            mcts_search, "python_search", side_effect=AssertionError
        ),
    ):
      action = controllers.choose(bot, state, 0)
    self.assertIn(action, state.legal_actions())
    kwargs = made.call_args.kwargs
    self.assertEqual(kwargs["max_simulations"], 50)
    self.assertEqual(kwargs.get("max_wall_clock_time"), clock)

  def test_a_python_game_keeps_the_python_search(self) -> None:
    controllers.configure_mcts(_OR_CLOCK)
    game = pyspiel.load_game("python_tic_tac_toe")
    bot = controllers.create(
        controllers.Spec(controllers.MCTS, {"max_simulations": 50}),
        game,
        seed=0,
    )
    state = game.new_initial_state()
    search = mcts_search.python_search
    with (
        mock.patch.object(
            mcts_search, "python_search", side_effect=search
        ) as python,
        mock.patch.object(pyspiel, "MCTSBot", side_effect=AssertionError),
    ):
      action = controllers.choose(bot, state, 0)
    self.assertIn(action, state.legal_actions())
    python.assert_called_once()
    self.assertNotIn("max_wall_clock_time", python.call_args.kwargs)


class BackstopTest(absltest.TestCase):

  def test_it_ends_a_margin_inside_the_budget(self) -> None:
    self.assertEqual(mcts_controller.backstop_seconds(10.0), 9.0)
    self.assertEqual(mcts_controller.backstop_seconds(30.0), 29.0)

  def test_it_takes_half_a_short_budget(self) -> None:
    self.assertEqual(mcts_controller.backstop_seconds(1.0), 0.5)

  def test_the_budget_must_be_positive(self) -> None:
    with self.assertRaisesRegex(ValueError, "must be positive"):
      mcts_controller.backstop_seconds(0.0)


if __name__ == "__main__":
  absltest.main()
