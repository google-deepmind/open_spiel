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


"""Tests for the MCTS controller's searches: C++ and Python."""

import threading
import time
from unittest import mock

from absl.testing import absltest

import pyspiel

# pylint: disable-next=unused-import
from open_spiel.python.games import iterated_prisoners_dilemma
from open_spiel.python.games import tic_tac_toe  # pylint: disable=unused-import
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import mcts_controller
from open_spiel.python.gui import mcts_search
from open_spiel.python.gui import move_budget

# The simulations MCTS is capped at when the C++ search gives it no clock.
_CAP = mcts_controller.SIMULATIONS[contract.MctsLimits.SIMULATIONS_ONLY].maximum

# pylint: disable-next=protected-access
_NodeBudgetReachedError = mcts_search._NodeBudgetReachedError

_MCTSBot = pyspiel.MCTSBot


def _search(game: pyspiel.Game, **kwargs: object) -> int:
  """Searches `game`'s initial position with the Python search."""
  settings: dict[str, object] = {
      "seed": 0,
      "max_simulations": _CAP,
      "uct_c": 2.0,
      "solve": False,
  }
  settings.update(kwargs)
  return mcts_search.python_search(
      game, game.new_initial_state(), **settings  # type: ignore[arg-type]
  )


class RolloutsTest(absltest.TestCase):

  def test_prior_counts_the_nodes_it_adds(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    rollouts = mcts_search.Rollouts(2, seed=0, node_budget=100)
    self.assertEqual(rollouts.nodes, 1)
    self.assertEqual(len(rollouts.prior(state)), 9)
    self.assertEqual(rollouts.nodes, 10)
    state.apply_action(4)
    self.assertEqual(len(rollouts.prior(state)), 8)
    self.assertEqual(rollouts.nodes, 18)

  def test_prior_refuses_to_grow_the_tree_past_its_budget(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    rollouts = mcts_search.Rollouts(2, seed=0, node_budget=17)
    rollouts.prior(state)
    state.apply_action(4)
    with self.assertRaises(_NodeBudgetReachedError):
      rollouts.prior(state)
    self.assertEqual(rollouts.nodes, 10)

  def test_the_root_is_expanded_whatever_the_budget(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    rollouts = mcts_search.Rollouts(2, seed=0, node_budget=1)
    self.assertEqual(len(rollouts.prior(state)), 9)

  def test_the_budget_is_positive(self) -> None:
    with self.assertRaises(ValueError):
      mcts_search.Rollouts(2, seed=0, node_budget=0)


class SearchTest(absltest.TestCase):

  def test_a_full_tree_stops_the_search_with_a_legal_move(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    evaluate = mcts_search.Rollouts.evaluate
    calls = []

    def counted(self: mcts_search.Rollouts, state: pyspiel.State) -> object:
      calls.append(state)
      return evaluate(self, state)

    with mock.patch.object(mcts_search.Rollouts, "evaluate", counted):
      action = _search(game, node_budget=200)
    self.assertIn(action, range(9))
    # Every simulation but the last adds at most nine nodes, so a tree of 200
    # nodes is full long before the simulation cap.
    self.assertGreaterEqual(len(calls), 20)
    self.assertLessEqual(len(calls), 200)

  def test_a_full_tree_plays_the_same_move_for_the_same_seed(self) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    moves = {_search(game, seed=7, node_budget=300) for _ in range(3)}
    self.assertEqual(len(moves), 1)

  def test_the_default_budget_bounds_a_search_of_many_actions(self) -> None:
    # A node is expanded on its second visit, so in 19x19 Go, 362 actions a
    # node, the tree grows past the root only after 362 simulations: more
    # than the controller's cap, which `python_search` does not apply. 2,000
    # simulations could grow a tree of over 500,000 nodes; a smaller default
    # budget stands in for the real one, which takes seconds to fill.
    game = pyspiel.load_game("go", {"board_size": 19})
    prior = mcts_search.Rollouts.prior
    trees = []

    def watched(
        self: mcts_search.Rollouts, state: pyspiel.State
    ) -> list[tuple[int, float]]:
      outcomes = prior(self, state)
      trees.append(self.nodes)
      return outcomes

    with (
        mock.patch.object(mcts_search, "NODE_BUDGET", 5_000),
        mock.patch.object(mcts_search.Rollouts, "prior", watched),
    ):
      action = _search(game, max_simulations=2_000)
    self.assertIn(action, game.new_initial_state().legal_actions())
    self.assertLessEqual(max(trees), 5_000)
    self.assertGreater(max(trees), 5_000 - 362)


class ImplementedInPythonTest(absltest.TestCase):

  def test_a_game_implemented_in_cpp_is_not(self) -> None:
    game = pyspiel.load_game("chess")
    self.assertFalse(mcts_search.implemented_in_python(game))

  def test_a_game_implemented_in_python_is(self) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    self.assertTrue(mcts_search.implemented_in_python(game))

  def test_a_cpp_wrapper_around_a_cpp_game_is_not(self) -> None:
    game = pyspiel.convert_to_turn_based(pyspiel.load_game("matrix_pd"))
    self.assertFalse(mcts_search.implemented_in_python(game))

  def test_a_cpp_wrapper_around_a_python_game_is(self) -> None:
    game = pyspiel.convert_to_turn_based(
        pyspiel.load_game("python_iterated_prisoners_dilemma")
    )
    self.assertEqual(type(game).__module__.partition(".")[0], "pyspiel")
    self.assertTrue(mcts_search.implemented_in_python(game))


class _Root:
  """A search tree's root, as the fake bots' searches return it."""

  def __init__(self, explore_count: int) -> None:
    self.explore_count = explore_count


class _HonouringBot:
  """A fake `pyspiel.MCTSBot` that stops at whichever limit comes first."""

  def __init__(
      self,
      *,
      max_simulations: int,
      max_wall_clock_time: float,
      **kwargs: object,
  ) -> None:
    self.max_simulations = max_simulations
    self.max_wall_clock_time = max_wall_clock_time
    self.kwargs = kwargs

  def mcts_search(self, state: pyspiel.State) -> _Root:
    """Returns the root of a search that ran `max_simulations`."""
    del state
    return _Root(self.max_simulations)


class _IgnoringBot(_HonouringBot):
  """A fake `pyspiel.MCTSBot` whose wall-clock time replaces its count."""

  def mcts_search(self, state: pyspiel.State) -> _Root:
    """Returns the root of a search that ran until its clock ran out."""
    del state
    time.sleep(self.max_wall_clock_time)
    return _Root(4000)


class _SlowBot(_HonouringBot):
  """A fake `pyspiel.MCTSBot` that runs its count, but takes the full time."""

  def mcts_search(self, state: pyspiel.State) -> _Root:
    """Returns the root of a search that ran `max_simulations` slowly."""
    del state
    time.sleep(self.max_wall_clock_time)
    return _Root(self.max_simulations)


class ProbeLimitsTest(absltest.TestCase):

  def test_a_search_that_stops_at_its_count_has_both_limits(self) -> None:
    with mock.patch.object(pyspiel, "MCTSBot", _HonouringBot):
      probe = mcts_search.probe_limits()
    self.assertEqual(probe.limits, contract.MctsLimits.SIMULATIONS_OR_CLOCK)
    self.assertEqual(probe.simulations, mcts_search.PROBE_SIMULATIONS)
    self.assertLess(probe.seconds, mcts_search.PROBE_SECONDS)

  def test_a_search_whose_clock_replaces_its_count_has_only_the_count(
      self,
  ) -> None:
    with mock.patch.object(pyspiel, "MCTSBot", _IgnoringBot):
      probe = mcts_search.probe_limits()
    self.assertEqual(probe.limits, contract.MctsLimits.SIMULATIONS_ONLY)
    self.assertEqual(probe.simulations, 4000)
    self.assertGreaterEqual(probe.seconds, mcts_search.PROBE_SECONDS)

  def test_a_search_that_could_be_either_is_refused(self) -> None:
    with (
        mock.patch.object(pyspiel, "MCTSBot", _SlowBot),
        self.assertRaisesRegex(RuntimeError, "is unknown"),
    ):
      mcts_search.probe_limits()

  def test_the_probe_sets_both_limits_and_no_more(self) -> None:
    made = []

    def record(**kwargs: object) -> _HonouringBot:
      made.append(_HonouringBot(**kwargs))  # type: ignore[arg-type]
      return made[-1]

    with mock.patch.object(pyspiel, "MCTSBot", side_effect=record):
      mcts_search.probe_limits()
    self.assertEqual(len(made), 1)
    bot = made[0]
    self.assertEqual(bot.max_simulations, mcts_search.PROBE_SIMULATIONS)
    self.assertEqual(bot.max_wall_clock_time, mcts_search.PROBE_SECONDS)
    self.assertFalse(bot.kwargs["solve"])

  def test_the_served_pyspiel_is_probed_quickly_and_consistently(
      self,
  ) -> None:
    probe = mcts_search.probe_limits()
    if probe.limits == contract.MctsLimits.SIMULATIONS_OR_CLOCK:
      self.assertEqual(probe.simulations, mcts_search.PROBE_SIMULATIONS)
      self.assertLess(probe.seconds, 0.05)
    else:
      self.assertGreater(probe.simulations, mcts_search.PROBE_SIMULATIONS)
      self.assertLess(probe.seconds, 2 * mcts_search.PROBE_SECONDS)

  def test_limits_probes_once(self) -> None:
    mcts_search.limits.cache_clear()
    self.addCleanup(mcts_search.limits.cache_clear)
    with mock.patch.object(
        mcts_search, "probe_limits", wraps=mcts_search.probe_limits
    ) as probe:
      first = mcts_search.limits()
      self.assertIs(mcts_search.limits(), first)
    probe.assert_called_once()


class CppBotTest(absltest.TestCase):

  def test_it_is_pyspiel_mcts_with_the_memory_cap_and_no_clock(self) -> None:
    game = pyspiel.load_game("chess")
    with mock.patch.object(pyspiel, "MCTSBot", side_effect=_MCTSBot) as made:
      bot = mcts_search.cpp_bot(
          game, seed=0, max_simulations=10, uct_c=2.0, solve=False
      )
    self.assertIsInstance(bot, _MCTSBot)
    kwargs = made.call_args.kwargs
    self.assertEqual(kwargs["max_memory_mb"], 1000)
    self.assertEqual(kwargs["max_memory_mb"], mcts_search.MAX_MEMORY_MB)
    self.assertEqual(kwargs["max_simulations"], 10)
    self.assertNotIn("max_wall_clock_time", kwargs)

  def test_a_wall_clock_time_is_passed_on(self) -> None:
    game = pyspiel.load_game("chess")
    with mock.patch.object(pyspiel, "MCTSBot", side_effect=_MCTSBot) as made:
      mcts_search.cpp_bot(
          game,
          seed=0,
          max_simulations=10,
          uct_c=2.0,
          solve=False,
          max_wall_clock_time=9.0,
      )
    self.assertEqual(made.call_args.kwargs["max_wall_clock_time"], 9.0)
    self.assertEqual(made.call_args.kwargs["max_simulations"], 10)

  def test_a_wall_clock_time_must_be_positive(self) -> None:
    game = pyspiel.load_game("chess")
    with self.assertRaisesRegex(ValueError, "must be positive"):
      mcts_search.cpp_bot(
          game,
          seed=0,
          max_simulations=10,
          uct_c=2.0,
          solve=False,
          max_wall_clock_time=0.0,
      )

  def test_it_refuses_a_game_implemented_in_python(self) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    with self.assertRaisesRegex(ValueError, "implemented in Python"):
      mcts_search.cpp_bot(
          game, seed=0, max_simulations=10, uct_c=2.0, solve=False
      )


# A move budget no search in these tests comes near, on any machine: the
# capped searches are bounded by their simulation count, which the tests
# check, never by the clock, so a slow or loaded host cannot fail them.
_GENEROUS_BUDGET_SECONDS = 3600.0

# A move budget a search that never returns overruns at once.
_TINY_BUDGET_SECONDS = 0.05


class _CountedBot:
  """Wraps a `pyspiel.MCTSBot`, recording how many simulations a move ran.

  `step` is the C++ bot's own: its search, then the root's best child.
  """

  def __init__(self, bot: pyspiel.MCTSBot) -> None:
    self._bot = bot
    self.simulations: list[int] = []

  def step(self, state: pyspiel.State) -> int:
    """Returns the bot's move, recording its search's root visit count."""
    root = self._bot.mcts_search(state)
    self.simulations.append(int(root.explore_count))
    return int(root.best_child().action)


class _BlockedBot:
  """A fake `pyspiel.MCTSBot` whose move waits until the test releases it."""

  def __init__(self, release: threading.Event) -> None:
    self._release = release

  def step(self, state: pyspiel.State) -> int:
    """Waits for `release`, then returns the first legal action."""
    self._release.wait()
    return int(state.legal_actions()[0])


class ControllerTest(absltest.TestCase):

  def set_move_budget(self, seconds: float) -> None:
    """Sets the move budget for the rest of the test."""
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    controllers.set_move_budget(seconds)

  def test_chess_is_searched_by_pyspiel_mcts_to_its_cap(self) -> None:
    self.assert_searched_by_pyspiel_mcts_to_its_cap("chess")

  def test_hive_is_searched_by_pyspiel_mcts_to_its_cap(self) -> None:
    # Hive is the slowest catalogued C++ game MCTS plays, the one the cap is
    # chosen for; see mcts_controller.SIMULATIONS. Whether the capped search
    # fits the 10 second budget depends on the machine and its load, so no
    # test asserts it.
    self.assert_searched_by_pyspiel_mcts_to_its_cap("hive")

  def assert_searched_by_pyspiel_mcts_to_its_cap(self, name: str) -> None:
    """Asserts a capped move of `name` is pyspiel's and stops at the cap.

    The search is given no wall-clock time, so the simulation cap alone
    bounds it; the move budget is set far beyond any machine's search time,
    so the assertions never depend on the machine's speed.

    Args:
      name: The game to search.
    """
    self.assertEqual(
        controllers.move_budget(), controllers.DEFAULT_MOVE_BUDGET_SECONDS
    )
    # The limits `_CAP` and the absent wall-clock time belong to.
    self.assertEqual(
        mcts_controller.configured_limits(),
        contract.MctsLimits.SIMULATIONS_ONLY,
    )
    self.set_move_budget(_GENEROUS_BUDGET_SECONDS)
    game = pyspiel.load_game(name)
    spec = controllers.Spec(
        controllers.MCTS,
        {"max_simulations": _CAP, "solve": False},
    )
    bot = controllers.create(spec, game, seed=0)
    bots = []
    counted = []

    def made(*args: object, **kwargs: object) -> object:
      bots.append(_MCTSBot(*args, **kwargs))
      counted.append(_CountedBot(bots[-1]))
      return counted[-1]

    state = game.new_initial_state()
    with (
        mock.patch.object(pyspiel, "MCTSBot", side_effect=made) as calls,
        mock.patch.object(
            mcts_search, "python_search", side_effect=AssertionError
        ),
    ):
      action = controllers.choose(bot, state, state.current_player())
    self.assertIn(action, state.legal_actions())
    self.assertEqual(len(bots), 1)
    self.assertIs(type(bots[0]), _MCTSBot)
    kwargs = calls.call_args.kwargs
    self.assertEqual(kwargs["max_simulations"], _CAP)
    self.assertEqual(kwargs["max_memory_mb"], 1000)
    self.assertNotIn("max_wall_clock_time", kwargs)
    self.assertEqual(counted[0].simulations, [_CAP])

  def test_the_counted_step_plays_the_bots_own_step(self) -> None:
    # `_CountedBot.step` stands in for `pyspiel.MCTSBot.step` above, so the
    # two must choose the same move from the same seed.
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()

    def bot() -> pyspiel.MCTSBot:
      return mcts_search.cpp_bot(
          game, seed=0, max_simulations=_CAP, uct_c=2.0, solve=False
      )

    real = bot()
    counted = _CountedBot(bot())
    self.assertEqual(counted.step(state), real.step(state))
    self.assertEqual(counted.simulations, [_CAP])

  def test_a_hive_search_over_the_budget_fails_its_seat(self) -> None:
    # The search is a fake that waits until released, so it overruns any
    # budget, however fast the machine.
    self.set_move_budget(_TINY_BUDGET_SECONDS)
    release = threading.Event()
    self.addCleanup(release.set)
    game = pyspiel.load_game("hive")
    spec = controllers.Spec(
        controllers.MCTS,
        {"max_simulations": _CAP, "solve": False},
    )
    bot = controllers.create(spec, game, seed=0)
    state = game.new_initial_state()
    seat = state.current_player()
    with (
        mock.patch.object(
            pyspiel, "MCTSBot", side_effect=lambda **_: _BlockedBot(release)
        ) as calls,
        self.assertRaises(controllers.ControllerError) as raised,
    ):
      controllers.choose(bot, state, seat)
    self.assertEqual(raised.exception.seat, seat)
    self.assertEqual(raised.exception.error, controllers.TIMEOUT)
    cause = raised.exception.__cause__
    assert isinstance(cause, move_budget.OverBudgetError), cause
    self.assertEqual(cause.seconds, _TINY_BUDGET_SECONDS)
    self.assertEqual(calls.call_args.kwargs["max_simulations"], _CAP)

  def test_it_plays_a_python_game_to_its_end_on_the_controller_thread(
      self,
  ) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    spec = controllers.Spec(controllers.MCTS, {"max_simulations": 200})
    bots = [controllers.create(spec, game, seed=seat) for seat in range(2)]
    threads = []
    search = mcts_search.python_search

    def recorded(*args: object, **kwargs: object) -> int:
      threads.append(threading.current_thread())
      return search(*args, **kwargs)  # type: ignore[arg-type]

    state = game.new_initial_state()
    with (
        mock.patch.object(mcts_search, "python_search", side_effect=recorded),
        mock.patch.object(pyspiel, "MCTSBot", side_effect=AssertionError),
    ):
      while not state.is_terminal():
        seat = state.current_player()
        state.apply_action(controllers.choose(bots[seat], state, seat))
    self.assertTrue(state.is_terminal())
    self.assertGreaterEqual(len(threads), 5)
    for thread in threads:
      self.assertIsNot(thread, threading.main_thread())
      self.assertRegex(thread.name, r"^controller-seat-[01]$")


if __name__ == "__main__":
  absltest.main()
