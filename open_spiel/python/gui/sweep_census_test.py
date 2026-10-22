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

"""Tests for judging the all-games sweep, on fake sweeps."""

import dataclasses

from absl.testing import absltest

from open_spiel.python.gui import fake_session
from open_spiel.python.gui import sweep
from open_spiel.python.gui import sweep_census
from open_spiel.python.gui import sweep_exceptions

_UNAVAILABLE = sweep_exceptions.FailureKind.UNAVAILABLE

_EXCEPTIONS = {
    "match": sweep_exceptions.ExpectedFailure(
        _UNAVAILABLE, "Not in this build."
    ),
    "tic_tac_toe": sweep_exceptions.ExpectedFailure(
        _UNAVAILABLE, "Not served."
    ),
}


def _fake_sweep(game: str, time_limit_seconds: float) -> sweep.Result:
  """Sweeps a fake tic-tac-toe session; each clock read is 10 s later."""
  readings = iter(range(0, 10_000, 10))
  result = sweep.sweep_game(
      "tic_tac_toe",
      lambda name: fake_session.FakeSession(name, seed=sweep.SEED),
      time_limit_seconds=time_limit_seconds,
      clock=lambda: float(next(readings)),
  )
  return dataclasses.replace(result, game=game)


def _passed(game: str) -> sweep.Result:
  """Returns a sweep of `game` that passed."""
  return _fake_sweep(game, time_limit_seconds=sweep.TIME_LIMIT_SECONDS)


def _too_slow(game: str) -> sweep.Result:
  """Returns a sweep of `game` that ran out of time."""
  return _fake_sweep(game, time_limit_seconds=25.0)


class SplitByRegistrationTest(absltest.TestCase):

  def test_listed_game_absent_from_the_registry_is_set_apart(self) -> None:
    present, absent = sweep_census.split_by_registration(
        _EXCEPTIONS, ["kuhn_poker", "tic_tac_toe"]
    )
    self.assertEqual(list(present), ["tic_tac_toe"])
    self.assertEqual(absent, ["match"])

  def test_every_listed_game_registered_sets_none_apart(self) -> None:
    present, absent = sweep_census.split_by_registration(
        _EXCEPTIONS, ["match", "tic_tac_toe"]
    )
    self.assertEqual(present, _EXCEPTIONS)
    self.assertEqual(absent, [])


class CensusTest(absltest.TestCase):

  def test_unlisted_game_that_passes_agrees(self) -> None:
    census = sweep_census.Census(_EXCEPTIONS, ["kuhn_poker", "match"])
    result = _passed("kuhn_poker")
    self.assertEqual(result.outcome, sweep.Outcome.PASSED, str(result))
    self.assertIsNone(census.record(result))
    self.assertTrue(census.passed)
    self.assertEqual(
        census.summary()[-1], "PASSED: 1 swept, 0 problems, 0 not judged"
    )

  def test_listed_game_that_passes_is_a_problem(self) -> None:
    census = sweep_census.Census(_EXCEPTIONS, ["tic_tac_toe"])
    problem = census.record(_passed("tic_tac_toe"))
    self.assertIsNotNone(problem)
    self.assertIn("listed as unavailable", str(problem))
    self.assertEqual(census.problems, [problem])
    self.assertFalse(census.passed)

  def test_game_listed_only_in_another_build_is_not_judged_by_its_entry(
      self,
  ) -> None:
    census = sweep_census.Census(_EXCEPTIONS, ["kuhn_poker", "tic_tac_toe"])
    self.assertEqual(census.unregistered, ["match"])
    self.assertIsNone(census.record(_passed("match")))
    self.assertIn(
        "Listed but not registered in this build: match", census.summary()
    )

  def test_game_out_of_time_is_not_judged_and_fails_the_run(self) -> None:
    census = sweep_census.Census({}, ["kuhn_poker", "pig"])
    result = _too_slow("pig")
    self.assertFalse(result.judged, str(result))
    problem = census.record(result)
    self.assertRegex(str(problem), "30.0 s elapsed, over the 25.0 s limit")
    self.assertEqual(census.not_judged, ["pig"])
    self.assertEqual(census.problems, [])
    self.assertFalse(census.passed)
    self.assertEqual(
        census.summary()[1:],
        [
            "Not judged, out of time: pig",
            "FAILED: 1 swept, 0 problems, 1 not judged",
        ],
    )

  def test_catalogue_that_agrees_with_the_list_is_no_problem(self) -> None:
    census = sweep_census.Census(_EXCEPTIONS, ["tic_tac_toe"])
    self.assertIsNone(census.check_catalogue(["tic_tac_toe"]))
    self.assertTrue(census.passed)

  def test_catalogue_that_disagrees_with_the_list_names_both_sides(
      self,
  ) -> None:
    census = sweep_census.Census(_EXCEPTIONS, ["match", "tic_tac_toe"])
    problem = census.check_catalogue(["crossword", "tic_tac_toe"])
    self.assertEqual(
        problem,
        "The games listed as unavailable are not those the catalogue marks "
        "not loadable: listed only: ['match']; catalogue only: ['crossword']",
    )
    self.assertFalse(census.passed)

  def test_empty_run_names_none(self) -> None:
    census = sweep_census.Census({}, [])
    self.assertEqual(
        census.summary(),
        [
            "Listed but not registered in this build: none",
            "Not judged, out of time: none",
            "PASSED: 0 swept, 0 problems, 0 not judged",
        ],
    )


if __name__ == "__main__":
  absltest.main()
