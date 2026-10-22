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

"""Tests for the games the sweep is expected to fail on."""

import json
import os

from absl.testing import absltest

from open_spiel.python.gui import sweep
from open_spiel.python.gui import sweep_exceptions

_Expected = sweep_exceptions.ExpectedFailure
_Kind = sweep_exceptions.FailureKind
_Outcome = sweep.Outcome
_Listed = sweep_exceptions.ExpectedOutcome


class LoadExceptionsTest(absltest.TestCase):

  def _write(self, listed: object) -> str:
    """Writes `listed` as JSON to a fresh file and returns its path."""
    content = listed if isinstance(listed, str) else json.dumps(listed)
    return os.fspath(self.create_tempfile(content=content))

  def test_loads_each_games_expected_failure(self) -> None:
    path = self._write({
        "a_game": {"kind": "unavailable", "reason": "Why."},
        "b_game": {
            "kind": "backend",
            "reason": "Because.",
            "outcome": "unfinished",
        },
    })
    self.assertEqual(
        sweep_exceptions.load_exceptions(path),
        {
            "a_game": _Expected(_Kind.UNAVAILABLE, "Why."),
            "b_game": _Expected(_Kind.BACKEND, "Because.", _Listed.UNFINISHED),
        },
    )

  def test_refuses_games_out_of_order(self) -> None:
    entry = {"kind": "unavailable", "reason": "Why."}
    path = self._write({"b_game": entry, "a_game": entry})
    with self.assertRaisesRegex(ValueError, "in order"):
      sweep_exceptions.load_exceptions(path)

  def test_refuses_a_game_listed_twice(self) -> None:
    entry = '{"kind": "unavailable", "reason": "Why."}'
    path = self._write(f'{{"a_game": {entry}, "a_game": {entry}}}')
    with self.assertRaisesRegex(ValueError, "repeated"):
      sweep_exceptions.load_exceptions(path)

  def test_refuses_an_unknown_kind(self) -> None:
    path = self._write({"a_game": {"kind": "flaky", "reason": "Why."}})
    with self.assertRaisesRegex(ValueError, "a_game"):
      sweep_exceptions.load_exceptions(path)

  def test_refuses_a_game_without_a_reason(self) -> None:
    path = self._write({"a_game": {"kind": "unavailable", "reason": " "}})
    with self.assertRaisesRegex(ValueError, "a_game"):
      sweep_exceptions.load_exceptions(path)

  def test_refuses_a_backend_failure_without_a_known_outcome(self) -> None:
    for entry in [
        {"kind": "backend", "reason": "Why."},
        {"kind": "backend", "reason": "Why.", "outcome": "passed"},
        {"kind": "backend", "reason": "Why.", "outcome": "refused"},
    ]:
      with self.subTest(entry=entry):
        path = self._write({"a_game": entry})
        with self.assertRaisesRegex(ValueError, "a_game"):
          sweep_exceptions.load_exceptions(path)

  def test_refuses_an_outcome_on_any_other_kind(self) -> None:
    entry = {"kind": "unavailable", "reason": "Why.", "outcome": "error"}
    path = self._write({"a_game": entry})
    with self.assertRaisesRegex(ValueError, "a_game"):
      sweep_exceptions.load_exceptions(path)

  def test_refuses_anything_but_an_object(self) -> None:
    path = self._write(["a_game"])
    with self.assertRaisesRegex(ValueError, "not an object"):
      sweep_exceptions.load_exceptions(path)

  def test_checked_in_exceptions_load(self) -> None:
    self.assertTrue(sweep_exceptions.load_exceptions())


class DisagreementTest(absltest.TestCase):

  _EXCEPTIONS = {
      "unavailable": _Expected(_Kind.UNAVAILABLE, "No default."),
      "unfinished": _Expected(_Kind.BACKEND, "Too long.", _Listed.UNFINISHED),
      "blind": _Expected(_Kind.BACKEND, "No observation.", _Listed.ERROR),
      "browser": _Expected(_Kind.BROWSER, "Too wide."),
  }

  def _disagreement(self, game: str, outcome: sweep.Outcome) -> str | None:
    """Returns the disagreement of a result of `game` with `outcome`."""
    result = sweep.Result(game, outcome, "", 0)
    return sweep_exceptions.disagreement(result, self._EXCEPTIONS)

  def test_agreeing_results(self) -> None:
    for game, outcome in [
        ("unlisted", _Outcome.PASSED),
        ("browser", _Outcome.PASSED),
        ("unavailable", _Outcome.REFUSED),
        ("unfinished", _Outcome.PLY_CAP),
        ("blind", _Outcome.ERROR),
    ]:
      with self.subTest(game=game, outcome=outcome):
        self.assertIsNone(self._disagreement(game, outcome))

  def test_unlisted_game_that_fails_disagrees(self) -> None:
    self.assertEqual(
        self._disagreement("unlisted", _Outcome.ERROR),
        "unlisted: error after 0 plies. Fix it, or list it in the exceptions "
        "with the reason.",
    )

  def test_listed_game_that_passes_disagrees(self) -> None:
    self.assertEqual(
        self._disagreement("unfinished", _Outcome.PASSED),
        "unfinished: passed after 0 plies, but it is listed as backend "
        "(unfinished): Too long.",
    )

  def test_leak_in_a_game_listed_as_unfinished_disagrees(self) -> None:
    self.assertEqual(
        self._disagreement("unfinished", _Outcome.INVARIANT),
        "unfinished: invariant after 0 plies, but it is listed as backend "
        "(unfinished): Too long.",
    )

  def test_game_that_fails_another_way_disagrees(self) -> None:
    for game, outcome in [
        ("browser", _Outcome.INVARIANT),
        ("unavailable", _Outcome.ERROR),
        ("unfinished", _Outcome.REFUSED),
        ("unfinished", _Outcome.PAUSED),
        ("blind", _Outcome.INVARIANT),
        ("blind", _Outcome.PLY_CAP),
    ]:
      with self.subTest(game=game, outcome=outcome):
        self.assertIsNotNone(self._disagreement(game, outcome))

  def test_runner_too_slow_is_neither_agreement_nor_disagreement(
      self,
  ) -> None:
    for game in ("unlisted", "unavailable", "unfinished", "blind", "browser"):
      with self.subTest(game=game):
        with self.assertRaisesRegex(ValueError, "neither agrees"):
          self._disagreement(game, _Outcome.TOO_SLOW)


class ExpectedFailureTest(absltest.TestCase):

  def test_only_a_backend_failure_has_an_outcome(self) -> None:
    with self.assertRaisesRegex(ValueError, "outcome"):
      _Expected(_Kind.BACKEND, "Why.")
    with self.assertRaisesRegex(ValueError, "outcome"):
      _Expected(_Kind.UNAVAILABLE, "Why.", _Listed.ERROR)


if __name__ == "__main__":
  absltest.main()
