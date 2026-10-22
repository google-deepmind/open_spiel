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

"""Tests for the sweep of a game through the session API."""

from absl.testing import absltest

from open_spiel.python.gui import fake_session
from open_spiel.python.gui import sweep

_Fault = fake_session.Fault
_Outcome = sweep.Outcome


class _StepClock:
  """A clock that reads 0 first, then `step` seconds later at each read."""

  def __init__(self, step: float) -> None:
    self._step = step
    self._next = 0.0

  def __call__(self) -> float:
    now = self._next
    self._next += self._step
    return now


def _opener(fault: _Fault = _Fault.NONE) -> sweep.OpenSession:
  """Opens fake sessions seeded from the sweep's seed, with `fault`."""
  return lambda game: fake_session.FakeSession(
      game, seed=sweep.SEED, fault=fault
  )


class SweepGameTest(absltest.TestCase):

  def test_well_behaved_session_passes(self) -> None:
    for game_name in ("kuhn_poker", "leduc_poker", "tic_tac_toe"):
      with self.subTest(game_name):
        result = sweep.sweep_game(game_name, _opener())
        self.assertEqual(result.outcome, _Outcome.PASSED, str(result))
        self.assertGreater(result.plies, 0)
        self.assertEqual(result.detail, "")

  def test_session_that_cannot_be_opened_is_refused(self) -> None:
    def refuse(game_name: str) -> sweep.SweptSession:
      raise ValueError(f"{game_name} needs a parameter.")

    result = sweep.sweep_game("kuhn_poker", refuse)
    self.assertEqual(result.outcome, _Outcome.REFUSED)
    self.assertEqual(result.detail, "ValueError: kuhn_poker needs a parameter.")

  def test_game_the_sweep_cannot_replay_is_an_error(self) -> None:
    def open_any(game_name: str) -> sweep.SweptSession:
      del game_name  # The session opens; the sweep's own replay cannot.
      return fake_session.FakeSession("tic_tac_toe", seed=sweep.SEED)

    result = sweep.sweep_game("no_such_game", open_any)
    self.assertEqual(result.outcome, _Outcome.ERROR)
    self.assertRegex(result.detail, "^Replay: ")

  def test_frame_that_breaks_a_check_is_an_invariant_failure(self) -> None:
    result = sweep.sweep_game("kuhn_poker", _opener(_Fault.SWAPPED_OBSERVATION))
    self.assertEqual(result.outcome, _Outcome.INVARIANT)
    self.assertIn("observation", result.detail)

  def test_advance_that_applies_nothing_is_an_invariant_failure(self) -> None:
    result = sweep.sweep_game("tic_tac_toe", _opener(_Fault.STALL))
    self.assertEqual(result.outcome, _Outcome.INVARIANT)
    self.assertIn("applied no action", result.detail)
    self.assertEqual(result.plies, 0)

  def test_failed_controller_pauses_the_sweep(self) -> None:
    result = sweep.sweep_game("tic_tac_toe", _opener(_Fault.PAUSE))
    self.assertEqual(result.outcome, _Outcome.PAUSED)
    self.assertEqual(
        result.detail, "seat 0: RuntimeError: The controller stopped."
    )

  def test_controller_failure_that_pauses_nothing_is_an_invariant_failure(
      self,
  ) -> None:
    result = sweep.sweep_game(
        "tic_tac_toe", _opener(_Fault.RAISE_WITHOUT_PAUSE)
    )
    self.assertEqual(result.outcome, _Outcome.INVARIANT)
    self.assertIn("is not paused", result.detail)
    self.assertEqual(result.plies, 0)

  def test_replay_that_leaks_the_history_is_an_invariant_failure(self) -> None:
    result = sweep.sweep_game(
        "kuhn_poker", _opener(_Fault.REPLAY_LEAKS_HISTORY)
    )
    self.assertEqual(result.outcome, _Outcome.INVARIANT)
    self.assertIn("is shown to", result.detail)

  def test_session_that_raises_is_an_error(self) -> None:
    result = sweep.sweep_game("tic_tac_toe", _opener(_Fault.BROKEN_FRAME))
    self.assertEqual(result.outcome, _Outcome.ERROR)
    self.assertEqual(result.detail, "RuntimeError: The frame cannot be built.")

  def test_game_not_over_at_the_ply_cap_stops(self) -> None:
    result = sweep.sweep_game("tic_tac_toe", _opener(), ply_cap=3)
    self.assertEqual(result.outcome, _Outcome.PLY_CAP)
    self.assertEqual(result.plies, 3)

  def test_runner_too_slow_stops_without_a_verdict(self) -> None:
    clock = _StepClock(step=10.0)
    result = sweep.sweep_game(
        "tic_tac_toe", _opener(), time_limit_seconds=25.0, clock=clock
    )
    # The clock reads 0 when the sweep starts, then 10, 20 and 30 before
    # plies 0, 1 and 2; 30 s is the first reading over the 25 s limit.
    self.assertEqual(result.outcome, _Outcome.TOO_SLOW)
    self.assertEqual(result.plies, 2)
    self.assertFalse(result.judged)
    self.assertIn("30.0 s elapsed, over the 25.0 s limit", result.detail)
    self.assertIn("not a verdict on the game", result.detail)

  def test_every_other_outcome_is_a_verdict(self) -> None:
    for outcome in _Outcome:
      with self.subTest(outcome=outcome):
        result = sweep.Result("chess", outcome, "", 0)
        self.assertEqual(result.judged, outcome != _Outcome.TOO_SLOW)

  def test_ply_cap_wins_over_the_time_limit_at_the_same_ply(self) -> None:
    result = sweep.sweep_game(
        "tic_tac_toe",
        _opener(),
        ply_cap=2,
        time_limit_seconds=25.0,
        clock=_StepClock(step=10.0),
    )
    self.assertEqual(result.outcome, _Outcome.PLY_CAP)
    self.assertEqual(result.plies, 2)

  def test_verdict_does_not_depend_on_the_clock(self) -> None:
    for game_name, ply_cap in [
        ("kuhn_poker", sweep.PLY_CAP),
        ("leduc_poker", sweep.PLY_CAP),
        ("tic_tac_toe", sweep.PLY_CAP),
        ("tic_tac_toe", 4),
    ]:
      with self.subTest(game_name=game_name, ply_cap=ply_cap):
        results = [
            sweep.sweep_game(game_name, _opener(), ply_cap=ply_cap, clock=clock)
            for clock in (
                _StepClock(step=0.0),
                _StepClock(step=1.0),
                _StepClock(step=sweep.TIME_LIMIT_SECONDS / 100),
            )
        ]
        self.assertTrue(results[0].judged, str(results[0]))
        self.assertEqual(results, [results[0]] * len(results))

  def test_same_seed_gives_the_same_result(self) -> None:
    for game_name in ("kuhn_poker", "leduc_poker", "tic_tac_toe"):
      with self.subTest(game_name):
        first = sweep.sweep_game(game_name, _opener())
        second = sweep.sweep_game(game_name, _opener())
        self.assertEqual(first, second)

  def test_result_reads_as_a_sentence(self) -> None:
    result = sweep.Result("chess", _Outcome.ERROR, "ValueError: no.", 7)
    self.assertEqual(str(result), "chess: error after 7 plies: ValueError: no.")


if __name__ == "__main__":
  absltest.main()
