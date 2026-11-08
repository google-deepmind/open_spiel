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

"""Tests for a session's controller builds: preparing, failing, superseded."""

import time

from absl.testing import absltest

from open_spiel.python.gui import builds
from open_spiel.python.gui import builds_testing
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import session as session_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_SEAT_1 = contract.Viewer(contract.ViewerKind.SEAT, 1)
_HUMAN = controllers.HUMAN_SPEC
_PREPARING = contract.SeatStatus.PREPARING
_READY = contract.SeatStatus.READY
_UNAVAILABLE = contract.SeatStatus.UNAVAILABLE


class SessionBuildTest(builds_testing.BuildsTestCase):
  """A session builds controllers off its requests' threads."""

  def create(
      self,
      players: tuple[controllers.Spec, ...],
      *,
      game: str = "tic_tac_toe",
      autoplay: tuple[bool, ...] = (),
  ) -> session_lib.Session:
    """Starts a session of `game`."""
    return session_lib.Session.create(
        game,
        {},
        seed=1,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=players,
        autoplay=autoplay,
    )

  def seat(
      self,
      session: session_lib.Session,
      index: int,
      viewer: contract.Viewer = _ORACLE,
  ) -> contract.SeatView:
    """Returns seat `index` as `viewer` sees it."""
    return session.frame(viewer).seats[index]

  def test_a_fast_build_is_used_in_its_request(self) -> None:
    self.register("fast", builds_testing.First)
    session = self.create(
        (controllers.Spec("fast"), _HUMAN), autoplay=(True, False)
    )
    frame = session.frame(_ORACLE)
    self.assertEqual(frame.seats[0].status, _READY)
    self.assertIsNone(frame.seats[0].message)
    self.assertEqual(frame.num_plies, 1)

  def test_a_slow_build_prepares_then_plays_while_requests_answer(
      self,
  ) -> None:
    built = self.register_slow("slow")
    start = time.monotonic()
    session = self.create(
        (controllers.Spec("slow"), _HUMAN), autoplay=(True, False)
    )
    self.assertLess(time.monotonic() - start, builds_testing.SLEEP_SECONDS)
    seat = self.seat(session, 0)
    self.assertEqual(seat.status, _PREPARING)
    self.assertEqual(seat.message, "Building Slow, for at most 30 seconds.")
    self.assertIsNone(seat.failure)
    frame = session.frame(_SPECTATOR)
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)
    self.assertEqual(frame.num_plies, 0)
    # An advance answers at once, and plays nothing for a preparing seat.
    self.assertEqual(session.advance(_ORACLE).num_plies, 0)
    builds_testing.wait_for(
        lambda: session.frame(_ORACLE).num_plies == 1, "autoplay"
    )
    self.assertEqual(self.seat(session, 0).status, _READY)
    self.assertIsNone(self.seat(session, 0).message)
    self.assertEqual(len(built), 1)
    self.assertEqual(built[0].moves, 1)

  def test_a_seat_change_returns_while_its_controller_builds(self) -> None:
    gate, built = self.register_gated("gated")
    session = self.create((_HUMAN, _HUMAN))
    start = time.monotonic()
    frame = session.set_controller(
        _ORACLE, 0, controllers.Spec("gated"), autoplay=True
    )
    self.assertLess(
        time.monotonic() - start, builds_testing.PATIENCE_SECONDS / 4
    )
    self.assertEqual(frame.seats[0].status, _PREPARING)
    self.assertEqual(frame.seats[0].controller, "gated")
    gate.set()
    builds_testing.wait_for(
        lambda: session.frame(_ORACLE).num_plies == 1, "autoplay"
    )
    self.assertEqual(built[0].moves, 1)

  def test_a_build_that_cannot_play_leaves_its_seat_unavailable(
      self,
  ) -> None:
    def refuse() -> controllers.Controller:
      time.sleep(builds_testing.SLEEP_SECONDS)
      raise controllers.UnavailableError("The checkpoint does not load.")

    self.register("broken", refuse)
    # Kuhn poker hides information, so a seat sees no failure message.
    session = self.create(
        (_HUMAN, controllers.Spec("broken")),
        game="kuhn_poker",
        autoplay=(False, True),
    )
    self.assertEqual(self.seat(session, 1).status, _PREPARING)
    builds_testing.wait_for(
        lambda: self.seat(session, 1).status == _UNAVAILABLE, "the failure"
    )
    oracle = session.frame(_ORACLE)
    self.assertEqual(oracle.status, contract.SessionStatus.PAUSED)
    self.assertEqual(
        oracle.seats[1].failure,
        contract.ControllerFailure(
            "UnavailableError", "The checkpoint does not load."
        ),
    )
    seated = self.seat(session, 1, _SEAT_1)
    self.assertEqual(seated.message, "The checkpoint does not load.")
    self.assertEqual(
        seated.failure, contract.ControllerFailure("UnavailableError", None)
    )
    # Nothing resumes a seat with no controller but a new one.
    session.reset(_ORACLE)
    self.assertEqual(self.seat(session, 1).status, _UNAVAILABLE)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_ORACLE)
    frame = session.set_controller(_ORACLE, 1, controllers.Spec("random"))
    self.assertEqual(frame.seats[1].status, _READY)
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)

  def test_a_build_that_raises_says_only_what_it_raised(self) -> None:
    def crash() -> controllers.Controller:
      time.sleep(builds_testing.SLEEP_SECONDS)
      raise KeyError("seed 12345")

    self.register("crashes", crash)
    session = self.create(
        (_HUMAN, controllers.Spec("crashes")), game="kuhn_poker"
    )
    builds_testing.wait_for(
        lambda: self.seat(session, 1).status == _UNAVAILABLE, "the failure"
    )
    seated = self.seat(session, 1, _SEAT_1)
    self.assertEqual(
        seated.message, "Crashes could not be built: it raised KeyError."
    )
    self.assertEqual(
        seated.failure, contract.ControllerFailure("KeyError", None)
    )
    self.assertEqual(
        self.seat(session, 1).failure,
        contract.ControllerFailure("KeyError", "'seed 12345'"),
    )

  def test_a_build_over_budget_leaves_its_seat_unavailable(self) -> None:
    builds.set_budget_seconds(0.2)
    gate, built = self.register_gated("gated")
    session = self.create(
        (controllers.Spec("gated"), _HUMAN), autoplay=(True, False)
    )
    self.assertEqual(self.seat(session, 0).status, _PREPARING)
    builds_testing.wait_for(
        lambda: self.seat(session, 0).status == _UNAVAILABLE, "the timeout"
    )
    seat = self.seat(session, 0)
    self.assertEqual(
        seat.message, "Gated was not ready within the 0.2 second build budget."
    )
    assert seat.failure is not None
    self.assertEqual(seat.failure.error, controllers.TIMEOUT)
    # The build that finishes late is dropped.
    gate.set()
    builds_testing.wait_for(lambda: bool(built), "the late build")
    time.sleep(0.1)
    frame = session.frame(_ORACLE)
    self.assertEqual(frame.seats[0].status, _UNAVAILABLE)
    self.assertEqual(frame.num_plies, 0)

  def test_a_change_during_a_build_drops_the_stale_build(self) -> None:
    gate, built = self.register_gated("gated")
    self.register("fast", builds_testing.First)
    session = self.create(
        (controllers.Spec("gated"), _HUMAN), autoplay=(True, False)
    )
    self.assertEqual(self.seat(session, 0).status, _PREPARING)
    frame = session.set_controller(
        _ORACLE, 0, controllers.Spec("fast"), autoplay=False
    )
    self.assertEqual(frame.seats[0].controller, "fast")
    self.assertEqual(frame.seats[0].status, _READY)
    self.assertEqual(frame.num_plies, 0)
    gate.set()
    builds_testing.wait_for(lambda: bool(built), "the stale build")
    time.sleep(0.1)
    frame = session.frame(_ORACLE)
    self.assertEqual(frame.seats[0].controller, "fast")
    self.assertEqual(frame.seats[0].status, _READY)
    self.assertFalse(frame.seats[0].autoplay)
    self.assertEqual(frame.num_plies, 0)
    self.assertEqual(built[0].moves, 0)

  def test_a_change_during_a_build_to_another_slow_build(self) -> None:
    first_gate, first = self.register_gated("first")
    second_gate, second = self.register_gated("second")
    session = self.create(
        (controllers.Spec("first"), _HUMAN), autoplay=(True, False)
    )
    session.set_controller(
        _ORACLE, 0, controllers.Spec("second"), autoplay=True
    )
    first_gate.set()
    builds_testing.wait_for(lambda: bool(first), "the stale build")
    time.sleep(0.1)
    self.assertEqual(self.seat(session, 0).status, _PREPARING)
    self.assertEqual(self.seat(session, 0).controller, "second")
    second_gate.set()
    builds_testing.wait_for(
        lambda: session.frame(_ORACLE).num_plies == 1, "autoplay"
    )
    self.assertEqual((first[0].moves, second[0].moves), (0, 1))

  def test_a_fast_refusal_refuses_its_request_as_before(self) -> None:
    def refuse() -> controllers.Controller:
      raise controllers.UnavailableError("No such checkpoint.")

    self.register("refuses", refuse)
    builds.set_wait_seconds(builds_testing.PATIENCE_SECONDS)
    session = self.create((_HUMAN, _HUMAN))
    with self.assertRaisesRegex(
        errors.ControllerUnavailableError, "No such checkpoint"
    ):
      session.set_controller(_ORACLE, 1, controllers.Spec("refuses"))
    seat = self.seat(session, 1)
    self.assertEqual((seat.controller, seat.status), ("human", _READY))
    with self.assertRaisesRegex(
        errors.ControllerUnavailableError, "No such checkpoint"
    ):
      self.create((_HUMAN, controllers.Spec("refuses")))

  def test_invalid_settings_are_refused_before_any_build(self) -> None:
    built = self.register_slow("slow")
    session = self.create((_HUMAN, _HUMAN))
    with self.assertRaises(errors.ControllerUnavailableError):
      session.set_controller(
          _ORACLE, 1, controllers.Spec("slow", {"unknown": 1})
      )
    with self.assertRaises(errors.ControllerUnavailableError):
      session.set_controller(_ORACLE, 1, _HUMAN, autoplay=True)
    time.sleep(builds_testing.SLEEP_SECONDS * 2)
    self.assertEqual(built, [])


if __name__ == "__main__":
  absltest.main()
