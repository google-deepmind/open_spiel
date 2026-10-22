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


"""Tests for who plays a seat and how a controller fails."""

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import seats
import pyspiel


class _Fixed:
  """A controller that always chooses the same action."""

  def __init__(self, action: object) -> None:
    self._action = action

  def choose_action(self, state: pyspiel.State, seat: int) -> object:
    """Returns the fixed action."""
    del state, seat
    return self._action


class _Mutating:
  """A controller that plays on the state it is given."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Plays the top-left corner on `state`, then chooses the centre."""
    del seat
    state.apply_action(0)
    return 4


def _seat(controller: object) -> seats.Seat:
  """Returns a seat played by `controller`."""
  return seats.Seat(
      spec=controllers.Spec(controllers.RANDOM),
      controller=controller,  # type: ignore[arg-type]
      autoplay=False,
  )


class SeedTest(absltest.TestCase):

  def test_derived_seeds_are_reproducible_and_distinct(self) -> None:
    self.assertEqual(
        seats.derive_seed(7, "seat", 0), seats.derive_seed(7, "seat", 0)
    )
    self.assertNotEqual(
        seats.derive_seed(7, "seat", 0), seats.derive_seed(7, "seat", 1)
    )


class MakeTest(absltest.TestCase):

  def test_a_human_seat_is_played_by_the_human_controller(self) -> None:
    seat = seats.Seat.make(
        controllers.HUMAN_SPEC,
        pyspiel.load_game("tic_tac_toe"),
        autoplay=False,
        seed=0,
    )
    self.assertIsInstance(seat.controller, controllers.Human)
    self.assertTrue(seat.is_human)
    self.assertTrue(seats.Seat.human().is_human)

  def test_a_random_seat_is_not_human(self) -> None:
    seat = seats.Seat.make(
        controllers.Spec(controllers.RANDOM),
        pyspiel.load_game("tic_tac_toe"),
        autoplay=True,
        seed=0,
    )
    self.assertFalse(seat.is_human)

  def test_a_human_seat_cannot_autoplay(self) -> None:
    with self.assertRaises(controllers.UnavailableError):
      seats.Seat.make(
          controllers.HUMAN_SPEC,
          pyspiel.load_game("tic_tac_toe"),
          autoplay=True,
          seed=0,
      )

  def test_an_unoffered_controller_is_refused(self) -> None:
    with self.assertRaises(controllers.UnavailableError):
      seats.Seat.make(
          controllers.Spec("no_such_controller"),
          pyspiel.load_game("tic_tac_toe"),
          autoplay=False,
          seed=0,
      )


class ChooseActionTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.state = pyspiel.load_game("tic_tac_toe").new_initial_state()

  def test_returns_a_legal_action(self) -> None:
    seat = _seat(_Fixed(4))
    self.assertEqual(seat.choose_action(self.state, 0), 4)
    self.assertIsNone(seat.failure)

  def test_an_illegal_action_is_a_failure(self) -> None:
    seat = _seat(_Fixed(9))
    self.assertIsNone(seat.choose_action(self.state, 0))
    self.assertEqual(
        seat.failure,
        contract.ControllerFailure(
            "IllegalAction",
            "Seat 0's controller chose 9, which is not one of its legal "
            "actions.",
        ),
    )

  def test_a_non_action_is_a_failure(self) -> None:
    for value in ("4", 4.0, True, None):
      seat = _seat(_Fixed(value))
      self.assertIsNone(seat.choose_action(self.state, 0), value)
      self.assertIsNotNone(seat.failure, value)

  def test_the_controller_cannot_change_the_live_state(self) -> None:
    seat = _seat(_Mutating())
    self.assertEqual(seat.choose_action(self.state, 0), 4)
    self.assertEqual(self.state.history(), [])


class SeatsTest(absltest.TestCase):

  def test_starts_human_and_reports_the_first_failure(self) -> None:
    every = seats.Seats(3)
    self.assertIsNone(every.failed)
    every[2].failure = contract.ControllerFailure("RuntimeError", "boom")
    every[1].failure = contract.ControllerFailure("RuntimeError", "boom")
    self.assertEqual(every.failed, 1)
    every.clear_failures()
    self.assertIsNone(every.failed)

  def test_views_list_every_seat_in_order(self) -> None:
    views = seats.Seats(2).views()
    self.assertEqual([view.seat for view in views], [0, 1])
    self.assertEqual({view.controller for view in views}, {contract.HUMAN})


if __name__ == "__main__":
  absltest.main()
