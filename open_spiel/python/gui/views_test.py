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


"""Tests for building frames for each kind of viewer."""

import dataclasses

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import loader
from open_spiel.python.gui import views
import pyspiel

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_HUMAN_SEATS = (
    contract.SeatView(0, contract.HUMAN, {}, False, None),
    contract.SeatView(1, contract.HUMAN, {}, False, None),
)


class OmniscienceTest(absltest.TestCase):

  def test_the_oracle_is_always_omniscient(self) -> None:
    self.assertTrue(views.is_omniscient(loader.load("kuhn_poker", {}), _ORACLE))

  def test_others_are_omniscient_only_where_nothing_is_hidden(self) -> None:
    self.assertFalse(
        views.is_omniscient(loader.load("kuhn_poker", {}), _SPECTATOR)
    )
    self.assertTrue(
        views.is_omniscient(loader.load("tic_tac_toe", {}), _SPECTATOR)
    )


class StatusTest(absltest.TestCase):

  def test_a_terminal_game_is_terminal(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    for action in (0, 3, 1, 4, 2):
      state.apply_action(action)
    self.assertEqual(
        views.status(state, _HUMAN_SEATS), contract.SessionStatus.TERMINAL
    )

  def test_a_failed_seat_pauses_the_game(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    failed = (
        _HUMAN_SEATS[0],
        contract.SeatView(
            1, "random", {}, False, contract.ControllerFailure("E", "m")
        ),
    )
    self.assertEqual(
        views.status(state, _HUMAN_SEATS), contract.SessionStatus.PLAYING
    )
    self.assertEqual(views.status(state, failed), contract.SessionStatus.PAUSED)


def _position(loaded: loader.LoadedGame) -> views.Position:
  """Returns the initial position of a two-seat session of `loaded`."""
  return views.Position(
      info=contract.SessionInfo(
          id="s",
          game=loaded.name,
          parameters=dict(loaded.parameters),
          num_seats=loaded.num_seats,
          chance_policy=contract.ChancePolicy.AUTO,
          turn_based=loaded.turn_based,
          hides_information=loaded.hides_information,
          seed=1,
          controllers=controllers.available(loaded.game),
      ),
      loaded=loaded,
      status=contract.SessionStatus.PLAYING,
      seats=_HUMAN_SEATS,
      state=loaded.game.new_initial_state(),
      ply=0,
      history=(),
  )


class BuildTest(absltest.TestCase):

  def test_refuses_a_seat_the_game_does_not_have(self) -> None:
    position = _position(loader.load("tic_tac_toe", {}))
    with self.assertRaisesRegex(ValueError, "no seat 2"):
      views.build(position, contract.Viewer(contract.ViewerKind.SEAT, 2))

  def test_refuses_a_seat_of_a_game_without_seat_views(self) -> None:
    loaded = dataclasses.replace(
        loader.load("kuhn_poker", {}),
        provides_observation=False,
        provides_information_state=False,
    )
    with self.assertRaisesRegex(ValueError, "no seat views"):
      views.build(
          _position(loaded), contract.Viewer(contract.ViewerKind.SEAT, 0)
      )


if __name__ == "__main__":
  absltest.main()
