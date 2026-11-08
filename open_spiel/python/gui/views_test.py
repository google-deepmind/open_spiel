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
import json

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


def _dealt(loaded: loader.LoadedGame) -> views.Position:
  """Returns a position of `loaded` after its opening chance outcomes."""
  state = loaded.game.new_initial_state()
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  return dataclasses.replace(_position(loaded), state=state)


class ObservationStructTest(absltest.TestCase):

  def test_a_seat_gets_its_own_observation_struct(self) -> None:
    position = _dealt(loader.load("gin_rummy", {}))
    for seat in (0, 1):
      built = views.build(
          position, contract.Viewer(contract.ViewerKind.SEAT, seat)
      )
      self.assertIsNone(built.struct)
      observed = built.observation_struct
      assert isinstance(observed, dict)
      self.assertEqual(observed["observing_player"], seat)
      hands = observed["hands"]
      self.assertNotIn("XX", hands[seat])
      self.assertEqual(set(hands[1 - seat]), {"XX"})

  def test_the_oracle_and_a_spectator_get_none(self) -> None:
    position = _dealt(loader.load("gin_rummy", {}))
    oracle = views.build(position, _ORACLE)
    self.assertIsNone(oracle.observation_struct)
    self.assertIsNotNone(oracle.struct)
    self.assertIsNone(views.build(position, _SPECTATOR).observation_struct)

  def test_a_seat_of_a_game_without_observation_structs_gets_none(
      self,
  ) -> None:
    position = _dealt(loader.load("kuhn_poker", {}))
    built = views.build(position, contract.Viewer(contract.ViewerKind.SEAT, 0))
    self.assertIsNone(built.observation_struct)

  def test_a_seat_of_a_game_that_hides_nothing_gets_both_structs(
      self,
  ) -> None:
    position = _position(loader.load("tic_tac_toe", {}))
    built = views.build(position, contract.Viewer(contract.ViewerKind.SEAT, 1))
    self.assertEqual(built.observation_struct, built.struct)


def _hearts(num_passes: int) -> views.Position:
  """Returns a hearts position passing left, `num_passes` cards passed."""
  loaded = loader.load("hearts", {})
  state = loaded.game.new_initial_state()
  state.apply_action(1)  # The pass direction: left.
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  for _ in range(num_passes):
    state.apply_action(state.legal_actions()[0])
  seats = tuple(
      contract.SeatView(seat, contract.HUMAN, {}, False, None)
      for seat in range(loaded.num_seats)
  )
  return dataclasses.replace(_position(loaded), seats=seats, state=state)


class HeartsPassingTest(absltest.TestCase):
  """The hearts observation struct is withheld from seats while passing."""

  def test_a_seat_gets_none_while_the_hand_is_passing(self) -> None:
    for num_passes in (0, 3, 11):
      position = _hearts(num_passes)
      for seat in range(4):
        built = views.build(
            position, contract.Viewer(contract.ViewerKind.SEAT, seat)
        )
        self.assertIsNone(built.observation_struct)
        self.assertIsNotNone(built.presentation)

  def test_a_seat_gets_its_struct_once_every_seat_has_passed(self) -> None:
    position = _hearts(12)
    for seat in range(4):
      built = views.build(
          position, contract.Viewer(contract.ViewerKind.SEAT, seat)
      )
      observed = built.observation_struct
      assert isinstance(observed, dict)
      self.assertEqual(observed["phase"], "Play")
      self.assertEqual(observed["observing_player"], seat)

  def test_the_oracle_is_unaffected(self) -> None:
    for num_passes in (0, 11, 12):
      position = _hearts(num_passes)
      oracle = views.build(position, _ORACLE)
      self.assertEqual(oracle.struct, json.loads(position.state.to_json()))
      self.assertIsNone(oracle.observation_struct)


if __name__ == "__main__":
  absltest.main()
