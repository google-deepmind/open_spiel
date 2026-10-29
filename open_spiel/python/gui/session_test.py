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


"""Tests for sessions: turns, privacy, chance and controllers.

Undo, reset and replay are tested in session_history_test.py.
"""

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import loader
from open_spiel.python.gui import request
from open_spiel.python.gui import session as session_lib
import pyspiel

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_SEAT_0 = contract.Viewer(contract.ViewerKind.SEAT, 0)
_SEAT_1 = contract.Viewer(contract.ViewerKind.SEAT, 1)
_HUMAN = controllers.HUMAN_SPEC
_RANDOM = controllers.Spec(controllers.RANDOM)
_AUTO = contract.ChancePolicy.AUTO
_EXPOSE = contract.ChancePolicy.EXPOSE


def _create(
    game: str,
    *,
    players: tuple[controllers.Spec, ...] = (_HUMAN, _HUMAN),
    chance_policy: contract.ChancePolicy = _AUTO,
    autoplay: tuple[bool, ...] = (),
    seed: int = 1,
) -> session_lib.Session:
  """Starts a session of `game` with no parameters."""
  return session_lib.Session.create(
      game,
      {},
      seed=seed,
      chance_policy=chance_policy,
      controllers=players,
      autoplay=autoplay,
  )


def _ids(
    choices: Sequence[contract.Action | contract.ChanceOutcome] | None,
) -> list[int] | None:
  """Returns the ids of offered actions or outcomes, or None if withheld."""
  return None if choices is None else [choice.id for choice in choices]


_FAILED = contract.ControllerFailure(
    "RuntimeError", "search exhausted its budget"
)


class _Raising:
  """A controller that always fails."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Raises."""
    del state, seat
    raise RuntimeError("search exhausted its budget")


class _Illegal:
  """A controller that always chooses an action that is never legal."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns an action id no game has."""
    del state, seat
    return -1


def _non_humans_played_by(
    controller: object,
) -> contextlib.AbstractContextManager[object]:
  """Builds every seat that is not a human's as `controller`."""
  builder = controllers.builder

  def fake_builder(
      spec: controllers.Spec, game: pyspiel.Game
  ) -> Callable[[int], object]:
    if spec.kind == controllers.HUMAN:
      return builder(spec, game)
    return lambda _: controller

  return mock.patch.object(controllers, "builder", side_effect=fake_builder)


class CreateTest(absltest.TestCase):

  def test_starts_at_the_initial_position(self) -> None:
    frame = _create("tic_tac_toe").frame(_SEAT_0)
    self.assertEqual(frame.ply, 0)
    self.assertEqual(frame.to_move, contract.ToMove(contract.MoverKind.SEAT, 0))
    self.assertEqual(_ids(frame.legal_actions), list(range(9)))
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)

  def test_refuses_an_unknown_game(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      _create("no_such_game")

  def test_refuses_a_game_it_cannot_serve(self) -> None:
    with self.assertRaises(errors.GameUnavailableError):
      _create("mfg_crowd_modelling", players=(_HUMAN,))

  def test_refuses_parameters_the_game_refuses(self) -> None:
    with self.assertRaises(errors.BadRequestError):
      session_lib.Session.create(
          "tic_tac_toe",
          {"no_such_parameter": 1},
          chance_policy=_AUTO,
          controllers=(_HUMAN, _HUMAN),
      )

  def test_refuses_the_wrong_number_of_seats(self) -> None:
    with self.assertRaises(errors.BadRequestError):
      _create("tic_tac_toe", players=(_HUMAN,))

  def test_refuses_an_autoplaying_human(self) -> None:
    with self.assertRaises(errors.ControllerUnavailableError):
      _create("tic_tac_toe", autoplay=(True, False))

  def test_refuses_a_controller_that_does_not_exist(self) -> None:
    with self.assertRaises(errors.ControllerUnavailableError):
      _create("tic_tac_toe", players=(_HUMAN, controllers.Spec("oracle_bot")))

  def test_serves_a_simultaneous_game_turn_by_turn(self) -> None:
    session = _create("matrix_rps")
    session.apply_action(_SEAT_0, 0)
    frame = session.frame(_SEAT_1)
    self.assertTrue(frame.session.turn_based)
    self.assertEqual(frame.to_move.seat, 1)
    self.assertIsNone(frame.history[0].action)


class TurnTest(absltest.TestCase):

  def test_a_seat_plays_its_own_turn(self) -> None:
    session = _create("tic_tac_toe")
    frame = session.apply_action(_SEAT_0, 4)
    self.assertEqual(session.history, (4,))
    self.assertEqual(frame.to_move.seat, 1)

  def test_a_seat_cannot_play_another_seats_turn(self) -> None:
    with self.assertRaises(errors.NotYourTurnError):
      _create("tic_tac_toe").apply_action(_SEAT_1, 4)

  def test_a_spectator_cannot_play(self) -> None:
    with self.assertRaises(errors.NotYourTurnError):
      _create("tic_tac_toe").apply_action(_SPECTATOR, 4)

  def test_the_oracle_plays_any_human_seat(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_ORACLE, 4)
    session.apply_action(_ORACLE, 0)
    self.assertEqual(session.history, (4, 0))

  def test_nobody_plays_a_controllers_turn(self) -> None:
    session = _create("tic_tac_toe", players=(_RANDOM, _HUMAN))
    with self.assertRaises(errors.NotYourTurnError):
      session.apply_action(_ORACLE, 4)

  def test_an_illegal_action_is_refused_with_its_ply(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.IllegalActionError) as raised:
      session.apply_action(_SEAT_1, 4)
    self.assertEqual(raised.exception.ply, 1)
    self.assertEqual(session.history, (4,))

  def test_a_finished_game_is_terminal(self) -> None:
    session = _create("tic_tac_toe")
    for seat, action in ((0, 0), (1, 3), (0, 1), (1, 4), (0, 2)):
      session.apply_action(
          contract.Viewer(contract.ViewerKind.SEAT, seat), action
      )
    frame = session.frame(_SPECTATOR)
    self.assertTrue(session.is_terminal)
    self.assertEqual(frame.status, contract.SessionStatus.TERMINAL)
    self.assertEqual(frame.returns, (1.0, -1.0))
    self.assertEqual(frame.legal_actions, ())


class PrivacyTest(absltest.TestCase):

  def test_the_session_says_whether_the_game_hides_information(self) -> None:
    self.assertTrue(
        _create("kuhn_poker").frame(_SPECTATOR).session.hides_information
    )
    frame = _create("tic_tac_toe").frame(_SPECTATOR)
    self.assertFalse(frame.session.hides_information)
    self.assertTrue(frame.omniscient)

  def test_a_seat_sees_its_observation_and_not_the_deal(self) -> None:
    frame = _create("kuhn_poker").frame(_SEAT_0)
    self.assertFalse(frame.omniscient)
    self.assertIsNone(frame.state_text)
    self.assertIsNone(frame.session.seed)
    self.assertIsNotNone(frame.observation_text)
    self.assertTrue(all(entry.withheld for entry in frame.history))

  def test_a_seat_sees_its_own_actions_only(self) -> None:
    session = _create("kuhn_poker")
    session.apply_action(_SEAT_0, 1)
    own, other = session.frame(_SEAT_0), session.frame(_SEAT_1)
    self.assertEqual(own.history[-1].action, 1)
    self.assertIsNone(other.history[-1].action)
    self.assertEqual(other.history[-1].seat, 0)

  def test_a_seat_gets_legal_actions_only_on_its_turn(self) -> None:
    session = _create("kuhn_poker")
    self.assertEqual(_ids(session.frame(_SEAT_0).legal_actions), [0, 1])
    self.assertIsNone(session.frame(_SEAT_1).legal_actions)
    self.assertIsNone(session.frame(_SPECTATOR).legal_actions)

  def test_a_spectator_sees_only_who_moved(self) -> None:
    frame = _create("kuhn_poker").frame(_SPECTATOR)
    self.assertIsNone(frame.observation_text)
    self.assertEqual([entry.seat for entry in frame.history], [None, None])
    self.assertTrue(all(entry.withheld for entry in frame.history))

  def test_the_oracle_sees_everything(self) -> None:
    frame = _create("kuhn_poker", seed=5).frame(_ORACLE)
    self.assertTrue(frame.omniscient)
    self.assertEqual(frame.session.seed, 5)
    self.assertIsNotNone(frame.state_text)
    self.assertFalse(any(entry.withheld for entry in frame.history))

  def test_everyone_sees_a_game_that_hides_nothing(self) -> None:
    frame = _create("tic_tac_toe").frame(_SPECTATOR)
    self.assertTrue(frame.omniscient)
    self.assertIsNotNone(frame.state_text)
    self.assertIsNone(frame.session.seed)

  def test_a_game_hiding_its_declared_state_is_treated_as_hidden(self) -> None:
    frame = _create("python_hangman", players=(_HUMAN,)).frame(_SEAT_0)
    self.assertFalse(frame.omniscient)
    self.assertIsNone(frame.state_text)
    self.assertIsNone(frame.information_state_text)
    self.assertIsNotNone(frame.observation_text)

  def test_a_seat_of_a_game_without_seat_views_is_refused(self) -> None:
    loaded = dataclasses.replace(
        loader.load("kuhn_poker", {}),
        provides_observation=False,
        provides_information_state=False,
    )
    session = session_lib.Session(loaded, seed=1, chance_policy=_AUTO)
    with self.assertRaises(errors.ViewerUnavailableError):
      session.frame(_SEAT_0)
    self.assertIsNotNone(session.frame(_ORACLE).state_text)

  def test_a_seat_the_game_lacks_is_not_found(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      _create("tic_tac_toe").frame(contract.Viewer(contract.ViewerKind.SEAT, 2))


class ChanceTest(absltest.TestCase):

  def test_auto_resolves_chance_before_the_first_decision(self) -> None:
    frame = _create("kuhn_poker").frame(_ORACLE)
    self.assertEqual(frame.ply, 2)
    self.assertEqual(frame.to_move.kind, contract.MoverKind.SEAT)
    self.assertEqual(frame.chance_outcomes, ())

  def test_a_seed_reproduces_the_deal(self) -> None:
    self.assertEqual(
        _create("kuhn_poker", seed=11).history,
        _create("kuhn_poker", seed=11).history,
    )

  def test_expose_offers_outcomes_to_the_oracle_only(self) -> None:
    session = _create("kuhn_poker", chance_policy=_EXPOSE)
    self.assertEqual(_ids(session.frame(_ORACLE).chance_outcomes), [0, 1, 2])
    self.assertIsNone(session.frame(_SEAT_0).chance_outcomes)
    with self.assertRaises(errors.NotYourTurnError):
      session.apply_chance(_SEAT_0, 0)

  def test_the_oracle_chooses_an_exposed_outcome(self) -> None:
    session = _create("kuhn_poker", chance_policy=_EXPOSE)
    session.apply_chance(_ORACLE, 2)
    session.apply_chance(_ORACLE, 0)
    self.assertEqual(session.history, (2, 0))
    with self.assertRaises(errors.NotYourTurnError):
      session.apply_chance(_ORACLE, 1)

  def test_an_impossible_outcome_is_refused(self) -> None:
    session = _create("kuhn_poker", chance_policy=_EXPOSE)
    session.apply_chance(_ORACLE, 2)
    with self.assertRaises(errors.IllegalActionError):
      session.apply_chance(_ORACLE, 2)

  def test_auto_never_takes_a_chosen_outcome(self) -> None:
    with self.assertRaises(errors.NotYourTurnError):
      _create("kuhn_poker").apply_chance(_ORACLE, 0)


class ControllerTest(absltest.TestCase):

  def test_an_autoplaying_seat_replies_at_once(self) -> None:
    session = _create(
        "tic_tac_toe", players=(_HUMAN, _RANDOM), autoplay=(False, True)
    )
    frame = session.apply_action(_SEAT_0, 4)
    self.assertEqual(frame.ply, 2)
    self.assertEqual(frame.to_move.seat, 0)

  def test_autoplay_stops_after_the_most_plies_an_advance_applies(
      self,
  ) -> None:
    with mock.patch.object(request, "MAX_ADVANCE_PLIES", 3):
      session = _create(
          "tic_tac_toe", players=(_RANDOM, _RANDOM), autoplay=(True, True)
      )
    self.assertLen(session.history, 3)
    self.assertFalse(session.is_terminal)

  def test_advance_plays_until_a_human_must_act(self) -> None:
    session = _create("tic_tac_toe", players=(_RANDOM, _HUMAN))
    self.assertEqual(session.history, ())
    frame = session.advance(_SPECTATOR)
    self.assertEqual(frame.ply, 1)
    self.assertEqual(frame.to_move.seat, 1)

  def test_advance_stops_after_max_plies(self) -> None:
    session = _create("tic_tac_toe", players=(_RANDOM, _RANDOM))
    self.assertEqual(session.advance(_SPECTATOR, max_plies=3).ply, 3)
    frame = session.advance(_SPECTATOR)
    self.assertEqual(frame.status, contract.SessionStatus.TERMINAL)
    self.assertTrue(session.is_terminal)

  def test_advance_never_stops_at_an_auto_chance_node(self) -> None:
    session = _create("pig", players=(_RANDOM, _RANDOM))
    while not session.is_terminal:
      frame = session.advance(_SPECTATOR, max_plies=1)
      self.assertNotEqual(frame.to_move.kind, contract.MoverKind.CHANCE)

  def test_a_failing_controller_pauses_its_seat(self) -> None:
    with _non_humans_played_by(_Raising()):
      session = _create("tic_tac_toe", players=(_HUMAN, _RANDOM))
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.ControllerFailedError) as raised:
      session.advance(_SEAT_0)
    self.assertEqual(raised.exception.seat, 1)
    frame = session.frame(_ORACLE)
    self.assertEqual(frame.status, contract.SessionStatus.PAUSED)
    self.assertEqual(frame.seats[1].failure, _FAILED)
    self.assertEqual(session.history, (4,))

  def test_nothing_plays_in_a_failed_seats_place(self) -> None:
    with _non_humans_played_by(_Illegal()):
      session = _create("tic_tac_toe", players=(_RANDOM, _HUMAN))
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SPECTATOR)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SPECTATOR)
    self.assertEqual(session.history, ())
    self.assertEqual(
        session.frame(_ORACLE).seats[0].failure,
        contract.ControllerFailure(
            "IllegalAction",
            "Seat 0's controller chose -1, which is not one of its legal "
            "actions.",
        ),
    )

  def test_a_failure_message_is_shown_only_to_omniscient_viewers(self) -> None:
    with _non_humans_played_by(_Raising()):
      session = _create(
          "kuhn_poker", players=(_RANDOM, _HUMAN), autoplay=(True, False)
      )
    self.assertEqual(
        session.frame(_SEAT_1).seats[0].failure,
        contract.ControllerFailure("RuntimeError", None),
    )
    self.assertEqual(session.frame(_ORACLE).seats[0].failure, _FAILED)

  def test_changing_the_controller_resumes_the_seat(self) -> None:
    with _non_humans_played_by(_Raising()):
      session = _create("tic_tac_toe", players=(_RANDOM, _HUMAN))
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SPECTATOR)
    session.set_controller(_ORACLE, 0, _HUMAN)
    frame = session.apply_action(_SEAT_0, 4)
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)

  def test_a_seat_the_game_lacks_cannot_be_given_a_controller(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      _create("tic_tac_toe").set_controller(_ORACLE, 2, _RANDOM)

  def test_any_viewer_but_a_spectator_changes_a_seat(self) -> None:
    session = _create("kuhn_poker")
    frame = session.set_controller(_SEAT_0, 1, _RANDOM)
    self.assertEqual(frame.seats[1].controller, controllers.RANDOM)
    self.assertFalse(frame.omniscient)
    for game in ("kuhn_poker", "tic_tac_toe"):
      with self.assertRaises(errors.NotYourTurnError):
        _create(game).set_controller(_SPECTATOR, 1, _RANDOM)

  def test_the_session_lists_controllers_it_cannot_use_with_a_reason(
      self,
  ) -> None:
    info = _create("tic_tac_toe").frame(_ORACLE).session
    names = [controller.name for controller in info.controllers]
    self.assertEqual(names[0], contract.HUMAN)
    self.assertEqual(
        names, [controller.name for controller in controllers.registered()]
    )
    for controller in info.controllers:
      self.assertIsNone(controller.unavailable_reason)


if __name__ == "__main__":
  absltest.main()
