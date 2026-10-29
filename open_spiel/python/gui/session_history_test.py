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


"""Tests for a session's history: undo, reset, replay and playing on."""

from collections.abc import Callable
import contextlib
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import plies
from open_spiel.python.gui import session as session_lib
import pyspiel

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_SEAT_0 = contract.Viewer(contract.ViewerKind.SEAT, 0)
_SEAT_1 = contract.Viewer(contract.ViewerKind.SEAT, 1)
_HUMAN = controllers.HUMAN_SPEC
_RANDOM = controllers.Spec(controllers.RANDOM)


def _create(
    game: str,
    *,
    players: tuple[controllers.Spec, ...] = (_HUMAN, _HUMAN),
    autoplay: tuple[bool, ...] = (),
    seed: int = 1,
) -> session_lib.Session:
  """Starts a session of `game` with no parameters, under `auto` chance."""
  return session_lib.Session.create(
      game,
      {},
      seed=seed,
      chance_policy=contract.ChancePolicy.AUTO,
      controllers=players,
      autoplay=autoplay,
  )


class _Raising:
  """A controller that always fails."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Raises."""
    del state, seat
    raise RuntimeError("search exhausted its budget")


def _raising_non_humans() -> contextlib.AbstractContextManager[object]:
  """Builds every seat that is not a human's as a controller that fails."""
  builder = controllers.builder

  def fake_builder(
      spec: controllers.Spec, game: pyspiel.Game
  ) -> Callable[[int], object]:
    if spec.kind == controllers.HUMAN:
      return builder(spec, game)
    return lambda _: _Raising()

  return mock.patch.object(controllers, "builder", side_effect=fake_builder)


class UndoTest(absltest.TestCase):

  def test_a_seat_takes_back_its_last_action(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    frame = session.undo(_SEAT_0)
    self.assertEqual(session.history, ())
    self.assertEqual(frame.num_plies, 0)

  def test_a_seat_takes_back_the_controller_reply_with_its_action(
      self,
  ) -> None:
    session = _create(
        "tic_tac_toe", players=(_HUMAN, _RANDOM), autoplay=(False, True)
    )
    session.apply_action(_SEAT_0, 4)
    self.assertLen(session.history, 2)
    session.undo(_SEAT_0)
    self.assertEqual(session.history, ())

  def test_a_seat_cannot_take_back_another_persons_action(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0)
    with self.assertRaises(errors.NothingToUndoError):
      session.undo(_SEAT_0)
    session.undo(_SEAT_1)
    self.assertEqual(session.history, (4,))

  def test_the_oracle_takes_back_the_last_action(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0)
    session.undo(_ORACLE)
    self.assertEqual(session.history, (4,))

  def test_undo_keeps_the_chance_outcomes_before_the_action(self) -> None:
    session = _create("kuhn_poker")
    deal = session.history
    session.apply_action(_SEAT_0, 1)
    session.undo(_SEAT_0)
    self.assertEqual(session.history, deal)

  def test_a_chance_outcome_is_drawn_afresh_after_an_undo(self) -> None:
    # Pig: seat 0 rolls (action 0), then chance throws the die. Were the
    # throw seeded by its ply, every retry would see the same face.
    session = _create("pig", seed=3)
    faces = set()
    for _ in range(8):
      session.apply_action(_SEAT_0, 0)
      faces.add(session.history[1])
      session.undo(_SEAT_0)
      self.assertEqual(session.history, ())
    self.assertGreater(len(faces), 1)

  def test_there_may_be_nothing_to_undo(self) -> None:
    session = _create("tic_tac_toe")
    with self.assertRaises(errors.NothingToUndoError):
      session.undo(_SEAT_0)
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.NothingToUndoError):
      session.undo(_SPECTATOR)

  def test_undo_resumes_a_paused_session(self) -> None:
    with _raising_non_humans():
      session = _create("tic_tac_toe", players=(_HUMAN, _RANDOM))
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SEAT_0)
    frame = session.undo(_SEAT_0)
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)


class ResetTest(absltest.TestCase):

  def test_reset_replays_a_history(self) -> None:
    session = _create("tic_tac_toe")
    session.reset(_SEAT_0, history=(4, 0, 8))
    self.assertEqual(session.history, (4, 0, 8))
    self.assertEqual(session.frame(_SEAT_1).to_move.seat, 1)

  def test_reset_to_nothing_restarts_the_game(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.reset(_SEAT_0)
    self.assertEqual(session.history, ())

  def test_reset_refuses_an_illegal_history_and_keeps_the_game(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.IllegalActionError) as raised:
      session.reset(_SEAT_0, history=(0, 0))
    self.assertEqual(raised.exception.ply, 1)
    self.assertEqual(session.history, (4,))

  def test_reset_resolves_auto_chance_after_the_history(self) -> None:
    session = _create("kuhn_poker")
    session.reset(_ORACLE, history=(2,))
    self.assertLen(session.history, 2)
    self.assertEqual(session.history[0], 2)

  def test_a_spectator_never_resets(self) -> None:
    for game in ("kuhn_poker", "tic_tac_toe"):
      session = _create(game)
      before = session.history
      with self.assertRaises(errors.NotYourTurnError):
        session.reset(_SPECTATOR, history=())
      self.assertEqual(session.history, before)

  def test_a_seat_resets_a_game_that_hides_information(self) -> None:
    session = _create("kuhn_poker")
    frame = session.reset(_SEAT_0, history=(0, 1))
    self.assertEqual(session.history[:2], (0, 1))
    self.assertFalse(frame.omniscient)


class ReplayTest(absltest.TestCase):

  def test_a_replay_shows_an_earlier_position(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0)
    frame = session.replay_frame(_SEAT_0, 1)
    self.assertEqual(frame.ply, 1)
    self.assertEqual(frame.num_plies, 2)
    self.assertFalse(frame.live)
    self.assertEqual(
        [action.id for action in frame.legal_actions or ()],
        [0, 1, 2, 3, 5, 6, 7, 8],
    )
    self.assertEqual(session.history, (4, 0))

  def test_a_replay_offers_actions_only_to_an_omniscient_viewer(self) -> None:
    session = _create("kuhn_poker")
    session.apply_action(_SEAT_0, 1)
    session.apply_action(_SEAT_1, 0)
    self.assertEqual(session.replay_frame(_SEAT_1, 3).legal_actions, ())
    self.assertEqual(
        [
            action.id
            for action in session.replay_frame(_ORACLE, 3).legal_actions or ()
        ],
        [0, 1],
    )

  def test_a_replay_keeps_the_viewers_privacy(self) -> None:
    session = _create("kuhn_poker")
    session.apply_action(_SEAT_0, 1)
    frame = session.replay_frame(_SEAT_1, 2)
    self.assertIsNone(frame.state_text)
    self.assertIsNone(frame.history[2].action)

  def test_a_replay_past_the_live_game_is_not_found(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      _create("tic_tac_toe").replay_frame(_SEAT_0, 1)

  def test_a_replay_that_drifts_from_play_is_refused(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0)
    drift = plies.DriftError("Ply 0 gave the hidden card 2.", ply=0)
    with mock.patch.object(plies, "rebuild", side_effect=drift):
      with self.assertRaises(session_lib.ReplayUnsupportedError) as raised:
        session.replay_frame(_SEAT_0, 1)
    message = str(raised.exception)
    self.assertIn("ply 0", message)
    self.assertNotIn("hidden card", message)


def _create_exposed(game: str) -> session_lib.Session:
  """Starts a session of `game` whose client chooses chance outcomes."""
  return session_lib.Session.create(
      game,
      {},
      seed=1,
      chance_policy=contract.ChancePolicy.EXPOSE,
      controllers=(_HUMAN, _HUMAN),
  )


class PlayFromEarlierTest(absltest.TestCase):

  def test_playing_at_an_earlier_ply_takes_back_the_later_plies(self) -> None:
    session = _create("tic_tac_toe")
    for viewer, action in ((_SEAT_0, 4), (_SEAT_1, 0), (_SEAT_0, 8)):
      session.apply_action(viewer, action)
    frame = session.apply_action(_SEAT_1, 2, ply=1)
    self.assertEqual(session.history, (4, 2))
    self.assertTrue(frame.live)
    self.assertEqual(frame.to_move.seat, 0)

  def test_playing_at_the_live_ply_plays_on(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0, ply=1)
    self.assertEqual(session.history, (4, 0))

  def test_an_illegal_action_at_an_earlier_ply_changes_nothing(self) -> None:
    session = _create("tic_tac_toe")
    for viewer, action in ((_SEAT_0, 4), (_SEAT_1, 0), (_SEAT_0, 8)):
      session.apply_action(viewer, action)
    with self.assertRaises(errors.IllegalActionError):
      session.apply_action(_SEAT_1, 4, ply=1)
    self.assertEqual(session.history, (4, 0, 8))

  def test_the_seat_to_move_there_plays(self) -> None:
    session = _create("tic_tac_toe")
    session.apply_action(_SEAT_0, 4)
    session.apply_action(_SEAT_1, 0)
    with self.assertRaises(errors.NotYourTurnError):
      session.apply_action(_SEAT_0, 2, ply=1)
    self.assertEqual(session.history, (4, 0))

  def test_only_an_omniscient_viewer_plays_from_an_earlier_position(
      self,
  ) -> None:
    session = _create("kuhn_poker")
    session.apply_action(_SEAT_0, 1)
    session.apply_action(_SEAT_1, 0)
    played = session.history
    with self.assertRaises(errors.NotYourTurnError):
      session.apply_action(_SEAT_1, 1, ply=3)
    self.assertEqual(session.history, played)
    session.apply_action(_ORACLE, 1, ply=3)
    self.assertEqual(session.history, (*played[:3], 1))

  def test_a_ply_past_the_live_game_is_not_found(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      _create("tic_tac_toe").apply_action(_SEAT_0, 4, ply=1)

  def test_playing_from_an_earlier_position_resumes_a_paused_session(
      self,
  ) -> None:
    with _raising_non_humans():
      session = _create("tic_tac_toe", players=(_HUMAN, _RANDOM))
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SEAT_0)
    frame = session.apply_action(_SEAT_0, 0, ply=0)
    self.assertEqual(frame.status, contract.SessionStatus.PLAYING)
    self.assertEqual(session.history, (0,))

  def test_a_paused_session_refuses_to_play_at_the_live_position(
      self,
  ) -> None:
    with _raising_non_humans():
      session = _create("tic_tac_toe", players=(_HUMAN, _RANDOM))
    session.apply_action(_SEAT_0, 4)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_SEAT_0)
    with self.assertRaises(errors.ControllerFailedError):
      session.apply_action(_SEAT_0, 0, ply=1)

  def test_a_chance_outcome_is_chosen_again_at_an_earlier_ply(self) -> None:
    session = _create_exposed("kuhn_poker")
    session.apply_chance(_ORACLE, 0)
    session.apply_chance(_ORACLE, 1)
    replay = session.replay_frame(_ORACLE, 0)
    self.assertEqual(
        [outcome.id for outcome in replay.chance_outcomes or ()], [0, 1, 2]
    )
    session.apply_chance(_ORACLE, 2, ply=0)
    self.assertEqual(session.history, (2,))


if __name__ == "__main__":
  absltest.main()
