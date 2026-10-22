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

"""Tests for the checks of a session's frames against its history."""

import dataclasses

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import fake_session
from open_spiel.python.gui import frame as frame_lib
from open_spiel.python.gui import frame_checks

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_SEAT_0 = contract.Viewer(contract.ViewerKind.SEAT, 0)
_SEAT_1 = contract.Viewer(contract.ViewerKind.SEAT, 1)


def _kuhn_poker() -> tuple[fake_session.FakeSession, frame_checks.Replay]:
  """Kuhn poker after the deal, seat 0 to act, and its replayed history."""
  session = fake_session.FakeSession("kuhn_poker")
  replay = frame_checks.Replay("kuhn_poker")
  replay.follow(session.history)
  return session, replay


def _play_checking_every_frame(game_name: str) -> None:
  """Plays a well-behaved session to its end, checking every viewer's frame."""
  session = fake_session.FakeSession(game_name, seed=3)
  replay = frame_checks.Replay(game_name)
  viewers = [_ORACLE, _SPECTATOR] + [
      contract.Viewer(contract.ViewerKind.SEAT, seat)
      for seat in range(replay.num_seats)
  ]
  while True:
    replay.follow(session.history)
    for viewer in viewers:
      frame_checks.check_live_frame(session.frame(viewer), viewer, replay)
    if session.is_terminal:
      return
    session.advance(_ORACLE, max_plies=1)


class ReplayTest(absltest.TestCase):

  def test_follows_the_history_and_records_who_moved(self) -> None:
    session, replay = _kuhn_poker()
    self.assertEqual(replay.num_plies, 2)
    self.assertEqual([move.seat for move in replay.moves], [None, None])
    session.advance(_ORACLE, max_plies=1)
    replay.follow(session.history)
    self.assertEqual(replay.moves[-1].seat, 0)
    self.assertEqual(replay.moves[-1].action, session.history[-1])
    self.assertEqual(
        replay.to_move(), contract.ToMove(contract.MoverKind.SEAT, 1)
    )

  def test_declares_what_the_game_hides(self) -> None:
    self.assertTrue(frame_checks.Replay("kuhn_poker").hides_information)
    self.assertFalse(frame_checks.Replay("tic_tac_toe").hides_information)
    self.assertTrue(frame_checks.Replay("goofspiel").hides_information)

  def test_refuses_an_illegal_action(self) -> None:
    replay = frame_checks.Replay("tic_tac_toe")
    with self.assertRaisesRegex(frame_checks.InvariantError, "illegal"):
      replay.follow([9])

  def test_refuses_a_history_that_changes_played_actions(self) -> None:
    replay = frame_checks.Replay("tic_tac_toe")
    replay.follow([4])
    with self.assertRaisesRegex(frame_checks.InvariantError, "changed"):
      replay.follow([3, 5])


class CheckLiveFrameTest(absltest.TestCase):

  def test_accepts_every_frame_of_a_well_behaved_session(self) -> None:
    for game_name in ("kuhn_poker", "leduc_poker", "tic_tac_toe"):
      with self.subTest(game_name):
        _play_checking_every_frame(game_name)

  def test_refuses_a_frame_built_for_another_viewer(self) -> None:
    session, replay = _kuhn_poker()
    with self.assertRaisesRegex(frame_checks.InvariantError, "another viewer"):
      frame_checks.check_live_frame(session.frame(_SEAT_1), _SEAT_0, replay)

  def test_refuses_a_frame_of_another_position(self) -> None:
    session, replay = _kuhn_poker()
    stale = session.frame(_ORACLE)
    session.advance(_ORACLE, max_plies=1)
    replay.follow(session.history)
    with self.assertRaisesRegex(frame_checks.InvariantError, "ply 2"):
      frame_checks.check_live_frame(stale, _ORACLE, replay)

  def test_refuses_a_history_entry_with_other_rewards(self) -> None:
    session, replay = _kuhn_poker()
    frame = session.frame(_ORACLE)
    deal = dataclasses.replace(frame.history[0], rewards=(1.0, -1.0))
    wrong = dataclasses.replace(frame, history=(deal, *frame.history[1:]))
    with self.assertRaisesRegex(frame_checks.InvariantError, "rewards"):
      frame_checks.check_live_frame(wrong, _ORACLE, replay)

  def test_refuses_legal_actions_the_position_does_not_have(self) -> None:
    session, replay = _kuhn_poker()
    frame = session.frame(_SEAT_0)
    assert frame.legal_actions is not None
    fewer = dataclasses.replace(frame, legal_actions=frame.legal_actions[:1])
    with self.assertRaisesRegex(frame_checks.InvariantError, "is offered"):
      frame_checks.check_live_frame(fewer, _SEAT_0, replay)

  def test_refuses_an_unlabelled_legal_action(self) -> None:
    session, replay = _kuhn_poker()
    frame = session.frame(_SEAT_0)
    assert frame.legal_actions is not None
    unlabelled = dataclasses.replace(
        frame,
        legal_actions=tuple(
            dataclasses.replace(action, label=" ")
            for action in frame.legal_actions
        ),
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "no label"):
      frame_checks.check_live_frame(unlabelled, _SEAT_0, replay)

  def test_refuses_legal_actions_withheld_from_the_seat_to_move(self) -> None:
    session, replay = _kuhn_poker()
    frame = dataclasses.replace(session.frame(_SEAT_0), legal_actions=None)
    with self.assertRaisesRegex(frame_checks.InvariantError, "withheld"):
      frame_checks.check_live_frame(frame, _SEAT_0, replay)

  def test_refuses_the_full_state_to_a_seat_of_a_hidden_game(self) -> None:
    session, replay = _kuhn_poker()
    seat = session.frame(_SEAT_0)
    oracle = session.frame(_ORACLE)

    def leak(session: contract.SessionInfo) -> frame_lib.Frame:
      """The oracle's frame, served to seat 0 under `session`."""
      return dataclasses.replace(
          oracle,
          session=session,
          viewer=_SEAT_0,
          observation_text=seat.observation_text,
          information_state_text=seat.information_state_text,
      )

    # The frame itself refuses the full state for a seat of a game its
    # session says hides information.
    with self.assertRaisesRegex(ValueError, "omniscient exactly when"):
      leak(seat.session)
    # A session that claims the game hides nothing builds; the check
    # judges the claim against the game.
    lying = dataclasses.replace(seat.session, hides_information=False)
    with self.assertRaisesRegex(frame_checks.InvariantError, "hides nothing"):
      frame_checks.check_live_frame(leak(lying), _SEAT_0, replay)

  def test_refuses_the_deal_to_the_spectator(self) -> None:
    session, replay = _kuhn_poker()
    oracle = session.frame(_ORACLE)
    leaked = dataclasses.replace(
        session.frame(_SPECTATOR),
        history=oracle.history,
        rewards=oracle.rewards,
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "shown to"):
      frame_checks.check_live_frame(leaked, _SPECTATOR, replay)

  def test_refuses_another_seats_observation(self) -> None:
    session, replay = _kuhn_poker()
    swapped = dataclasses.replace(
        session.frame(_SEAT_0),
        observation_text=session.frame(_SEAT_1).observation_text,
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "observation"):
      frame_checks.check_live_frame(swapped, _SEAT_0, replay)

  def test_refuses_a_seat_of_a_hidden_game_no_observation(self) -> None:
    session, replay = _kuhn_poker()
    blind = dataclasses.replace(
        session.frame(_SEAT_0),
        observation_text=None,
        information_state_text=None,
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "no observation"):
      frame_checks.check_live_frame(blind, _SEAT_0, replay)

  def test_refuses_a_seat_its_observation_when_the_game_provides_it(
      self,
  ) -> None:
    session, replay = _kuhn_poker()
    # The information state alone would satisfy "some view of the seat";
    # the game provides observations, so the seat must get its own.
    no_observation = dataclasses.replace(
        session.frame(_SEAT_0), observation_text=None
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "observation"):
      frame_checks.check_live_frame(no_observation, _SEAT_0, replay)

  def test_refuses_a_seat_its_information_state_when_provided(self) -> None:
    session, replay = _kuhn_poker()
    no_information_state = dataclasses.replace(
        session.frame(_SEAT_0), information_state_text=None
    )
    with self.assertRaisesRegex(
        frame_checks.InvariantError, "information state"
    ):
      frame_checks.check_live_frame(no_information_state, _SEAT_0, replay)

  def test_refuses_wrong_returns(self) -> None:
    session = fake_session.FakeSession("tic_tac_toe")
    session.advance(_ORACLE)
    replay = frame_checks.Replay("tic_tac_toe")
    replay.follow(session.history)
    frame = session.frame(_ORACLE)
    assert frame.returns is not None
    wrong = dataclasses.replace(frame, returns=(5.0,) * len(frame.returns))
    with self.assertRaisesRegex(frame_checks.InvariantError, "returns"):
      frame_checks.check_live_frame(wrong, _ORACLE, replay)


class CheckReplayFrameTest(absltest.TestCase):

  def test_accepts_the_replay_of_every_position(self) -> None:
    session = fake_session.FakeSession("tic_tac_toe")
    session.advance(_ORACLE)
    replay = frame_checks.Replay("tic_tac_toe")
    replay.follow(session.history)
    for step in range(replay.num_plies + 1):
      frame = session.replay_frame(_ORACLE, step)
      frame_checks.check_replay_frame(frame, _ORACLE, step, replay)

  def test_accepts_every_viewers_replay_of_a_hidden_game(self) -> None:
    session = fake_session.FakeSession("kuhn_poker", seed=3)
    session.advance(_ORACLE)
    replay = frame_checks.Replay("kuhn_poker")
    replay.follow(session.history)
    for step in range(replay.num_plies + 1):
      for viewer in (_ORACLE, _SPECTATOR, _SEAT_0, _SEAT_1):
        with self.subTest(step=step, viewer=viewer):
          frame = session.replay_frame(viewer, step)
          frame_checks.check_replay_frame(frame, viewer, step, replay)

  def test_refuses_a_replay_that_shows_a_seat_the_deal(self) -> None:
    session = fake_session.FakeSession(
        "kuhn_poker", fault=fake_session.Fault.REPLAY_LEAKS_HISTORY
    )
    session.advance(_ORACLE)
    replay = frame_checks.Replay("kuhn_poker")
    replay.follow(session.history)
    frame = session.replay_frame(_SEAT_0, 2)
    with self.assertRaisesRegex(frame_checks.InvariantError, "shown to seat 0"):
      frame_checks.check_replay_frame(frame, _SEAT_0, 2, replay)

  def test_refuses_a_seat_replay_with_the_live_observation(self) -> None:
    session = fake_session.FakeSession("kuhn_poker", seed=3)
    session.advance(_ORACLE)
    replay = frame_checks.Replay("kuhn_poker")
    replay.follow(session.history)
    live = session.frame(_SEAT_0)
    stale = dataclasses.replace(
        session.replay_frame(_SEAT_0, 0),
        observation_text=live.observation_text,
    )
    with self.assertRaisesRegex(frame_checks.InvariantError, "observation"):
      frame_checks.check_replay_frame(stale, _SEAT_0, 0, replay)

  def test_refuses_the_replay_of_another_position(self) -> None:
    session = fake_session.FakeSession("tic_tac_toe")
    session.advance(_ORACLE)
    replay = frame_checks.Replay("tic_tac_toe")
    replay.follow(session.history)
    frame = session.replay_frame(_ORACLE, 1)
    with self.assertRaisesRegex(frame_checks.InvariantError, "ply 2"):
      frame_checks.check_replay_frame(frame, _ORACLE, 2, replay)


if __name__ == "__main__":
  absltest.main()
