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

"""Tests for a seat's notice: who receives it, when it clears, what it is."""

from collections.abc import Mapping
import dataclasses
from unittest import mock

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_registry
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import session as session_lib

_NAME = "noting"
_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)


def _seat(index: int) -> contract.Viewer:
  """The viewer of seat `index`."""
  return contract.Viewer(contract.ViewerKind.SEAT, index)


class _Noting:
  """Plays the lowest legal action, with its setting's notice (or none)."""

  def __init__(self, notice: object) -> None:
    self._notice = notice
    self.last_notice: object = None

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Plays the lowest legal action, noting `notice` about it."""
    self.last_notice = self._notice
    return int(state.legal_actions(seat)[0])


def _build(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> controllers.Controller:
  """Builds the noting controller; notice "" gives none, "!" one too long."""
  del game, seed
  text = str(config["notice"])
  notice: object = {"": None, "!": "x" * 201, "#": 7}.get(text, text)
  return _Noting(notice)


_KIND = controllers.Kind(
    "Noting",
    (
        controllers.Setting("notice", str, "", "The notice to give."),
        controllers.Setting(
            "full_state", bool, False, "Whether it reads the full state."
        ),
    ),
    reads_full_state=False,
    build=_build,
    config_reads_full_state=lambda config: bool(config["full_state"]),
)


class NoticeTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    controllers.register(_NAME, _KIND)
    self.addCleanup(controllers.unregister, _NAME)

  def open(
      self, game: str, notice: str, full_state: bool = False
  ) -> session_lib.Session:
    """Opens `game`, seat 0 human and seat 1 noting `notice`."""
    return session_lib.Session.create(
        game,
        {},
        seed=1,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=[
            controllers.HUMAN_SPEC,
            controllers.Spec(
                _NAME, {"notice": notice, "full_state": full_state}
            ),
        ],
    )

  def notices(self, session: session_lib.Session) -> dict[str, str | None]:
    """Seat 1's notice, as each viewer receives it."""
    return {
        name: session.frame(viewer).seats[1].notice
        for name, viewer in (
            ("oracle", _ORACLE),
            ("spectator", _SPECTATOR),
            ("seat 0", _seat(0)),
            ("seat 1", _seat(1)),
        )
    }

  def test_in_a_game_that_hides_information_only_the_seat_and_oracle(
      self,
  ) -> None:
    session = self.open("kuhn_poker", "fell back to a safe move")
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    note = "fell back to a safe move"
    self.assertEqual(
        self.notices(session),
        {"oracle": note, "spectator": None, "seat 0": None, "seat 1": note},
    )

  def test_in_a_game_that_hides_nothing_every_viewer_is_omniscient(
      self,
  ) -> None:
    session = self.open("tic_tac_toe", "book move")
    session.apply_action(_seat(0), 4)
    session.advance(_ORACLE)
    self.assertEqual(set(self.notices(session).values()), {"book move"})

  def test_a_seat_without_a_notice_has_none(self) -> None:
    session = self.open("tic_tac_toe", "")
    session.apply_action(_seat(0), 4)
    session.advance(_ORACLE)
    self.assertEqual(set(self.notices(session).values()), {None})

  def test_an_undo_a_reset_or_a_new_controller_clears_it(self) -> None:

    def undo(session: session_lib.Session) -> None:
      session.undo(_ORACLE)

    def reset(session: session_lib.Session) -> None:
      session.reset(_ORACLE, history=[4])

    def reseat(session: session_lib.Session) -> None:
      session.set_controller(_ORACLE, 1, controllers.Spec("random"))

    for clear in (undo, reset, reseat):
      with self.subTest(clear.__name__):
        session = self.open("tic_tac_toe", "book move")
        session.apply_action(_seat(0), 4)
        session.advance(_ORACLE)
        self.assertEqual(session.frame(_ORACLE).seats[1].notice, "book move")
        clear(session)
        self.assertIsNone(session.frame(_ORACLE).seats[1].notice)

  def test_the_next_choice_without_one_clears_it(self) -> None:
    session = self.open("tic_tac_toe", "book move")
    session.apply_action(_seat(0), 4)
    session.advance(_ORACLE)
    session.set_controller(
        _ORACLE, 1, controllers.Spec(_NAME, {"notice": "first"})
    )
    session.apply_action(_seat(0), 8)
    session.advance(_ORACLE)
    self.assertEqual(session.frame(_ORACLE).seats[1].notice, "first")

  def test_a_notice_does_not_change_play(self) -> None:
    quiet, noisy = self.open("tic_tac_toe", ""), self.open("tic_tac_toe", "n")
    for session in (quiet, noisy):
      session.apply_action(_seat(0), 4)
      session.advance(_ORACLE)
    self.assertEqual(quiet.history, noisy.history)
    self.assertEqual(quiet.frame(_ORACLE).status, noisy.frame(_ORACLE).status)

  def test_a_notice_that_is_not_short_text_fails_the_seat(self) -> None:
    for notice in ("!", "#"):
      with self.subTest(notice):
        session = self.open("tic_tac_toe", notice)
        session.apply_action(_seat(0), 4)
        with self.assertRaises(errors.ControllerFailedError):
          session.advance(_ORACLE)
        failure = session.frame(_ORACLE).seats[1].failure
        assert failure is not None
        self.assertEqual(failure.error, "InvalidNotice")

  def test_a_frame_refuses_another_seat_s_notice(self) -> None:
    session = self.open("kuhn_poker", "note")
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    with self.assertRaisesRegex(ValueError, "notice"):
      dataclasses.replace(
          session.frame(_seat(0)),
          seats=session.frame(_ORACLE).seats,
      )

  def test_a_full_state_seat_s_notice_reaches_its_seat(self) -> None:
    # Under the analysis policy a full-state seat's notice is passed on, as
    # any other: free text from a bot that sees the whole state, which may
    # carry hidden information to the seat's own viewer.
    session = self.open("kuhn_poker", "saw everything", full_state=True)
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    self.assertEqual(self.notices(session)["seat 1"], "saw everything")

  def test_the_policy_hook_can_drop_a_full_state_seat_s_notice(self) -> None:
    # A stricter policy (a match) would drop or limit, in
    # `full_state_notice`, the notices of seats that read the full state.
    asked = []

    def drop_full_state(
        game: pyspiel.Game, reads_full_state: bool, notice: str | None
    ) -> str | None:
      asked.append((game.get_type().short_name, reads_full_state, notice))
      return None if reads_full_state else notice

    with mock.patch.object(
        controller_registry, "full_state_notice", drop_full_state
    ):
      for full_state in (True, False):
        session = self.open("kuhn_poker", "a note", full_state=full_state)
        session.apply_action(_seat(0), 0)
        session.advance(_ORACLE)
        self.assertEqual(
            self.notices(session)["seat 1"],
            None if full_state else "a note",
        )
    self.assertEqual(
        asked,
        [("kuhn_poker", True, "a note"), ("kuhn_poker", False, "a note")],
    )

  def test_a_human_seat_has_no_notice(self) -> None:
    with self.assertRaisesRegex(ValueError, "notice"):
      contract.SeatView(
          seat=0,
          controller=contract.HUMAN,
          config={},
          autoplay=False,
          failure=None,
          notice="hi",
      )


if __name__ == "__main__":
  absltest.main()
