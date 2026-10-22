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

"""Tests for the frame's visibility and consistency invariants."""

import dataclasses
from typing import Any

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import frame

_DEAL = (
    contract.HistoryEntry(None, 2, "Deal:2", (0.0, 0.0)),
    contract.HistoryEntry(None, 0, "Deal:0", (0.0, 0.0)),
)
_HIDDEN_DEAL = (contract.HistoryEntry(None, None, None, None),) * 2
_ACTIONS = (contract.Action(0, "Pass"), contract.Action(1, "Bet"))
_HUMAN = contract.Controller(contract.HUMAN, "Human", {})
_MCTS = contract.Controller("mcts", "MCTS", {"type": "object"})
_SEATS = (
    contract.SeatView(0, contract.HUMAN, {}, False, None),
    contract.SeatView(1, contract.HUMAN, {}, False, None),
)
_FAILED_SEATS = (
    _SEATS[0],
    contract.SeatView(
        1, "mcts", {}, True, contract.ControllerFailure("RuntimeError", "oom")
    ),
)
_SESSION = contract.SessionInfo(
    id="s",
    game="kuhn_poker",
    parameters={"players": 2},
    num_seats=2,
    chance_policy=contract.ChancePolicy.AUTO,
    turn_based=False,
    hides_information=True,
    seed=None,
    controllers=(_HUMAN, _MCTS),
)
_EXPOSED = dataclasses.replace(
    _SESSION, chance_policy=contract.ChancePolicy.EXPOSE
)

# Kuhn poker after the deal, seat 0 to act, as the oracle sees it.
_ORACLE_FRAME = frame.Frame(
    session=dataclasses.replace(_SESSION, seed=7),
    viewer=contract.Viewer(contract.ViewerKind.ORACLE),
    omniscient=True,
    status=contract.SessionStatus.PLAYING,
    seats=_SEATS,
    ply=2,
    num_plies=2,
    to_move=contract.ToMove(contract.MoverKind.SEAT, 0),
    legal_actions=_ACTIONS,
    chance_outcomes=(),
    history=_DEAL,
    rewards=(0.0, 0.0),
    returns=None,
    state_text="2 0",
    observation_text=None,
    information_state_text=None,
    struct={"cards": [2, 0]},
)


def _seat_frame(seat: int) -> frame.Frame:
  """The same position, as seat `seat` sees it."""
  return dataclasses.replace(
      _ORACLE_FRAME,
      session=_SESSION,
      viewer=contract.Viewer(contract.ViewerKind.SEAT, seat),
      omniscient=False,
      legal_actions=_ACTIONS if seat == 0 else None,
      chance_outcomes=None,
      history=_HIDDEN_DEAL,
      rewards=None,
      state_text=None,
      observation_text=f"card {2 - 2 * seat}",
      struct=None,
  )


_SPECTATOR_FRAME = dataclasses.replace(
    _seat_frame(1),
    viewer=contract.Viewer(contract.ViewerKind.SPECTATOR),
    observation_text=None,
)


class AudienceTest(absltest.TestCase):

  def test_each_viewer_has_a_valid_frame(self) -> None:
    for built in (_ORACLE_FRAME, _seat_frame(0), _seat_frame(1)):
      self.assertTrue(built.live)
    self.assertIsNone(_SPECTATOR_FRAME.legal_actions)

  def test_the_oracle_is_omniscient(self) -> None:
    with self.assertRaisesRegex(ValueError, "oracle is omniscient"):
      dataclasses.replace(_seat_frame(0), viewer=_ORACLE_FRAME.viewer)

  def test_omniscience_follows_the_game(self) -> None:
    # A seat of a game that hides information is not omniscient, even when
    # the frame says so and holds nothing else it may not see.
    with self.assertRaisesRegex(ValueError, "omniscient exactly when"):
      dataclasses.replace(
          _ORACLE_FRAME,
          session=_SESSION,
          viewer=contract.Viewer(contract.ViewerKind.SEAT, 1),
      )
    with self.assertRaisesRegex(ValueError, "omniscient exactly when"):
      dataclasses.replace(
          _ORACLE_FRAME,
          session=_SESSION,
          viewer=contract.Viewer(contract.ViewerKind.SPECTATOR),
      )
    # Every viewer of a game that hides nothing is omniscient.
    open_session = dataclasses.replace(_SESSION, hides_information=False)
    with self.assertRaisesRegex(ValueError, "omniscient exactly when"):
      dataclasses.replace(_seat_frame(0), session=open_session)
    dataclasses.replace(
        _ORACLE_FRAME,
        session=open_session,
        viewer=contract.Viewer(contract.ViewerKind.SPECTATOR),
        struct=None,
    )

  def test_state_text_reaches_exactly_omniscient_viewers(self) -> None:
    with self.assertRaisesRegex(ValueError, "state text"):
      dataclasses.replace(_seat_frame(0), state_text="2 0")
    with self.assertRaisesRegex(ValueError, "state text"):
      dataclasses.replace(_ORACLE_FRAME, state_text=None)

  def test_the_struct_reaches_only_omniscient_viewers(self) -> None:
    with self.assertRaisesRegex(ValueError, "struct"):
      dataclasses.replace(_seat_frame(0), struct={"cards": [2, 0]})
    dataclasses.replace(_ORACLE_FRAME, struct=None)

  def test_the_seed_reaches_only_the_oracle(self) -> None:
    with self.assertRaisesRegex(ValueError, "seed"):
      dataclasses.replace(_seat_frame(0), session=_ORACLE_FRAME.session)

  def test_per_seat_texts_reach_only_a_seat(self) -> None:
    with self.assertRaisesRegex(ValueError, "Per-seat"):
      dataclasses.replace(_ORACLE_FRAME, observation_text="card 2")
    with self.assertRaisesRegex(ValueError, "Per-seat"):
      dataclasses.replace(_SPECTATOR_FRAME, information_state_text="2")

  def test_nothing_is_withheld_from_an_omniscient_viewer(self) -> None:
    with self.assertRaisesRegex(ValueError, "omniscient viewer"):
      dataclasses.replace(_ORACLE_FRAME, legal_actions=None)
    with self.assertRaisesRegex(ValueError, "omniscient viewer"):
      dataclasses.replace(_ORACLE_FRAME, history=_HIDDEN_DEAL, rewards=None)

  def test_legal_actions_reach_only_the_acting_seat(self) -> None:
    with self.assertRaisesRegex(ValueError, "acting seat"):
      dataclasses.replace(_seat_frame(1), legal_actions=_ACTIONS)
    with self.assertRaisesRegex(ValueError, "acting seat"):
      dataclasses.replace(_SPECTATOR_FRAME, legal_actions=_ACTIONS)

  def test_a_seat_sees_its_own_actions(self) -> None:
    with self.assertRaisesRegex(ValueError, "its own actions"):
      dataclasses.replace(
          _seat_frame(0),
          ply=3,
          num_plies=3,
          history=_HIDDEN_DEAL + (contract.HistoryEntry(0, None, None, None),),
          to_move=contract.ToMove(contract.MoverKind.SEAT, 1),
          legal_actions=None,
      )

  def test_chance_outcomes_reach_only_an_omniscient_viewer(self) -> None:
    with self.assertRaisesRegex(ValueError, "Chance outcomes"):
      dataclasses.replace(
          _seat_frame(0),
          session=_EXPOSED,
          to_move=contract.ToMove(contract.MoverKind.CHANCE),
          legal_actions=(),
          chance_outcomes=(contract.ChanceOutcome(1, "Deal:1", 1.0),),
      )

  def test_failure_messages_reach_only_omniscient_viewers(self) -> None:
    failed = dataclasses.replace(
        _ORACLE_FRAME, status=contract.SessionStatus.PAUSED, seats=_FAILED_SEATS
    )
    with self.assertRaisesRegex(ValueError, "Failure messages"):
      dataclasses.replace(
          _seat_frame(0),
          status=contract.SessionStatus.PAUSED,
          seats=failed.seats,
      )


class PositionTest(absltest.TestCase):

  def test_chance_outcomes_are_served_only_under_expose(self) -> None:
    chance = dataclasses.replace(
        _ORACLE_FRAME,
        session=dataclasses.replace(_EXPOSED, seed=7),
        to_move=contract.ToMove(contract.MoverKind.CHANCE),
        legal_actions=(),
        chance_outcomes=(contract.ChanceOutcome(1, "Deal:1", 1.0),),
    )
    with self.assertRaisesRegex(ValueError, "expose policy"):
      dataclasses.replace(chance, session=_ORACLE_FRAME.session)

  def test_a_replay_offers_actions_only_to_an_omniscient_viewer(self) -> None:
    replay = dataclasses.replace(
        _ORACLE_FRAME, ply=1, legal_actions=(), struct=None
    )
    self.assertFalse(replay.live)
    self.assertEqual(
        dataclasses.replace(replay, legal_actions=_ACTIONS).legal_actions,
        _ACTIONS,
    )
    seat_replay = dataclasses.replace(_seat_frame(0), ply=1, legal_actions=())
    with self.assertRaisesRegex(ValueError, "omniscient viewer"):
      dataclasses.replace(seat_replay, legal_actions=_ACTIONS)

  def test_ply_lies_within_the_history(self) -> None:
    for ply in (-1, 3):
      with self.subTest(ply=ply):
        with self.assertRaisesRegex(ValueError, "history entries"):
          dataclasses.replace(_ORACLE_FRAME, ply=ply)

  def test_the_history_has_one_entry_per_ply(self) -> None:
    with self.assertRaisesRegex(ValueError, "history entries"):
      dataclasses.replace(_ORACLE_FRAME, ply=3, num_plies=3)

  def test_rewards_are_those_of_the_action_that_reached_the_position(
      self,
  ) -> None:
    dataclasses.replace(_ORACLE_FRAME, ply=0, legal_actions=(), rewards=None)
    with self.assertRaisesRegex(ValueError, "reached it"):
      dataclasses.replace(_ORACLE_FRAME, rewards=(1.0, -1.0))
    with self.assertRaisesRegex(ValueError, "every seat"):
      dataclasses.replace(
          _ORACLE_FRAME,
          history=(_DEAL[0], contract.HistoryEntry(None, 0, "Deal:0", (0.0,))),
          rewards=(0.0,),
      )

  def test_returns_are_served_exactly_at_a_terminal_position(self) -> None:
    with self.assertRaisesRegex(ValueError, "Returns"):
      dataclasses.replace(_ORACLE_FRAME, returns=(1.0, -1.0))
    with self.assertRaisesRegex(ValueError, "Returns"):
      dataclasses.replace(
          _ORACLE_FRAME,
          status=contract.SessionStatus.TERMINAL,
          to_move=contract.ToMove(contract.MoverKind.TERMINAL),
          legal_actions=(),
      )

  def test_the_live_position_agrees_with_the_session_status(self) -> None:
    with self.assertRaisesRegex(ValueError, "terminal exactly"):
      dataclasses.replace(_ORACLE_FRAME, status=contract.SessionStatus.TERMINAL)

  def test_a_replay_of_a_finished_game_need_not_be_terminal(self) -> None:
    dataclasses.replace(
        _ORACLE_FRAME,
        ply=1,
        legal_actions=(),
        status=contract.SessionStatus.TERMINAL,
    )

  def test_the_contract_version_is_current(self) -> None:
    stale: Any = 1  # Not a `contract.ContractVersion`.
    with self.assertRaisesRegex(ValueError, "Contract version"):
      dataclasses.replace(_ORACLE_FRAME, contract_version=stale)


class SeatsTest(absltest.TestCase):

  def test_a_session_is_paused_exactly_when_a_controller_failed(self) -> None:
    dataclasses.replace(
        _ORACLE_FRAME, status=contract.SessionStatus.PAUSED, seats=_FAILED_SEATS
    )
    with self.assertRaisesRegex(ValueError, "paused"):
      dataclasses.replace(_ORACLE_FRAME, status=contract.SessionStatus.PAUSED)
    with self.assertRaisesRegex(ValueError, "paused"):
      dataclasses.replace(_ORACLE_FRAME, seats=_FAILED_SEATS)

  def test_seats_are_listed_once_in_order(self) -> None:
    with self.assertRaisesRegex(ValueError, "seat order"):
      dataclasses.replace(_ORACLE_FRAME, seats=_SEATS[::-1])

  def test_seats_are_played_by_offered_controllers(self) -> None:
    alphazero = contract.SeatView(1, "alphazero", {}, False, None)
    with self.assertRaisesRegex(ValueError, "does not offer"):
      dataclasses.replace(_ORACLE_FRAME, seats=(_SEATS[0], alphazero))


if __name__ == "__main__":
  absltest.main()
