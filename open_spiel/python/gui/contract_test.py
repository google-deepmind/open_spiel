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

"""Tests for the session parts of the wire contract and their serialisation."""

from absl.testing import absltest

from open_spiel.python.gui import contract


class ViewerTest(absltest.TestCase):

  def test_a_seat_viewer_names_its_seat(self) -> None:
    with self.assertRaisesRegex(ValueError, "kind seat"):
      contract.Viewer(contract.ViewerKind.SEAT)

  def test_only_a_seat_viewer_names_a_seat(self) -> None:
    with self.assertRaisesRegex(ValueError, "kind oracle"):
      contract.Viewer(contract.ViewerKind.ORACLE, 0)


class ToMoveTest(absltest.TestCase):

  def test_a_seat_mover_names_its_seat(self) -> None:
    with self.assertRaisesRegex(ValueError, "kind seat"):
      contract.ToMove(contract.MoverKind.SEAT)

  def test_chance_and_terminal_name_no_seat(self) -> None:
    for kind in (contract.MoverKind.CHANCE, contract.MoverKind.TERMINAL):
      with self.subTest(kind=kind), self.assertRaises(ValueError):
        contract.ToMove(kind, 0)


class SeatViewTest(absltest.TestCase):

  def test_a_human_seat_never_autoplays(self) -> None:
    contract.SeatView(0, "mcts", {}, True, None)
    with self.assertRaisesRegex(ValueError, "human"):
      contract.SeatView(0, contract.HUMAN, {}, True, None)

  def test_a_seat_is_ready_unless_its_controller_is_being_built(self) -> None:
    self.assertEqual(
        contract.SeatView(0, "mcts", {}, True, None).status,
        contract.SeatStatus.READY,
    )
    preparing = contract.SeatStatus.PREPARING
    contract.SeatView(0, "mcts", {}, True, None, preparing, None)
    contract.SeatView(0, "mcts", {}, True, None, preparing, "Building MCTS.")
    with self.assertRaisesRegex(ValueError, "always ready"):
      contract.SeatView(0, contract.HUMAN, {}, False, None, preparing, None)
    with self.assertRaisesRegex(ValueError, "no message"):
      contract.SeatView(0, "mcts", {}, True, None, message="Ready.")

  def test_a_seat_fails_only_once_built_or_when_it_cannot_be(self) -> None:
    failure = contract.ControllerFailure("UnavailableError", None)
    unavailable = contract.SeatStatus.UNAVAILABLE
    contract.SeatView(0, "mcts", {}, True, failure, unavailable, "No model.")
    with self.assertRaisesRegex(ValueError, "preparing"):
      contract.SeatView(
          0, "mcts", {}, True, failure, contract.SeatStatus.PREPARING, None
      )
    with self.assertRaisesRegex(ValueError, "a failure and a message"):
      contract.SeatView(0, "mcts", {}, True, failure, unavailable, None)
    with self.assertRaisesRegex(ValueError, "a failure and a message"):
      contract.SeatView(0, "mcts", {}, True, None, unavailable, "No model.")


class ChanceOutcomeTest(absltest.TestCase):

  def test_probability_lies_in_the_unit_interval(self) -> None:
    contract.ChanceOutcome(0, "Heads", 1.0)
    for probability in (0.0, 1.5):
      with self.subTest(probability=probability), self.assertRaises(ValueError):
        contract.ChanceOutcome(0, "Heads", probability)


class HistoryEntryTest(absltest.TestCase):

  def test_action_label_and_rewards_are_withheld_together(self) -> None:
    self.assertFalse(contract.HistoryEntry(0, 3, "Bet", (0.0, 0.0)).withheld)
    self.assertTrue(contract.HistoryEntry(0, None, None, None).withheld)
    with self.assertRaisesRegex(ValueError, "together"):
      contract.HistoryEntry(0, 3, None, None)
    with self.assertRaisesRegex(ValueError, "together"):
      contract.HistoryEntry(0, 3, "Bet", None)

  def test_only_a_withheld_entry_has_a_public_label(self) -> None:
    entry = contract.HistoryEntry(0, None, None, None, "Discard Jd")
    self.assertTrue(entry.withheld)
    self.assertEqual(entry.public_label, "Discard Jd")
    self.assertIsNone(contract.HistoryEntry(0, None, None, None).public_label)
    with self.assertRaisesRegex(ValueError, "Only a withheld"):
      contract.HistoryEntry(0, 3, "Bet", (0.0, 0.0), "Bet")


class SessionInfoTest(absltest.TestCase):

  def test_a_session_offers_the_human_controller(self) -> None:
    random = contract.Controller("random", "Random", {})
    with self.assertRaisesRegex(ValueError, "human"):
      contract.SessionInfo(
          "s",
          "go",
          {},
          2,
          contract.ChancePolicy.AUTO,
          False,
          False,
          None,
          (random,),
      )


class ToJsonTest(absltest.TestCase):

  def test_every_field_is_present_and_enums_are_values(self) -> None:
    self.assertEqual(
        contract.to_json(contract.Viewer(contract.ViewerKind.SPECTATOR)),
        {"kind": "spectator", "seat": None},
    )

  def test_nested_values_become_json(self) -> None:
    entry = contract.HistoryEntry(1, 4, "Bet", (-1.0, 1.0))
    self.assertEqual(
        contract.to_json(entry),
        {
            "seat": 1,
            "action": 4,
            "label": "Bet",
            "rewards": [-1.0, 1.0],
            "public_label": None,
        },
    )

  def test_values_outside_the_contract_are_refused(self) -> None:
    with self.assertRaises(TypeError):
      contract.to_json({"when": object()})


if __name__ == "__main__":
  absltest.main()
