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


"""Tests for applying, recording and rebuilding plies."""

import dataclasses
import random
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import plies
from open_spiel.python.gui import presenters
import pyspiel


class ToMoveTest(absltest.TestCase):

  def test_names_the_seat_chance_or_the_end(self) -> None:
    kuhn = pyspiel.load_game("kuhn_poker")
    state = kuhn.new_initial_state()
    self.assertEqual(plies.to_move(state).kind, contract.MoverKind.CHANCE)
    state.apply_action(0)
    state.apply_action(1)
    self.assertEqual(
        plies.to_move(state), contract.ToMove(contract.MoverKind.SEAT, 0)
    )
    state.apply_action(0)
    state.apply_action(0)
    self.assertEqual(plies.to_move(state).kind, contract.MoverKind.TERMINAL)

  def test_lists_possible_chance_outcomes_as_legal(self) -> None:
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    self.assertEqual(plies.legal_actions(state), (0, 1, 2))
    state.apply_action(1)
    self.assertEqual(plies.legal_actions(state), (0, 2))


class ApplyTest(absltest.TestCase):

  def test_records_the_seat_label_and_rewards(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    ply = plies.apply(state, 4, ply=0)
    self.assertEqual(ply, plies.Ply(0, 4, "x(1,1)", (0.0, 0.0)))
    self.assertEqual(state.history(), [4])

  def test_records_the_adapters_public_label_from_the_state_before(
      self,
  ) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    grid = presenters.adapter_for("tic_tac_toe")
    seen = []

    def public(before: pyspiel.State, action: int) -> str | None:
      seen.append((str(before), action))
      return f"seat {before.current_player()} took {action}"

    with mock.patch.object(grid, "public_action_label", public):
      ply = plies.apply(state, 4, ply=0)
    self.assertEqual(ply.public_label, "seat 0 took 4")
    self.assertEqual(seen, [(str(game.new_initial_state()), 4)])
    self.assertIsNone(plies.apply(state, 0, ply=1).public_label)

  def test_records_a_chance_outcome_without_a_seat(self) -> None:
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    self.assertIsNone(plies.apply(state, 2, ply=0).seat)

  def test_refuses_an_illegal_action_and_leaves_the_state(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    plies.apply(state, 4, ply=0)
    with self.assertRaises(plies.IllegalPlyError) as raised:
      plies.apply(state, 4, ply=1)
    self.assertEqual((raised.exception.ply, raised.exception.seat), (1, 1))
    self.assertEqual(state.history(), [4])

  def test_samples_chance_reproducibly(self) -> None:
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    self.assertEqual(
        plies.sample_chance(state, random.Random(3)),
        plies.sample_chance(state, random.Random(3)),
    )


class RebuildTest(absltest.TestCase):

  def test_reproduces_the_position_and_its_plies(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    state, recorded = plies.rebuild(game, [4, 0, 8])
    self.assertEqual(state.history(), [4, 0, 8])
    self.assertEqual([ply.seat for ply in recorded], [0, 1, 0])

  def test_checks_the_rebuilt_plies_against_their_record(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    _, recorded = plies.rebuild(game, [4, 0])
    plies.rebuild(game, [4, 0], expected=recorded)
    tampered = (recorded[0], dataclasses.replace(recorded[1], label="o(2,2)"))
    with self.assertRaises(plies.DriftError):
      plies.rebuild(game, [4, 0], expected=tampered)

  def test_reports_an_illegal_history_with_its_index(self) -> None:
    with self.assertRaises(plies.IllegalPlyError) as raised:
      plies.rebuild(pyspiel.load_game("tic_tac_toe"), [4, 4])
    self.assertEqual(raised.exception.ply, 1)

  def test_an_illegal_recorded_history_is_drift(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    _, recorded = plies.rebuild(game, [4])
    forged = (dataclasses.replace(recorded[0], action=9),)
    with self.assertRaises(plies.DriftError):
      plies.rebuild(game, [9], expected=forged)


if __name__ == "__main__":
  absltest.main()
