# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OpenSpiel observation histories."""

from absl.testing import absltest

import pyspiel


def state_from_sequence(game, action_sequence):
  state = game.new_initial_state()
  for action in action_sequence:
    state.apply_action(action)
  return state


class ObservationHistoryTest(absltest.TestCase):

  def test_invalid_aoh_construction_raises_error(self):
    with self.assertRaises(RuntimeError):
      pyspiel.ActionObservationHistory(0, ["not tuple"])
    with self.assertRaises(RuntimeError):
      pyspiel.ActionObservationHistory(0, [("tuple", "too", "long")])
    with self.assertRaises(RuntimeError):
      pyspiel.ActionObservationHistory(0, [("not int", "obs")])
    with self.assertRaises(RuntimeError):
      pyspiel.ActionObservationHistory(0, [(1, 1)])  # obs not string

  def test_kuhn_rollout(self):
    game = pyspiel.load_game("kuhn_poker")

    # Test on this specific sequence.
    action_sequence = [2, 1, 0, 1, 1]

    root_state = game.new_initial_state()
    terminal = state_from_sequence(game, action_sequence)
    self.assertTrue(terminal.is_terminal())

    # Check prefixes, extensions and correspondences of both
    # public-observation histories / action-observation histories
    # while rolling out.
    seq_idx = 0

    def advance_and_test(state):
      nonlocal seq_idx
      action = action_sequence[seq_idx]

      parent_state = state.clone()
      state.apply_action(action)

      # Both PublicObservationHistory and ActionObservationHistory need to pass
      # the same correspondence / prefix / extension tests, for both
      # construction from state (and player for AOH) or just passing state (and
      # player) without history construction. So this tests them together.
      # These are passed as constructors (lefts) and targets (rights).
      constructors_with_targets = [
          # (p)layer, (s)tate
          (pyspiel.ActionObservationHistory, [
              lambda p, s: [pyspiel.ActionObservationHistory(p, s)],
              lambda p, s: [p, s]
          ]),
          # Since PublicObservationHistory does not need a player it is ignored.
          (lambda _, s: pyspiel.PublicObservationHistory(s), [
              lambda _, s: [pyspiel.PublicObservationHistory(s)],
              lambda _, s: [s]
          ])
      ]

      for (left, rights) in constructors_with_targets:
        for right in rights:
          for player in range(2):
            # Shortcuts for conciseness: the most important things
            # relevant for the tests have long names.
            # In other words, when reading the code just skip over
            # the short vars to gain understanding.
            l, r, p = left, right, player

            self.assertTrue(l(p, state).corresponds_to(*r(p, state)))
            self.assertFalse(l(p, parent_state).corresponds_to(*r(p, state)))
            if state.is_terminal():
              self.assertTrue(l(p, terminal).corresponds_to(*r(p, state)))
            else:
              self.assertFalse(l(p, terminal).corresponds_to(*r(p, state)))
            if state.is_initial_state():
              self.assertTrue(l(p, root_state).corresponds_to(*r(p, state)))
            else:
              self.assertFalse(l(p, root_state).corresponds_to(*r(p, state)))

            self.assertTrue(l(p, parent_state).is_prefix_of(*r(p, state)))
            self.assertFalse(l(p, state).is_prefix_of(*r(p, parent_state)))
            self.assertTrue(l(p, root_state).is_prefix_of(*r(p, state)))
            if state.is_terminal():
              self.assertTrue(l(p, terminal).is_prefix_of(*r(p, state)))
            else:
              self.assertFalse(l(p, terminal).is_prefix_of(*r(p, state)))

            self.assertFalse(l(p, parent_state).is_extension_of(*r(p, state)))
            self.assertTrue(l(p, state).is_extension_of(*r(p, parent_state)))
            self.assertTrue(l(p, terminal).is_extension_of(*r(p, state)))
            if state.is_initial_state():
              self.assertTrue(l(p, root_state).is_extension_of(*r(p, state)))
            else:
              self.assertFalse(l(p, root_state).is_extension_of(*r(p, state)))

      seq_idx += 1
      return state

    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory([pyspiel.PublicObservation.START_GAME
                                         ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(
            0, [(None, pyspiel.PrivateObservation.NOTHING)]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(
            1, [(None, pyspiel.PrivateObservation.NOTHING)]))

    advance_and_test(state)
    self.assertTrue(state.is_chance_node())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory(
            [pyspiel.PublicObservation.START_GAME, "Deal to player 0"]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(0, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "211")
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(1, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, pyspiel.PrivateObservation.NOTHING)
        ]))

    advance_and_test(state)
    self.assertTrue(state.is_player_node())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory([
            pyspiel.PublicObservation.START_GAME, "Deal to player 0",
            "Deal to player 1"
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(0, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "211"),
            (None, "211")
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(1, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "111")
        ]))

    advance_and_test(state)
    self.assertTrue(state.is_player_node())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory([
            pyspiel.PublicObservation.START_GAME, "Deal to player 0",
            "Deal to player 1", "Pass"
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(0, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "211"),
            (None, "211"),
            (0, "211")
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(1, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "111"),
            (None, "111")
        ]))

    advance_and_test(state)
    self.assertTrue(state.is_player_node())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory([
            pyspiel.PublicObservation.START_GAME, "Deal to player 0",
            "Deal to player 1", "Pass", "Bet"
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(0, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "211"),
            (None, "211"),
            (0, "211"),
            (None, "212")
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(1, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "111"),
            (None, "111"),
            (1, "112")
        ]))

    advance_and_test(state)
    self.assertTrue(state.is_terminal())
    self.assertEqual(
        pyspiel.PublicObservationHistory(state),
        pyspiel.PublicObservationHistory([
            pyspiel.PublicObservation.START_GAME, "Deal to player 0",
            "Deal to player 1", "Pass", "Bet", "Bet"
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(0, state),
        pyspiel.ActionObservationHistory(0, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "211"),
            (None, "211"),
            (0, "211"),
            (None, "212"),
            (1, "222")
        ]))
    self.assertEqual(
        pyspiel.ActionObservationHistory(1, state),
        pyspiel.ActionObservationHistory(1, [
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, pyspiel.PrivateObservation.NOTHING),
            (None, "111"),
            (None, "111"),
            (1, "112"),
            (None, "122")
        ]))


if __name__ == "__main__":
  absltest.main()
