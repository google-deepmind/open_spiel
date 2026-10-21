# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for chaupar.py."""

import random

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.games import chaupar
import pyspiel


class ChauparShellScoreTest(absltest.TestCase):

  def test_default_seven_shell_table_matches_verified_source(self):
    # Verified against Wikipedia's raw article text for Chaupar; see the
    # module docstring for the citation.
    expected = {0: 7, 1: 10, 2: 2, 3: 3, 4: 4, 5: 25, 6: 30, 7: 14}
    self.assertEqual(chaupar._shell_scores(7), expected)

  def test_high_throws_are_exactly_the_documented_set(self):
    game = pyspiel.load_game("python_chaupar")
    self.assertEqual(game._high_throws, frozenset({10, 25, 30}))

  def test_value_one_is_never_achievable(self):
    """The overshoot-clamping fix (see chaupar.py) exists specifically
    because this is true: no shell count ever scores exactly 1 point."""
    for num_shells in (5, 6, 7, 8, 9):
      scores = chaupar._shell_scores(num_shells)
      self.assertNotIn(1, scores.values())

  def test_true_extreme_throw_is_not_high_for_non_default_shell_counts(self):
    """Regression test: the "all shells up" throw (k == num_shells) must
    not be treated as "high", mirroring the verified 7-shell table, where
    k=7 scores 14 -- notably excluded from the high set {10, 25, 30}."""
    for num_shells in (5, 6, 8, 9):
      game = pyspiel.load_game("python_chaupar", {"num_shells": num_shells})
      scores = game._shell_scores
      self.assertNotIn(scores[0], game._high_throws)
      self.assertNotIn(scores[num_shells], game._high_throws)


class ChauparTest(parameterized.TestCase):

  def test_load_default(self):
    game = pyspiel.load_game("python_chaupar")
    self.assertEqual(game.num_players(), 4)

  @parameterized.parameters(2, 3, 4)
  def test_variable_num_players(self, num_players):
    game = pyspiel.load_game("python_chaupar", {"players": num_players})
    self.assertEqual(game.num_players(), num_players)

  def test_players_out_of_range_raises(self):
    with self.assertRaises(ValueError):
      pyspiel.load_game("python_chaupar", {"players": 1})
    with self.assertRaises(ValueError):
      pyspiel.load_game("python_chaupar", {"players": 5})

  def test_initial_chance_node_matches_binomial_distribution(self):
    game = pyspiel.load_game("python_chaupar")
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    outcomes = dict(state.chance_outcomes())
    self.assertAlmostEqual(sum(outcomes.values()), 1.0)
    # Symmetric binomial(7, 0.5): P(0 up) == P(7 up), etc.
    self.assertAlmostEqual(outcomes[0], outcomes[7])
    self.assertAlmostEqual(outcomes[1], outcomes[6])
    self.assertAlmostEqual(outcomes[3], outcomes[4])

  def test_only_high_throws_allow_entering(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    # 2 shells up -> value 2, not a high throw: no piece can enter.
    state.apply_action(2)
    self.assertTrue(state.is_terminal() is False)
    # Since no piece has entered and 2 is not high, the turn should have
    # auto-passed with no decision node; we should be back at a chance node.
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)

  def test_high_throw_lets_player_enter_a_piece(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    state.apply_action(1)  # 1 shell up -> value 10 (high).
    self.assertEqual(state.current_player(), 0)
    legal = state.legal_actions()
    self.assertNotEmpty(legal)
    state.apply_action(legal[0])
    self.assertEqual(state._positions[0][legal[0]], 0)

  def test_high_throw_grants_extra_turn(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    state.apply_action(1)  # value 10, high.
    state.apply_action(state.legal_actions()[0])  # Enter a piece.
    # Player 0 should still be current (extra turn), now awaiting a new roll.
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertEqual(state._current_player, 0)

  def test_three_consecutive_high_throws_burns_the_turn(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    state.apply_action(1)  # High (value 10): enter a piece, extra turn.
    state.apply_action(state.legal_actions()[0])
    state.apply_action(1)  # High again (2nd in a row): extra turn.
    state.apply_action(state.legal_actions()[0])
    self.assertEqual(state._current_player, 0)
    state.apply_action(1)  # High a 3rd time in a row: burned.
    # Should now be player 1's turn -- no decision node was offered for P0.
    self.assertEqual(state._current_player, 1)

  def test_capture_sends_piece_back_to_start(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    # Enter one piece each for both players onto the shared track, then
    # engineer a capture by directly placing them on a colliding, non-safe
    # absolute square (bypassing the chance mechanic for precise control).
    state._positions[0][0] = 5  # Player 0's arm offset is 0 -> absolute 5.
    # Player 1's arm offset is 34; pick their relative position so their
    # absolute position also lands on 5.
    state._positions[1][0] = (
        5 - state._arm_offset[1]) % chaupar._SHARED_TRACK_LENGTH
    self.assertNotIn(5, chaupar._SAFE_SQUARES)
    self.assertEqual(state._abs_position(0, state._positions[0][0]), 5)
    self.assertEqual(state._abs_position(1, state._positions[1][0]), 5)

    # Move player 0's other piece onto the same square via the public API,
    # by fast-forwarding to a state where it's player 0's turn with a roll
    # that lands exactly there. Simpler: directly invoke the private move
    # applier used by the engine, mirroring what apply_action(piece) does.
    state._current_player = 0
    state._current_roll = 5
    state._awaiting_chance = False
    state._positions[0][1] = 0  # A second P0 piece already on the track.
    state.apply_action(1)  # Move piece 1 by 5 -> lands on absolute square 5.
    self.assertEqual(state._positions[1][0], chaupar._NOT_STARTED)
    self.assertTrue(state._has_captured[0])

  def test_safe_square_blocks_capture(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    safe_square = next(iter(chaupar._SAFE_SQUARES))
    state._positions[1][0] = (
        safe_square - state._arm_offset[1]) % chaupar._SHARED_TRACK_LENGTH
    state._positions[0][0] = 0
    state._current_player = 0
    state._current_roll = safe_square - state._arm_offset[0]
    state._awaiting_chance = False
    state.apply_action(0)
    # Landing on a safe square must not capture the opponent's piece there.
    self.assertNotEqual(state._positions[1][0], chaupar._NOT_STARTED)
    self.assertFalse(state._has_captured[0])

  def test_two_stacked_opponent_pieces_form_an_uncapturable_block(self):
    """Two of the same opponent's pieces stacked on one non-safe square
    form a "block": landing there does not capture either piece."""
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    target_square = 5
    self.assertNotIn(target_square, chaupar._SAFE_SQUARES)
    rel = (target_square - state._arm_offset[1]) % chaupar._SHARED_TRACK_LENGTH
    state._positions[1][0] = rel
    state._positions[1][1] = rel  # A second P1 piece stacked on the same
    state._positions[0][0] = 0    # square as the first.
    state._current_player = 0
    state._current_roll = target_square - state._arm_offset[0]
    state._awaiting_chance = False
    state.apply_action(0)
    self.assertEqual(state._positions[1][0], rel)
    self.assertEqual(state._positions[1][1], rel)
    self.assertFalse(state._has_captured[0])

  def test_cannot_enter_home_column_before_capturing(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    state._positions[0][0] = 66  # 2 away from the shared-track/home border.
    state._current_player = 0
    state._awaiting_chance = False
    state._current_roll = 3  # Would cross into the home column (pos 69).
    self.assertFalse(state._has_captured[0])
    self.assertEmpty(state._legal_piece_moves(0, 3))

  def test_capture_gate_bypassed_only_when_it_is_the_only_option(self):
    """Regression test for a real deadlock: if every one of a player's
    pieces is backed up at the end of the shared track with no capture ever
    made, the game must not lock up forever."""
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    for i in range(4):
      state._positions[0][i] = 66
    self.assertFalse(state._has_captured[0])
    # Gated: truly no legal move (would need to enter home uncaptured).
    self.assertEmpty(state._legal_piece_moves(0, 3))
    # With the fallback, the player is not stuck forever.
    self.assertNotEmpty(state._legal_piece_moves_with_fallback(0, 3))

  def test_win_condition_and_zero_sum_returns(self):
    game = pyspiel.load_game("python_chaupar", {"players": 2})
    state = game.new_initial_state()
    state._positions[0] = [chaupar._FINISHED] * 3 + [70]
    state._has_captured[0] = True
    state._current_player = 0
    state._awaiting_chance = False
    state._current_roll = 10  # Overshoots -> clamps to FINISHED.
    state.apply_action(3)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

  def test_random_games_terminate_and_are_zero_sum(self):
    """Regression test for the two deadlocks found & fixed during
    development: the impossible-throw-of-1 overshoot bug, and the
    all-pieces-stuck-uncaptured capture-gate deadlock."""
    for num_players in (2, 3, 4):
      game = pyspiel.load_game("python_chaupar", {"players": num_players})
      for trial in range(20):
        rng = random.Random(trial + num_players * 1000)
        state = game.new_initial_state()
        steps = 0
        while not state.is_terminal() and steps < 5000:
          if state.current_player() == pyspiel.PlayerId.CHANCE:
            actions, probs = zip(*state.chance_outcomes())
            action = rng.choices(actions, weights=probs)[0]
          else:
            action = rng.choice(state.legal_actions())
          state.apply_action(action)
          steps += 1
        self.assertTrue(state.is_terminal(),
                         f"{num_players}p trial {trial} did not terminate")
        self.assertAlmostEqual(sum(state.returns()), 0.0)

  @parameterized.parameters(2, 3, 4)
  def test_random_sim(self, num_players):
    game = pyspiel.load_game("python_chaupar", {"players": num_players})
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)

  @parameterized.parameters(5, 6, 7, 8)
  def test_random_sim_num_shells(self, num_shells):
    game = pyspiel.load_game("python_chaupar", {"num_shells": num_shells})
    pyspiel.random_sim_test(game, num_sims=3, serialize=True, verbose=False)


if __name__ == "__main__":
  absltest.main()
