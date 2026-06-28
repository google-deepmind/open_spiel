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

"""Tests for Python Boop."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import boop
from open_spiel.python.observation import make_observation
import pyspiel


class BoopTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.current_player(), 0)

  def test_initial_hand(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    self.assertEqual(state._hand[0], [8, 0])
    self.assertEqual(state._hand[1], [8, 0])

  def test_initial_board_empty(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    self.assertTrue(np.all(state.board == boop._EMPTY))

  def test_place_kitten_updates_board_and_hand(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    # Action 0 = place kitten at (0, 0)
    state.apply_action(0)
    self.assertEqual(state.board[0, 0], boop._P0_KITTEN)
    self.assertEqual(state._hand[0][0], 7)  # one kitten used
    self.assertEqual(state.current_player(), 1)  # turn passed

  def test_boop_pushes_kitten_away(self):
    """Placing at (2,1) should boop (2,2) to (2,3)."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    # Place P0 kitten at (2, 2)
    state.apply_action(2 * 6 + 2)  # action 14
    self.assertEqual(state.board[2, 2], boop._P0_KITTEN)
    # Place P1 kitten at (2, 1) -> should boop P0 kitten right to (2, 3)
    state.apply_action(2 * 6 + 1)  # action 13
    self.assertEqual(state.board[2, 2], boop._EMPTY)
    self.assertEqual(state.board[2, 3], boop._P0_KITTEN)

  def test_boop_off_board_returns_to_hand(self):
    """Kitten booped off the board edge returns to owner's hand."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    # P0 kitten at (0, 0)
    state.apply_action(0 * 6 + 0)
    self.assertEqual(state._hand[0][0], 7)
    # P1 kitten at (1, 1) -> diagonally boops P0 kitten at (0,0) off board
    state.apply_action(1 * 6 + 1)
    self.assertEqual(state.board[0, 0], boop._EMPTY)
    self.assertEqual(state._hand[0][0], 8)  # returned to P0 hand

  def test_kitten_cannot_push_cat(self):
    """A placed kitten should not boop an adjacent cat."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    # Manually place P1 cat at (2, 2)
    state.board[2, 2] = boop._P1_CAT
    state._hand[1][1] = 0
    # P0 places kitten at (2, 1); should not push P1 cat
    state.apply_action(2 * 6 + 1)
    self.assertEqual(state.board[2, 2], boop._P1_CAT)  # unmoved

  def test_cat_pushes_cat(self):
    """A placed cat should push an adjacent cat."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    # P0 has a cat, P1 cat already on board
    state.board[2, 2] = boop._P1_CAT
    state._hand[0][0] = 0  # no kittens
    state._hand[0][1] = 1  # one cat
    # Place P0 cat at (2, 1)
    state.apply_action(boop._NUM_CELLS + (2 * 6 + 1))
    # P1 cat at (2,2) should be pushed to (2,3)
    self.assertEqual(state.board[2, 2], boop._P0_CAT)
    self.assertEqual(state.board[2, 3], boop._P1_CAT)

  def test_promotion_three_kittens_in_row(self):
    """3 kittens in a row should be removed and player earns cats."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    state.board[2, 0] = boop._P0_KITTEN
    state.board[2, 1] = boop._P0_KITTEN
    state.board[2, 2] = boop._P0_KITTEN
    state._hand[0][0] = 5  # 3 used on board
    state._promote_kittens(0)
    self.assertEqual(state.board[2, 0], boop._EMPTY)
    self.assertEqual(state.board[2, 1], boop._EMPTY)
    self.assertEqual(state.board[2, 2], boop._EMPTY)
    self.assertEqual(state._hand[0][0], 8)   # 5 + 3 returned
    self.assertEqual(state._hand[0][1], 3)   # 3 cats earned

  def test_promotion_cat_cap(self):
    """Cat earnings are capped at 6 total (hand + board)."""
    game = boop.BoopGame()
    state = game.new_initial_state()
    state.board[2, 0] = boop._P0_KITTEN
    state.board[2, 1] = boop._P0_KITTEN
    state.board[2, 2] = boop._P0_KITTEN
    state._hand[0][0] = 5
    state._hand[0][1] = 4  # already have 4 cats
    # Cap is 6; on board: 0 cats. So can earn at most 6-0-4=2 more
    state._promote_kittens(0)
    self.assertEqual(state._hand[0][1], 6)  # capped at 6

  def test_win_three_cats_in_row(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    state.board[0, 0] = boop._P0_CAT
    state.board[0, 1] = boop._P0_CAT
    state._hand[0][0] = 0
    state._hand[0][1] = 1
    # Place P0 cat at (0, 2) -> 3 in a row -> P0 wins
    state.apply_action(boop._NUM_CELLS + (0 * 6 + 2))
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

  def test_draw_at_move_limit(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    state._move_count = boop._MAX_GAME_LENGTH - 1
    state.apply_action(0)  # one more move triggers draw
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [0.0, 0.0])
    self.assertIsNone(state._winner)

  def test_legal_actions_only_kittens_initially(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    legal = state.legal_actions()
    self.assertEqual(len(legal), 36)  # all cells, kitten only
    self.assertTrue(all(a < boop._NUM_CELLS for a in legal))

  def test_legal_actions_include_cats_when_available(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    state._hand[0][1] = 1  # give P0 a cat
    legal = state.legal_actions()
    # Should have both kitten and cat actions for each empty cell
    self.assertEqual(len(legal), 72)

  def test_legal_actions_respect_occupied_cells(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    state.board[0, 0] = boop._P0_KITTEN
    state._hand[0][0] = 7
    legal = state.legal_actions()
    self.assertEqual(len(legal), 35)  # 35 remaining empty cells
    self.assertNotIn(0, legal)

  def test_observation_tensor_shape(self):
    game = pyspiel.load_game('python_boop')
    state = game.new_initial_state()
    from open_spiel.python.observation import make_observation
    obs = make_observation(game)
    obs.set_from(state, 0)
    self.assertEqual(len(obs.tensor), 184)  # 5*36 + 4

  def test_api_random_sim(self):
    game = pyspiel.load_game('python_boop')
    pyspiel.random_sim_test(game, num_sims=5, serialize=False, verbose=False)

  def test_board_str(self):
    game = boop.BoopGame()
    state = game.new_initial_state()
    s = str(state)
    self.assertIn('......', s)  # empty board row
    self.assertIn('P0:', s)
    self.assertIn('P1:', s)


if __name__ == '__main__':
  absltest.main()
