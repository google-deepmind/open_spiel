# Copyright 2026 Georgios Tritsaris
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

"""Tests for the Python HEX0N game."""

import pickle

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python.games import hex0n
import pyspiel


def _action(kind, row, col, size=5):
  return kind * size * size + row * size + col


class Hex0nTest(parameterized.TestCase):

  def test_can_create_game_and_state(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(game.num_distinct_actions(), 101)
    # On an empty board only squares can be placed.
    self.assertEqual(state.legal_actions(), list(range(25)))

  @parameterized.parameters(
      {"size": 4},
      {"size": 5},
      {"size": 5, "opponent_pieces": False},
      {"size": 5, "tie_break": False},
      {"size": 7},
  )
  def test_random_sim(self, **params):
    game = pyspiel.load_game("python_hex0n", params)
    pyspiel.random_sim_test(game, num_sims=10, serialize=True, verbose=False)

  def test_triad_without_chance(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    state.apply_action(_action(hex0n.SQUARE, 0, 0))  # player 0
    state.apply_action(_action(hex0n.CIRCLE, 0, 1))  # player 1
    # Player 0's triangle at (1,0) touches one eligible pair: no chance node.
    state.apply_action(_action(hex0n.TRIANGLE, 1, 0))
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(state.legal_actions(),
                     [_action(hex0n.HEX, 0, 0), _action(hex0n.HEX, 0, 1),
                      _action(hex0n.HEX, 1, 0)])
    state.apply_action(_action(hex0n.HEX, 1, 0))
    self.assertEqual(state.current_player(), 1)
    self.assertEqual(state.hexes, [1, 0])
    self.assertEqual(state.board[5], (hex0n.HEX, 0))
    self.assertIsNone(state.board[0])
    self.assertIsNone(state.board[1])

  def test_triad_needs_own_pieces_without_opponent_pieces(self):
    game = pyspiel.load_game("python_hex0n", {"opponent_pieces": False})
    state = game.new_initial_state()
    state.apply_action(_action(hex0n.SQUARE, 0, 0))
    state.apply_action(_action(hex0n.CIRCLE, 0, 1))  # player 1's circle
    self.assertNotIn(_action(hex0n.TRIANGLE, 1, 0), state.legal_actions())

  def test_sixteen_chance_outcomes(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    # Ring the centre cell (2,2) with 4 circles and 4 squares of player 0.
    for cell in (6, 8, 16, 18):
      state.board[cell] = (hex0n.CIRCLE, 0)
    for cell in (7, 11, 13, 17):
      state.board[cell] = (hex0n.SQUARE, 0)
    state.apply_action(_action(hex0n.TRIANGLE, 2, 2))
    self.assertTrue(state.is_chance_node())
    outcomes = state.chance_outcomes()
    self.assertLen(outcomes, game.max_chance_outcomes())
    self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)
    # Outcome 0 removes the first pair: circle (1,1) with square (1,2).
    state.apply_action(0)
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(state.legal_actions(),
                     [_action(hex0n.HEX, 1, 1), _action(hex0n.HEX, 1, 2),
                      _action(hex0n.HEX, 2, 2)])

  def test_pass_only_without_placements(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    state.squares_left = [0, 8]  # player 0 has no squares, and nothing else fits
    self.assertEqual(state.legal_actions(), [4 * 25])
    state.apply_action(4 * 25)
    self.assertEqual(state.current_player(), 1)
    self.assertFalse(state.is_terminal())

  @parameterized.parameters(
      {"tie_break": True, "returns": [1.0, -1.0]},
      {"tie_break": False, "returns": [0.0, 0.0]},
  )
  def test_equal_hexes(self, tie_break, returns):
    game = pyspiel.load_game("python_hex0n", {"tie_break": tie_break})
    state = game.new_initial_state()
    # Three hexes each: player 0's form one group, player 1's are apart.
    for col in (0, 1, 2):
      state.board[col] = (hex0n.HEX, 0)
    for col in (0, 2, 4):
      state.board[20 + col] = (hex0n.HEX, 1)
    state.hexes = [3, 3]
    # No squares anywhere and none left: nobody can place, so a pass ends it.
    state.squares_left = [0, 0]
    self.assertEqual(state.legal_actions(), [4 * 25])
    state.apply_action(4 * 25)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), returns)

  def test_observation_tensor(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    state.apply_action(_action(hex0n.SQUARE, 2, 2))  # player 0
    obs = np.reshape(state.observation_tensor(1), (11, 5, 5))
    self.assertEqual(obs[4, 2, 2], 1)  # player 1 sees the opponent's square
    self.assertEqual(obs[0].sum(), 0)
    self.assertEqual(obs[8].sum(), 24)
    self.assertAlmostEqual(obs[10, 0, 0], 1 / 8)  # 8 left vs 7 left

  def test_pickle(self):
    game = pyspiel.load_game("python_hex0n", {"size": 7})
    unpickled_game = pickle.loads(pickle.dumps(game))
    self.assertEqual(str(game), str(unpickled_game))
    state = game.new_initial_state()
    for action in (24, 49 + 25):  # square (3,3), circle (3,4)
      state.apply_action(action)
    unpickled_state = pickle.loads(pickle.dumps(state))
    self.assertEqual(str(state), str(unpickled_state))
    self.assertEqual(state.history(), unpickled_state.history())

  def test_clone_is_independent(self):
    game = pyspiel.load_game("python_hex0n")
    state = game.new_initial_state()
    state.apply_action(_action(hex0n.SQUARE, 2, 2))
    clone = state.clone()
    clone.apply_action(_action(hex0n.CIRCLE, 1, 1))
    self.assertIsNone(state.board[6])
    self.assertEqual(clone.board[6], (hex0n.CIRCLE, 1))
    self.assertEqual(state.squares_left, [7, 8])


if __name__ == "__main__":
  absltest.main()
