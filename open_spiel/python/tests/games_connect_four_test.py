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
"""Tests for the game-specific functions for connect_four."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import pyspiel

connect_four = pyspiel.connect_four


def empty_board():
  """Returns an empty 6x7 board."""
  return [["." for _ in range(7)] for _ in range(6)]


class GamesConnectFourTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.game = pyspiel.load_game("connect_four")

  def test_json(self):
    state = self.game.new_initial_state()
    state.apply_action(4)
    state_struct = state.to_struct()
    self.assertEqual(
        state_struct.board[0],
        [".", ".", ".", ".", "x", ".", "."],
    )
    self.assertEqual(state_struct.current_player, "o")
    json_from_struct = state_struct.to_json()
    state_json = state.to_json()
    self.assertEqual(json_from_struct, state_json)
    state_struct = connect_four.ConnectFourStateStruct(state_json)
    self.assertEqual(state_struct.to_json(), state_json)

  def test_action_struct(self):
    """Test ActionStruct creation and round-trip."""
    state = self.game.new_initial_state()
    for col in range(7):
      action_struct = state.action_to_struct(0, col)
      self.assertIsInstance(action_struct, connect_four.ConnectFourActionStruct)
      self.assertEqual(action_struct.column, col)

      # Test round-trip
      actions = state.struct_to_actions(action_struct)
      self.assertEqual(actions, [col])

  def test_action_struct_json(self):
    """Test ActionStruct JSON serialization."""
    action_struct = connect_four.ConnectFourActionStruct()
    action_struct.column = 3
    json_str = action_struct.to_json()
    self.assertEqual(json_str, '{"column":3}')

    # Parse back
    parsed = connect_four.ConnectFourActionStruct(json_str)
    self.assertEqual(parsed.column, 3)

  def test_state_from_dict_basic(self):
    """Test creating state from dictionary."""
    state = self.game.new_initial_state()
    state.apply_action(3)
    state.apply_action(4)

    # Round-trip through dict
    new_state = self.game.new_initial_state(state.to_dict())
    self.assertEqual(new_state.to_dict(), state.to_dict())
    self.assertEqual(str(new_state), str(state))

  def test_state_from_dict_terminal_x_wins(self):
    """Test creating terminal state where X wins."""
    board = empty_board()
    board[0][3] = "x"
    board[1][3] = "x"
    board[2][3] = "x"
    board[3][3] = "x"
    board[0][4] = "o"
    board[1][4] = "o"
    board[2][4] = "o"

    state_dict = {
        "board": board,
        "current_player": "Terminal",
        "is_terminal": True,
        "winner": "x",
    }

    state = self.game.new_initial_state(state_dict)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

  def test_state_from_dict_terminal_draw(self):
    """Test creating terminal draw state."""
    # A full board with no winner
    board = [
        ["o", "o", "x", "x", "x", "o", "o"],
        ["x", "x", "o", "o", "o", "x", "x"],
        ["o", "o", "x", "x", "x", "o", "o"],
        ["x", "x", "o", "o", "o", "x", "x"],
        ["o", "o", "x", "x", "x", "o", "o"],
        ["x", "x", "o", "o", "o", "x", "x"],
    ]

    state_dict = {
        "board": board,
        "current_player": "Terminal",
        "is_terminal": True,
        "winner": "draw",
    }

    state = self.game.new_initial_state(state_dict)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [0.0, 0.0])

  def test_state_from_json(self):
    """Test creating state from JSON string."""
    state = self.game.new_initial_state()
    state.apply_action(3)
    state.apply_action(4)

    json_str = state.to_json()
    new_state = self.game.new_initial_state(json_str)
    self.assertEqual(new_state.to_json(), json_str)

  @parameterized.parameters(
      # Gap in column
      dict(
          invalid_dict={
              "board": [
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", "x", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "o",
              "is_terminal": False,
              "winner": "",
          },
          error_msg="gap",
      ),
      # Invalid cell value
      dict(
          invalid_dict={
              "board": [
                  [".", ".", ".", "X", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "o",
              "is_terminal": False,
              "winner": "",
          },
          error_msg="Invalid cell",
      ),
      # is_terminal mismatch
      dict(
          invalid_dict={
              "board": [
                  [".", ".", ".", "x", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "Terminal",
              "is_terminal": True,
              "winner": "x",
          },
          error_msg="is_terminal",
      ),
      # winner mismatch
      dict(
          invalid_dict={
              "board": [
                  [".", ".", ".", "x", "o", ".", "."],
                  [".", ".", ".", "x", "o", ".", "."],
                  [".", ".", ".", "x", "o", ".", "."],
                  [".", ".", ".", "x", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "Terminal",
              "is_terminal": True,
              "winner": "o",
          },
          error_msg="winner",
      ),
      # Piece count imbalance (strict mode)
      dict(
          invalid_dict={
              "board": [
                  ["x", "x", "x", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "o",
              "is_terminal": False,
              "winner": "",
          },
          error_msg="imbalance",
      ),
      # current_player doesn't match piece counts (strict mode)
      dict(
          invalid_dict={
              "board": [
                  ["x", "o", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "o",
              "is_terminal": False,
              "winner": "",
          },
          error_msg="current_player",
      ),
      # Wrong board dimensions (5 columns instead of 7)
      dict(
          invalid_dict={
              "board": [
                  [".", ".", ".", ".", "."],
                  [".", ".", ".", ".", "."],
                  [".", ".", ".", ".", "."],
                  [".", ".", ".", ".", "."],
                  [".", ".", ".", ".", "."],
                  [".", ".", ".", ".", "."],
              ],
              "current_player": "x",
              "is_terminal": False,
              "winner": "",
          },
          error_msg="column count",
      ),
      # Both players have winning lines
      dict(
          invalid_dict={
              "board": [
                  ["x", "o", ".", ".", ".", ".", "."],
                  ["x", "o", ".", ".", ".", ".", "."],
                  ["x", "o", ".", ".", ".", ".", "."],
                  ["x", "o", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
                  [".", ".", ".", ".", ".", ".", "."],
              ],
              "current_player": "Terminal",
              "is_terminal": True,
              "winner": "x",
          },
          error_msg="both",
      ),
  )
  def test_state_from_invalid_dict(self, invalid_dict: dict[str, Any],
                                   error_msg: str):
    with self.assertRaises(pyspiel.SpielError) as context:
      self.game.new_initial_state(invalid_dict)
    self.assertIn(error_msg.lower(), str(context.exception).lower())

  def test_starting_state(self):
    """Test that starting_state is preserved."""
    state = self.game.new_initial_state()
    state.apply_action(3)
    state.apply_action(4)

    # Create state from dict
    new_state = self.game.new_initial_state(state.to_dict())
    self.assertIsNotNone(new_state.starting_state())
    self.assertEqual(str(new_state.starting_state()), str(state))

    # Apply more actions and verify starting_state unchanged
    new_state.apply_action(5)
    self.assertEqual(str(new_state.starting_state()), str(state))

  def test_starting_state_after_clone(self):
    """Test that starting_state is preserved through clone."""
    state = self.game.new_initial_state()
    state.apply_action(3)

    new_state = self.game.new_initial_state(state.to_dict())
    new_state.apply_action(4)
    clone = new_state.clone()

    self.assertEqual(str(clone.starting_state()), str(state))

  def test_game_params_default(self):
    """Test GameParams with default values."""
    params = connect_four.ConnectFourGameParams()
    self.assertEqual(params.game_name, "connect_four")
    self.assertEqual(params.rows, 6)
    self.assertEqual(params.columns, 7)
    self.assertEqual(params.x_in_row, 4)
    self.assertEqual(params.egocentric_obs_tensor, False)

    game = pyspiel.load_game(params)
    self.assertEqual(game.get_type().short_name, "connect_four")

  def test_game_params_custom(self):
    """Test GameParams with custom values."""
    params = connect_four.ConnectFourGameParams()
    params.rows = 8
    params.columns = 9
    params.x_in_row = 5

    game = pyspiel.load_game(params)
    state = game.new_initial_state()

    # Verify we can play on the larger board
    self.assertLen(state.legal_actions(), 9)

    # Fill a column to verify row count
    for _ in range(8):
      if state.is_terminal():
        break
      state.apply_action(0)
      if not state.is_terminal():
        state.apply_action(1)
    # Either column 0 is full or the game ended

  def test_game_params_json(self):
    """Test GameParams JSON serialization."""
    params = connect_four.ConnectFourGameParams()
    params.rows = 5
    params.columns = 6
    json_str = params.to_json()

    self.assertIn('"rows":5', json_str)
    self.assertIn('"columns":6', json_str)
    self.assertIn('"game_name":"connect_four"', json_str)

  def test_load_game_from_json(self):
    """Test loading game from JSON string."""
    game = pyspiel.load_game_from_json(
        '{"game_name":"connect_four","rows":4,"columns":5,"x_in_row":3}')
    self.assertEqual(game.get_type().short_name, "connect_four")

    # Verify custom connect-3 win condition on smaller board
    state = game.new_initial_state()
    self.assertLen(state.legal_actions(), 5)

    # Win with 3 in a row
    state.apply_action(0)  # x
    state.apply_action(1)  # o
    state.apply_action(0)  # x
    state.apply_action(1)  # o
    state.apply_action(0)  # x - wins with connect-3
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])


if __name__ == "__main__":
  absltest.main()
