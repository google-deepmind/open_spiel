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

"""Tests for the game-specific functions for tic_tac_toe."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

import pyspiel
ttt = pyspiel.tic_tac_toe


def make_game():
  return pyspiel.load_game("tic_tac_toe")


class GamesTicTacToeTest(parameterized.TestCase):

  def test_constants(self):
    self.assertEqual(ttt.NUM_ROWS, 3)
    self.assertEqual(ttt.NUM_COLS, 3)
    self.assertEqual(ttt.NUM_CELLS, 9)
    self.assertEqual(ttt.CellState.EMPTY.value, 0)
    self.assertEqual(ttt.CellState.NOUGHT.value, 1)
    self.assertEqual(ttt.CellState.CROSS.value, 2)

  @parameterized.parameters(
      (0, ttt.CellState.CROSS),
      (1, ttt.CellState.NOUGHT),
  )
  def test_player_to_cellstate(
      self, player: int, expected_cellstate: ttt.CellState
  ):
    self.assertEqual(ttt.player_to_cellstate(player), expected_cellstate)

  @parameterized.parameters(
      (ttt.CellState.EMPTY, "."),
      (ttt.CellState.NOUGHT, "o"),
      (ttt.CellState.CROSS, "x"),
  )
  def test_cellstate_to_string(
      self, cellstate: ttt.CellState, expected_string: str
  ):
    self.assertEqual(ttt.cellstate_to_string(cellstate), expected_string)

  def test_board_at(self):
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(4)
    self.assertEqual(state.board_at(1, 1), ttt.CellState.CROSS)

  def test_board(self):
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    self.assertEqual(state.board(), [
        ttt.CellState.CROSS,
        ttt.CellState.NOUGHT] + [ttt.CellState.EMPTY] * 7)

  def test_json(self):
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(4)
    state_struct = state.to_struct()
    self.assertEqual(
        state_struct.board,
        [".", ".", ".", ".", "x", ".", ".", ".", "."],
    )
    self.assertEqual(state_struct.current_player, "o")
    json_from_struct = state_struct.to_json()
    state_json = state.to_json()
    self.assertEqual(
        state_json,
        '{"board":[".",".",".",".","x",".",".",".","."],"current_player":"o"}',
    )
    self.assertEqual(json_from_struct, state_json)
    state_struct = ttt.TicTacToeStateStruct(state_json)
    self.assertEqual(state_struct.to_json(), state_json)

  @parameterized.parameters(
      dict(
          actions=[4],
          expected_dict={
              "board": [".", ".", ".", ".", "x", ".", ".", ".", "."],
              "current_player": "o",
          },
      ),
      dict(
          actions=[0, 1, 2],
          expected_dict={
              "board": ["x", "o", "x", ".", ".", ".", ".", ".", "."],
              "current_player": "o",
          },
      ),
      dict(
          actions=[0, 1, 3, 4, 6],  # X wins
          expected_dict={
              "board": ["x", "o", ".", "x", "o", ".", "x", ".", "."],
              "current_player": "Terminal",
          },
      ),
      dict(
          actions=[0, 3, 1, 4, 5, 6, 2],  # X wins
          expected_dict={
              "board": ["x", "x", "x", "o", "o", "x", "o", ".", "."],
              "current_player": "Terminal",
          },
      ),
  )
  def test_state_to_from_dict(
      self, actions: list[int], expected_dict: dict[str, Any]
  ):
    game = make_game()
    state = game.new_initial_state()
    for action in actions:
      state.apply_action(action)
      # Test round trip.
      new_state = game.new_initial_state(state.to_dict())
      self.assertEqual(new_state.to_dict(), state.to_dict())
    state_dict = state.to_dict()
    self.assertEqual(state_dict, expected_dict)

  @parameterized.parameters(
      dict(
          invalid_dict={
              "board": ["."],  # Wrong length.
              "current_player": "x",
          }
      ),
      dict(
          invalid_dict={
              "board": [".", ".", ".", ".", "x", ".", ".", ".", "."],
              "current_player": "x",  # Should be o's turn.
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "x", ".", ".", ".", ".", ".", ".", "."],
              "current_player": "o",  # Count difference > 1.
          }
      ),
      dict(
          invalid_dict={
              "board": ["o", ".", ".", ".", ".", ".", ".", ".", "."],
              "current_player": "o",  # Player o cannot start.
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "o", ".", ".", ".", ".", ".", ".", "."],
              "current_player": "o",  # Should be x's turn.
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "o", "y", ".", ".", ".", ".", ".", "."],
              "current_player": "x",  # Invalid character 'y'.
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "x", "x", "o", "o", ".", ".", ".", "."],
              "current_player": "o",  # Should be "Terminal".
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "x", "x", "o", "o", "o", ".", ".", "."],
              "current_player": "Terminal",  # Both players have a line.
          }
      ),
      dict(
          invalid_dict={
              "board": ["x", "x", "x", "o", "o", ".", "o", ".", "."],
              "current_player": "Terminal",  # x has a line but o just moved.
          }
      ),
  )
  def test_state_from_invalid_dict(self, invalid_dict: dict[str, Any]):
    game = make_game()
    with self.assertRaises(pyspiel.SpielError):
      game.new_initial_state(invalid_dict)


if __name__ == "__main__":
  absltest.main()
