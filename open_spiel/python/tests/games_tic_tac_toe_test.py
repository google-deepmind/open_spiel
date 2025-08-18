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


from absl.testing import absltest

import pyspiel
ttt = pyspiel.tic_tac_toe


def make_game():
  return pyspiel.load_game("tic_tac_toe")


class GamesTicTacToeTest(absltest.TestCase):

  def test_constants(self):
    self.assertEqual(ttt.NUM_ROWS, 3)
    self.assertEqual(ttt.NUM_COLS, 3)
    self.assertEqual(ttt.NUM_CELLS, 9)
    self.assertEqual(ttt.CellState.EMPTY.value, 0)
    self.assertEqual(ttt.CellState.NOUGHT.value, 1)
    self.assertEqual(ttt.CellState.CROSS.value, 2)

  def test_player_to_cellstate(self):
    self.assertEqual(ttt.player_to_cellstate(0),
                     ttt.CellState.CROSS)
    self.assertEqual(ttt.player_to_cellstate(1),
                     ttt.CellState.NOUGHT)

  def test_cellstate_to_string(self):
    self.assertEqual(ttt.cellstate_to_string(ttt.CellState.EMPTY), ".")
    self.assertEqual(ttt.cellstate_to_string(ttt.CellState.NOUGHT), "o")
    self.assertEqual(ttt.cellstate_to_string(ttt.CellState.CROSS), "x")

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


if __name__ == "__main__":
  absltest.main()

