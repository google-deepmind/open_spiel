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

"""Tests for the game-specific functions for chess."""


from absl.testing import absltest
import numpy as np

import pyspiel
chess = pyspiel.chess


class GamesChessTest(absltest.TestCase):

  def test_bindings_sim(self):
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    board = None
    while not state.is_terminal():
      print(state)
      player = state.current_player()
      legal_actions = state.legal_actions()
      board = state.board()
      for action in legal_actions:
        action_str = state.action_to_string(player, action)
        move = chess.action_to_move(action, board)
        move_from = move.from_square
        move_to = move.to_square
        decoded_from_to = (f"({move_from.x} {move_from.y}) -> " +
                           f"({move_to.x} {move_to.y})")
        print(f"Legal action: {action_str} decoded from to {decoded_from_to}")
        print(f"Move representations: {move.to_string()} | " +
              f"{move.to_lan()} | {move.to_san(board)}")
        # Now do the reverse mapping from both string representations to check
        # that they correspond to this action.
        action_from_lan = state.parse_move_to_action(move.to_lan())
        action_from_san = state.parse_move_to_action(move.to_san(board))
        self.assertEqual(action, action_from_lan)
        self.assertEqual(action, action_from_san)
      action = np.random.choice(legal_actions)
      state.apply_action(action)
    print(board.to_unicode_string())
    print(board.debug_string())
    print("Moves history:")
    print(" ".join([move.to_lan() for move in state.moves_history()]))
    self.assertTrue(state.is_terminal())

  def test_state_from_fen(self):
    game = pyspiel.load_game("chess")
    fen_string = "8/k1P5/8/1K6/8/8/8/8 w - - 0 1"
    state = game.new_initial_state(fen_string)
    self.assertEqual(state.board().to_fen(), fen_string)


if __name__ == "__main__":
  np.random.seed(87375711)
  absltest.main()
