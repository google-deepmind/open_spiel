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


from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel
from open_spiel.python.utils import file_utils

chess = pyspiel.chess


FLAGS = flags.FLAGS

# From CMakeLists.txt:Python tests are run from the main binary directory which
# will be something like build/python.
flags.DEFINE_string(
    "chess960_fens_file",
    "../../open_spiel/games/chess/chess960_starting_positions.txt",
    "FENs database for chess960",
)


class GamesChessTest(parameterized.TestCase):

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
    self.assertEqual(state.num_repetitions(state), 1)

  @parameterized.parameters(
      "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1",
      "rnbnkbqr/pppppppp/8/8/8/8/PPPPPPPP/RNBNKBQR w KQkq - 0 1",
      "rkrnnqbb/pppppppp/8/8/8/8/PPPPPPPP/RKRNNQBB w KQkq - 0 1",
  )
  def test_chess960_sim_specific_fens(self, initial_fen):
    game = pyspiel.load_game("chess(chess960=true)")
    state = game.new_initial_state(initial_fen)
    while not state.is_terminal():
      assert not state.is_chance_node()
      legal_actions = state.legal_actions()
      action = np.random.choice(legal_actions)
      state.apply_action(action)

  def test_chess_action_conversions(self):
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    for _ in range(10):
      while not state.is_terminal():
        assert not state.is_chance_node()
        legal_actions = state.legal_actions()
        for action in legal_actions:
          move = chess.action_to_move(action, state.board())
          move_uci = move.to_lan()
          action_mapped = chess.move_to_action(move, 8)
          self.assertEqual(
              action, action_mapped, f"Error for action {move_uci}"
          )
        action = np.random.choice(legal_actions)
        state.apply_action(action)

  def test_chess960_game_sim(self):
    fens_filename = file_utils.find_file(FLAGS.chess960_fens_file, 1)
    if fens_filename is not None:
      print("Found chess960 fens file. Running simulation tests.")
      game = pyspiel.load_game(
          f"chess(chess960=true,chess960_fens_file={fens_filename})"
      )
      for _ in range(10):
        state = game.new_initial_state()
        assert state.is_chance_node()
        outcomes = state.chance_outcomes()
        assert len(outcomes) == 960
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
        while not state.is_terminal():
          assert not state.is_chance_node()
          legal_actions = state.legal_actions()
          for action in legal_actions:
            move = chess.action_to_move(action, state.board())
            move_uci = move.to_lan()
            action_mapped = chess.move_to_action(move, 8)
            self.assertEqual(
                action, action_mapped, f"Error for action {move_uci}"
            )
          action = np.random.choice(legal_actions)
          state.apply_action(action)


if __name__ == "__main__":
  np.random.seed(87375711)
  absltest.main()
