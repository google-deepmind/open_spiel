// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace dark_chess {
namespace {

namespace testing = open_spiel::testing;

void BasicDarkChessTests(int board_size) {
  GameParameters params;
  params["board_size"] = GameParameter(board_size);

  testing::LoadGameTest("dark_chess");
  testing::NoChanceOutcomesTest(*LoadGame("dark_chess", params));
  testing::RandomSimTest(*LoadGame("dark_chess", params), 100);
  testing::RandomSimTestWithUndo(*LoadGame("dark_chess", params), 1);
}

void ChessBoardFlagPropagationTest(bool flag_king_in_check_allowed) {
  auto tested_move =
      chess::Move(/*from=*/{3, 0}, /*to=*/{2, 0},
                  {chess::Color::kWhite, chess::PieceType::kKing});

  auto board = chess::ChessBoard::BoardFromFEN("1kr1/4/4/3K w - - 0 1",
                                               /*board_size=*/4,
                                               flag_king_in_check_allowed);
  bool move_allowed = false;
  board->GenerateLegalMoves(
      [&move_allowed, tested_move](const chess::Move& found_move) {
        if (found_move == tested_move) {
          move_allowed = true;
          return false;
        }
        return true;
      });

  SPIEL_CHECK_EQ(move_allowed, flag_king_in_check_allowed);
}

}  // namespace
}  // namespace dark_chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dark_chess::BasicDarkChessTests(/*board_size=*/4);
  open_spiel::dark_chess::BasicDarkChessTests(/*board_size=*/8);

  open_spiel::dark_chess::ChessBoardFlagPropagationTest(
      /*flag_king_in_check_allowed=*/true);
  open_spiel::dark_chess::ChessBoardFlagPropagationTest(
      /*flag_king_in_check_allowed=*/false);
}
