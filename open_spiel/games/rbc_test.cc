// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/rbc.h"

#include "open_spiel/games/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace rbc {
namespace {

namespace testing = open_spiel::testing;

void RbcTestPassMove() {
  auto game = LoadGame("rbc");
  std::unique_ptr<State> s = game->NewInitialState();
  SPIEL_CHECK_EQ(s->ToString(), chess::kDefaultStandardFEN);

  // First move
  SPIEL_CHECK_EQ(s->ActionToString(Player{0}, 0), "Sense a1");
  s->ApplyAction(0);  // Sense phase
  SPIEL_CHECK_EQ(s->ActionToString(Player{0}, chess::kPassAction), "pass");
  s->ApplyAction(chess::kPassAction);  // Move phase
  std::string black_fen = chess::kDefaultStandardFEN;
  std::replace(black_fen.begin(), black_fen.end(), 'w', 'b');  // Switch sides.
  SPIEL_CHECK_EQ(s->ToString(), black_fen);

  // Second move
  SPIEL_CHECK_EQ(s->ActionToString(Player{1}, 0), "Sense a1");
  s->ApplyAction(0);  // Sense phase
  SPIEL_CHECK_EQ(s->ActionToString(Player{1}, chess::kPassAction), "pass");
  s->ApplyAction(chess::kPassAction);  // Move phase
  std::string white_fen = chess::kDefaultStandardFEN;
  std::replace(white_fen.begin(), white_fen.end(), '1', '2');  // Update clock.
  SPIEL_CHECK_EQ(s->ToString(), white_fen);
}

void BasicRbcTests(int board_size) {
  GameParameters params;
  params["board_size"] = GameParameter(board_size);

  testing::LoadGameTest("rbc");
  testing::NoChanceOutcomesTest(*LoadGame("rbc", params));
  testing::RandomSimTest(*LoadGame("rbc", params), 100000);
  testing::RandomSimTestWithUndo(*LoadGame("rbc", params), 100000);
}

}  // namespace
}  // namespace rbc
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::rbc::RbcTestPassMove();
  open_spiel::rbc::BasicRbcTests(/*board_size=*/8);
  open_spiel::rbc::BasicRbcTests(/*board_size=*/4);
}
