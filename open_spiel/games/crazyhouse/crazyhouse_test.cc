
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

#include "absl/strings/match.h"
#include "open_spiel/games/crazyhouse/crazyhouse.h"
#include "open_spiel/games/crazyhouse/crazyhouse_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crazyhouse {
namespace {

namespace testing = open_spiel::testing;

void BasicGameTests() {
  testing::LoadGameTest("crazyhouse");
  testing::RandomSimTest(*LoadGame("crazyhouse"), 10);
}

void PocketTest() {
  CrazyhouseBoard board;
  // White pocket empty
  SPIEL_CHECK_EQ(
      board.pocket(
          chess::Color::kWhite)[static_cast<int>(chess::PieceType::kPawn)],
      0);

  // Add pawn to white pocket
  board.AddToPocket(chess::Color::kWhite, chess::PieceType::kPawn);
  SPIEL_CHECK_EQ(
      board.pocket(
          chess::Color::kWhite)[static_cast<int>(chess::PieceType::kPawn)],
      1);

  // Serialize FEN
  std::string fen = board.ToFEN();
  // Expect standard FEN + [P]
  SPIEL_CHECK_TRUE(absl::StrContains(fen, "[P]"));

  // Parse back
  auto board2 = CrazyhouseBoard::BoardFromFEN(fen);
  SPIEL_CHECK_TRUE(board2.has_value());
  SPIEL_CHECK_EQ(
      board2->pocket(
          chess::Color::kWhite)[static_cast<int>(chess::PieceType::kPawn)],
      1);
}

void DropMoveTest() {
  // Empty board for simplicity tests?
  // No, start standard.
  std::shared_ptr<const Game> game = LoadGame("crazyhouse");
  std::unique_ptr<State> state = game->NewInitialState();

  // 1. e4 d5
  state->ApplyAction(state->StringToAction("e2e4"));
  state->ApplyAction(state->StringToAction("d7d5"));

  // 2. exd5 (White captures Pawn)
  state->ApplyAction(state->StringToAction("e4d5"));

  auto ch_state = static_cast<const CrazyhouseState *>(state.get());
  // Black to move. White should have a Pawn in pocket?
  // Capture happened on White's turn?
  // e4 captures d5. White moves.
  // Now it is Black's turn.
  // White captured a black pawn. So White's pocket should have a pawn.
  // Wait, `CrazyhouseBoard::ApplyMove` adds to `OppColor(captured.color)`?
  // Captured piece was Black (d5).
  // OppColor(Black) = White.
  // So White gets the pawn. Correct.

  const CrazyhouseBoard &board = ch_state->Board();
  SPIEL_CHECK_EQ(
      board.pocket(
          chess::Color::kWhite)[static_cast<int>(chess::PieceType::kPawn)],
      1);
  SPIEL_CHECK_EQ(
      board.pocket(
          chess::Color::kBlack)[static_cast<int>(chess::PieceType::kPawn)],
      0);

  // Black moves 2... Qxd5
  state->ApplyAction(state->StringToAction("d8d5"));

  // White moves 3. P@e6 (Drop pawn on e6) - Valid?
  // Encode drop: P@e6
  // e6 is empty.
  // Action string should be "P@e6"
  Action drop_action = state->StringToAction("P@e6");
  SPIEL_CHECK_NE(drop_action, kInvalidAction);

  // Is it legal?
  std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), drop_action) !=
                   legal.end());

  // Apply drop
  state->ApplyAction(drop_action);

  // Check board state: e6 has White Pawn
  const CrazyhouseBoard &board_after =
      static_cast<const CrazyhouseState *>(state.get())->Board();
  chess::Square e6{4, 5}; // e=4, 1-based rank 6 -> y=5
  chess::Piece p = board_after.at(e6);
  SPIEL_CHECK_EQ(p.type, chess::PieceType::kPawn);
  SPIEL_CHECK_EQ(p.color, chess::Color::kWhite);

  // Pocket should be empty now
  SPIEL_CHECK_EQ(
      board_after.pocket(
          chess::Color::kWhite)[static_cast<int>(chess::PieceType::kPawn)],
      0);

  std::cout << "DropMoveTest Passed!" << std::endl;
}

} // namespace
} // namespace crazyhouse
} // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::crazyhouse::BasicGameTests();
  open_spiel::crazyhouse::PocketTest();
  open_spiel::crazyhouse::DropMoveTest();
}
