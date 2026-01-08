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

#include "open_spiel/games/antichess/antichess.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace antichess {
namespace {

using chess::Color;
using chess::ColorToPlayer;
using chess::LossUtility;
using chess::Move;
using chess::Piece;
using chess::PieceType;
using chess::WinUtility;

inline const int kBlackPlayer = ColorToPlayer(Color::kBlack);
inline const int kWhitePlayer = ColorToPlayer(Color::kWhite);

namespace testing = open_spiel::testing;

void BasicAntichessTests() {
  testing::LoadGameTest("antichess");
  testing::NoChanceOutcomesTest(*LoadGame("antichess"));
  testing::RandomSimTest(*LoadGame("antichess"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("antichess"), 10);
}

void ApplySANMove(const char* move_san, AntichessState* state) {
  absl::optional<Move> maybe_move = state->Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  state->ApplyAction(antichess::MoveToAction(*maybe_move));
}

// Test that captures are mandatory.
void MandatoryCaptureTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // Position where white has to capture.
  AntichessState state(game, "8/8/8/3p4/4P3/8/8/8 w - - 0 1");

  std::vector<Action> legal_actions = state.LegalActions();

  // There should only be one capture move available.
  bool all_captures = true;
  for (Action action : legal_actions) {
    Move move = antichess::ActionToMove(action, state.Board());
    Piece target = state.Board().at(move.to);
    if (target.type == PieceType::kEmpty) {
      all_captures = false;
      break;
    }
  }

  SPIEL_CHECK_TRUE(all_captures);
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
}

// Test that player wins when they lose all pieces.
void WinByLosingAllPiecesTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // White has no pieces, black has a king.
  AntichessState state(game, "8/8/8/8/8/8/8/k7 w - - 0 1");

  SPIEL_CHECK_TRUE(state.IsTerminal());
  std::vector<double> returns = state.Returns();

  // White should win because they lost all pieces.
  SPIEL_CHECK_EQ(returns[kBlackPlayer], LossUtility());
  SPIEL_CHECK_EQ(returns[kWhitePlayer], WinUtility());
}

// Test that player wins when they have no legal moves.
void WinByNoMovesTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // The only white piece to move is a pawn that is blocked.
  AntichessState state(game, "8/8/8/8/8/6p1/6P1/8 w - - 0 1");

  SPIEL_CHECK_TRUE(state.IsTerminal());
  std::vector<double> returns = state.Returns();

  // White should win because they have no legal moves.
  SPIEL_CHECK_EQ(returns[kBlackPlayer], LossUtility());
  SPIEL_CHECK_EQ(returns[kWhitePlayer], WinUtility());
}

// Test that castling is not allowed.
void NoCastlingTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // Standard starting position.
  AntichessState state(game);

  // Make some moves to enable castling in regular chess.
  ApplySANMove("e3", &state);
  ApplySANMove("e6", &state);
  ApplySANMove("Nf3", &state);
  ApplySANMove("Nf6", &state);
  ApplySANMove("Be2", &state);
  ApplySANMove("Be7", &state);

  // Check that castling is not in legal moves.
  std::vector<Action> legal_actions = state.LegalActions();
  for (Action action : legal_actions) {
    Move move = antichess::ActionToMove(action, state.Board());
    SPIEL_CHECK_FALSE(move.is_castling());
  }
}

// Test a simple game where captures are mandatory.
void MandatoryCaptureGameTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");
  AntichessState state(game);

  ApplySANMove("e3", &state);
  ApplySANMove("b6", &state);
  ApplySANMove("Ba6", &state);

  // Black must capture the bishop.
  std::vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2);

  // All legal moves should be captures of the bishop on a6.
  for (Action action : legal_actions) {
    Move move = antichess::ActionToMove(action, state.Board());
    // File a.
    SPIEL_CHECK_EQ(move.to.x, 0);
    // Rank 6.
    SPIEL_CHECK_EQ(move.to.y, 5);
  }
}

// Test that kings can be captured (no check/checkmate).
void KingsCanBeCapturedTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // White king must be captured by black bishop.
  AntichessState state(game, "8/8/8/3b4/4K3/8/8/8 b - - 0 1");

  std::vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  Move move = antichess::ActionToMove(legal_actions[0], state.Board());

  // Bishop captures king on e4.
  SPIEL_CHECK_TRUE(move.to.x == 4 && move.to.y == 3);
  Piece target = state.Board().at(move.to);
  SPIEL_CHECK_EQ(target.type, PieceType::kKing);
}

// Test serialization and deserialization.
void SerializationTest() {
  auto game = LoadGame("antichess");

  std::unique_ptr<State> state = game->NewInitialState();
  std::shared_ptr<State> deserialized_state =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());

  // FEN starting position.
  state = game->NewInitialState(
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b - - 1 2");
  deserialized_state = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

// Test repetition draw.
void RepetitionDrawTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");
  AntichessState state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 50 1");

  ApplySANMove("Kd1", &state);
  ApplySANMove("Ra2", &state);
  ApplySANMove("Kc1", &state);
  ApplySANMove("Rh2", &state);
  ApplySANMove("Kd1", &state);
  ApplySANMove("Ra2", &state);
  ApplySANMove("Kc1", &state);
  SPIEL_CHECK_FALSE(state.IsTerminal());
  ApplySANMove("Rh2", &state);
  SPIEL_CHECK_TRUE(state.IsTerminal());
  SPIEL_CHECK_EQ(state.Returns(), std::vector<double>(chess::NumPlayers(),
                                                      chess::DrawUtility()));
}

// Test that pawns can promote to kings.
void KingPromotionTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // White pawn on 7th rank, can promote to king
  AntichessState state(game, "8/7P/8/8/8/8/8/k6K w - - 0 1");

  std::vector<Action> legal_actions = state.LegalActions();

  // Should have 5 promotion options: Q, R, B, N, K
  // Can move straight to h8 or move King to 3 available squares.
  SPIEL_CHECK_GE(legal_actions.size(), 8);

  // Check that we have a king promotion action.
  bool has_king_promotion = false;
  for (Action action : legal_actions) {
    Move move = antichess::ActionToMove(action, state.Board());
    if (move.promotion_type == PieceType::kKing) {
      has_king_promotion = true;

      // Verify the move details
      SPIEL_CHECK_EQ(move.from.x, 7);  // file 'e'
      SPIEL_CHECK_EQ(move.from.y, 6);  // rank 7 (index 6)
      SPIEL_CHECK_EQ(move.to.x, 7);    // file 'e'
      SPIEL_CHECK_EQ(move.to.y, 7);    // rank 8 (index 7)

      // Apply the king promotion move
      AntichessState test_state = state;
      test_state.ApplyAction(action);

      // Check that there's now a white king on e8
      Piece piece = test_state.Board().at(chess::Square{7, 7});
      SPIEL_CHECK_EQ(piece.type, PieceType::kKing);
      SPIEL_CHECK_EQ(piece.color, Color::kWhite);

      break;
    }
  }

  SPIEL_CHECK_TRUE(has_king_promotion);
}

// Test king promotion with capture
void KingPromotionCaptureTest() {
  std::shared_ptr<const Game> game = LoadGame("antichess");

  // White pawn on 7th rank, must capture and can promote to king
  AntichessState state(game, "3r4/4P3/8/8/8/8/8/8 w - - 0 1");

  std::vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 5);

  bool has_king_promotion = false;
  // All moves should be captures to d8.
  for (Action action : legal_actions) {
    Move move = antichess::ActionToMove(action, state.Board());
    // File d.
    SPIEL_CHECK_EQ(move.to.x, 3);
    // Rank 8.
    SPIEL_CHECK_EQ(move.to.y, 7);
    if (move.promotion_type == PieceType::kKing) {
      has_king_promotion = true;
    }
  }

  SPIEL_CHECK_TRUE(has_king_promotion);
}

}  // namespace
}  // namespace antichess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::antichess::BasicAntichessTests();
  open_spiel::antichess::MandatoryCaptureTest();
  open_spiel::antichess::WinByLosingAllPiecesTest();
  open_spiel::antichess::WinByNoMovesTest();
  open_spiel::antichess::NoCastlingTest();
  open_spiel::antichess::MandatoryCaptureGameTest();
  open_spiel::antichess::KingsCanBeCapturedTest();
  open_spiel::antichess::SerializationTest();
  open_spiel::antichess::RepetitionDrawTest();
  open_spiel::antichess::KingPromotionTest();
  open_spiel::antichess::KingPromotionCaptureTest();
}
