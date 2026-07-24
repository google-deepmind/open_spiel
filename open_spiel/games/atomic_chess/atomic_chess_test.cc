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

#include "open_spiel/games/atomic_chess/atomic_chess.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace atomic_chess {
namespace {

using chess::Color;
using chess::ColorToPlayer;
using chess::LossUtility;
using chess::Move;
using chess::Piece;
using chess::PieceType;
using chess::Square;
using chess::WinUtility;

inline const int kBlackPlayer = ColorToPlayer(Color::kBlack);
inline const int kWhitePlayer = ColorToPlayer(Color::kWhite);

namespace testing = open_spiel::testing;

void ApplySANMove(const char* move_san, AtomicChessState* state) {
  std::optional<Move> maybe_move = state->Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  state->ApplyAction(chess::MoveToAction(*maybe_move, state->BoardSize()));
}

bool AnyActionLandsOn(const AtomicChessState& state, int8_t x, int8_t y) {
  for (Action action : state.LegalActions()) {
    Move move = chess::ActionToMove(action, state.Board());
    if (move.to.x == x && move.to.y == y) return true;
  }
  return false;
}

void BasicAtomicChessTests() {
  testing::LoadGameTest("atomic_chess");
  testing::NoChanceOutcomesTest(*LoadGame("atomic_chess"));
  testing::RandomSimTest(*LoadGame("atomic_chess"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("atomic_chess"), 3);
}

// Kings may not capture (a king capture would explode the king itself).
void KingCannotCaptureTest() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");
  // White king e4 is checked by a black rook on e5; it must NOT be able to
  // capture the rook (it would explode itself). Black king far away (a8).
  AtomicChessState state(game, "k7/8/8/4r3/4K3/8/8/8 w - - 0 1");
  // e5 = file e (x=4), rank 5 (y=4).
  SPIEL_CHECK_FALSE(AnyActionLandsOn(state, /*x=*/4, /*y=*/4));
  // But the king does have legal escape moves.
  SPIEL_CHECK_GT(state.LegalActions().size(), 0);
}

// A capture that explodes the enemy king wins immediately.
void WinByExplodingKingTest() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");
  // White rook e1 captures the black rook e7; the explosion (centered on e7)
  // removes the black king on e8. White king safe on a1.
  AtomicChessState state(game, "4k3/4r3/8/8/8/8/8/K3R3 w - - 0 1");
  ApplySANMove("Rxe7", &state);
  SPIEL_CHECK_TRUE(state.IsTerminal());
  std::vector<double> returns = state.Returns();
  SPIEL_CHECK_EQ(returns[kWhitePlayer], WinUtility());
  SPIEL_CHECK_EQ(returns[kBlackPlayer], LossUtility());
}

// A move that would explode one's own king is illegal.
void SelfExplosionIllegalTest() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");
  // White rook a2 could capture the black knight d2, but d2 is adjacent to the
  // white king on e1, so the capture would explode the white king -> illegal.
  AtomicChessState state(game, "k7/8/8/8/8/8/R2n4/4K3 w - - 0 1");
  // d2 = file d (x=3), rank 2 (y=1).
  SPIEL_CHECK_FALSE(AnyActionLandsOn(state, /*x=*/3, /*y=*/1));
}

// Pawns survive explosions; other pieces adjacent to the blast do not.
void PawnSurvivesExplosionTest() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");
  // White rook d1 captures black rook d5. Blast at d5 destroys the white
  // knight on e5 but leaves the white pawn on c4 intact.
  AtomicChessState state(game, "k7/8/8/3rN3/2P5/8/8/K2R4 w - - 0 1");
  ApplySANMove("Rxd5", &state);
  // c4 = (x=2, y=3) pawn survives.
  Piece pawn = state.Board().at(Square{2, 3});
  SPIEL_CHECK_EQ(pawn.type, PieceType::kPawn);
  SPIEL_CHECK_EQ(pawn.color, Color::kWhite);
  // e5 = (x=4, y=4) knight destroyed; d5 = (x=3, y=4) empty.
  SPIEL_CHECK_EQ(state.Board().at(Square{4, 4}).type, PieceType::kEmpty);
  SPIEL_CHECK_EQ(state.Board().at(Square{3, 4}).type, PieceType::kEmpty);
}

// Stalemate is a draw (unlike antichess, where it is a win).
void StalemateIsDrawTest() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");
  // Black (only a king on a8) is not in check but has no legal move:
  // a7/b7 covered by the rook on c7, b8 covered by the knight on a6.
  AtomicChessState state(game, "k7/2R5/N7/8/8/8/8/4K3 b - - 0 1");
  SPIEL_CHECK_TRUE(state.IsTerminal());
  SPIEL_CHECK_EQ(state.Returns(), std::vector<double>(chess::NumPlayers(),
                                                      chess::DrawUtility()));
}

void SerializationTest() {
  auto game = LoadGame("atomic_chess");
  std::unique_ptr<State> state = game->NewInitialState();
  std::shared_ptr<State> deserialized_state =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());

  state = game->NewInitialState(
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
  deserialized_state = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

}  // namespace
}  // namespace atomic_chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::atomic_chess::BasicAtomicChessTests();
  open_spiel::atomic_chess::KingCannotCaptureTest();
  open_spiel::atomic_chess::WinByExplodingKingTest();
  open_spiel::atomic_chess::SelfExplosionIllegalTest();
  open_spiel::atomic_chess::PawnSurvivesExplosionTest();
  open_spiel::atomic_chess::StalemateIsDrawTest();
  open_spiel::atomic_chess::SerializationTest();
}
