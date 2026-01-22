
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

#ifndef OPEN_SPIEL_GAMES_CRAZYHOUSE_BOARD_H_
#define OPEN_SPIEL_GAMES_CRAZYHOUSE_BOARD_H_

#include <array>
#include <vector>

#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess/chess_common.h"

namespace open_spiel {
namespace crazyhouse {

// Pockets store the count of captured pieces available to be dropped.
// Index 0 = Black Pockets, Index 1 = White Pockets.
// Inside array: index corresponds to PieceType (e.g. kPawn=6, etc.)
// We use a simplified mapping or just a map. Since PieceType is int8_t, 
// we can use a small array. Max PieceType is kPawn=6. So size 8 is enough.
using Pocket = std::array<int, 8>;

class CrazyhouseBoard : public chess::ChessBoard {
 public:
  CrazyhouseBoard(int board_size = chess::kDefaultBoardSize,
                  bool king_in_check_allowed = false,
                  bool allow_pass_move = false);

  explicit CrazyhouseBoard(const chess::ChessBoard& other) 
      : chess::ChessBoard(other) {
    pockets_[0].fill(0);
    pockets_[1].fill(0);
  }

  // Parse FEN with pocket support.
  // Standard FEN + "[PocketString]" e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1[PNBR]"
  static absl::optional<CrazyhouseBoard> BoardFromFEN(
      const std::string& fen, int board_size = 8);

  const Pocket& pocket(chess::Color color) const {
    return pockets_[chess::ToInt(color)];
  }
  Pocket& pocket(chess::Color color) {
    return pockets_[chess::ToInt(color)];
  }

  void AddToPocket(chess::Color color, chess::PieceType type);
  void RemoveFromPocket(chess::Color color, chess::PieceType type);

  // Generate legal moves including drops.
  void GenerateLegalMoves(const MoveYieldFn& yield) const;
  
  // Apply a move (handles drops and captures adding to pocket).
  void ApplyMove(const chess::Move& move);

  std::string ToFEN() const;
  std::string DebugString() const;

 private:
  std::array<Pocket, 2> pockets_;
  
  // Helper for generating drop moves
  void GenerateDropMoves(const MoveYieldFn& yield, chess::Color color) const;
};

// Returns a string representation of the pocket, e.g. "PNBR"
std::string PocketToString(const Pocket& pocket, chess::Color color);

// Helper to determine if a move is a drop move.
// We can use a special logic, e.g. from_square == kInvalidSquare and piece.type != kEmpty
bool IsDropMove(const chess::Move& move);

}  // namespace crazyhouse
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CRAZYHOUSE_BOARD_H_
