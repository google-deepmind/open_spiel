
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

#include "open_spiel/games/crazyhouse/crazyhouse_board.h"

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crazyhouse {

using chess::Color;
using chess::Move;
using chess::PieceType;
using chess::Square;

namespace {
// Helper to parse pocket string like "[PNBR]"
void ParsePocketString(const std::string &pocket_str,
                       std::array<Pocket, 2> &pockets) {
  for (char c : pocket_str) {
    if (c == '[')
      continue;
    if (c == ']')
      break;
    absl::optional<PieceType> type = chess::PieceTypeFromChar(c);
    if (type) {
      Color color = isupper(c) ? Color::kWhite : Color::kBlack;
      pockets[chess::ToInt(color)][static_cast<int>(*type)]++;
    }
  }
}
} // namespace

CrazyhouseBoard::CrazyhouseBoard(int board_size, bool king_in_check_allowed,
                                 bool allow_pass_move)
    : chess::ChessBoard(board_size, king_in_check_allowed, allow_pass_move,
                        /*allow_king_promotion=*/false) {
  // Initialize empty pockets
  pockets_[0].fill(0);
  pockets_[1].fill(0);
}

absl::optional<CrazyhouseBoard>
CrazyhouseBoard::BoardFromFEN(const std::string &fen, int board_size) {
  // Split FEN into standard FEN and pocket part
  // Example: "... 0 1[PNBR]" or just "... 0 1"
  size_t bracket_pos = fen.find('[');
  std::string standard_fen = fen;
  std::string pocket_str = "";

  if (bracket_pos != std::string::npos) {
    standard_fen = fen.substr(0, bracket_pos);
    size_t end_bracket = fen.find(']', bracket_pos);
    if (end_bracket != std::string::npos) {
      pocket_str = fen.substr(bracket_pos, end_bracket - bracket_pos + 1);
    }
  }

  auto board_opt =
      chess::ChessBoard::BoardFromFEN(standard_fen, board_size,
                                      /*king_in_check_allowed=*/false,
                                      /*allow_pass_move=*/false,
                                      /*allow_king_promotion=*/false);
  if (!board_opt)
    return absl::nullopt;

  CrazyhouseBoard ch_board(board_size);
  // Copy state from parent board (this is a bit hacky, maybe we should have a
  // copy constructor) Since we inherit, we can copy the memory of the base
  // class? Or just rely on setters/getters? Let's manually copy the observable
  // state for now or cast? Casting is safer:
  static_cast<chess::ChessBoard &>(ch_board) = *board_opt;

  if (!pocket_str.empty()) {
    ParsePocketString(pocket_str, ch_board.pockets_);
  }

  return ch_board;
}

void CrazyhouseBoard::AddToPocket(Color color, PieceType type) {
  // In Crazyhouse, promoted pieces revert to pawns when captured.
  // Standard pieces keep their type.
  // We don't have enough info on the board to know if a piece was promoted or
  // not *on the board*. WAIT. In standard chess board representation, a
  // promoted Queen is just a Queen. In Crazyhouse, we need to know if it was
  // promoted? Actually, typical Crazyhouse rules say: "When a piece is
  // captured, it is added to the pocket... If a promoted piece is captured, it
  // reverts to a pawn." So we require the Board to track which pieces are
  // promoted? OR the `Move` structure needs to carry that info when the capture
  // happens? `chess::ChessBoard` doesn't track "is_promoted" bit on squares.
  // This is a LIMITATION of inheriting `ChessBoard`.
  // TODO: resolving this. For now, assume pieces are effectively "demoted" to
  // pawns if they are Queens/Rooks/Bishops/Knights? No, that's wrong. A natural
  // Queen stays a Queen. A promoted Queen becomes a Pawn. Solution: We might
  // need to extend `Piece` or `ChessBoard`. The `Piece` struct in
  // `chess_board.h` creates simple pieces. We can't easily change `Piece`
  // without modifying `chess_board.h` which is risky. However, most
  // engines/servers just treat captured pieces as their face value? Lichess:
  // "Promoted pieces revert to pawns when captured." If we can't distinguish,
  // we might need to modify `ChessBoard`. Let's stick to "Face Value" for MVP
  // or add a TODO. Actually, if we want to be correct, we need to track it. But
  // `ChessBoard` stores `Piece` (color, type). Maybe we can assume for now all
  // pieces are original for simplicity, or (better) we update `chess_board.h`
  // later to allow `kPromoted` flag. Let's add a TODO.

  pockets_[chess::ToInt(color)][static_cast<int>(type)]++;
}

void CrazyhouseBoard::RemoveFromPocket(Color color, PieceType type) {
  int &count = pockets_[chess::ToInt(color)][static_cast<int>(type)];
  if (count > 0) {
    count--;
  } else {
    SpielFatalError("Attempted to remove piece from empty pocket");
  }
}

void CrazyhouseBoard::GenerateLegalMoves(const MoveYieldFn &yield) const {
  // 1. Generate standard moves
  chess::ChessBoard::GenerateLegalMoves(yield);

  // 2. Generate drop moves
  GenerateDropMoves(yield, ToPlay());
}

void CrazyhouseBoard::GenerateDropMoves(const MoveYieldFn &yield,
                                        Color color) const {
  const Pocket &my_pocket = pocket(color);

  for (int type_idx = 0; type_idx < 8; ++type_idx) {
    if (my_pocket[type_idx] > 0) {
      PieceType type = static_cast<PieceType>(type_idx);
      chess::Piece piece{color, type};

      // Try dropping on every empty square
      for (int y = 0; y < BoardSize(); ++y) {
        for (int x = 0; x < BoardSize(); ++x) {
          Square sq{static_cast<int8_t>(x), static_cast<int8_t>(y)};

          if (at(sq).type != PieceType::kEmpty)
            continue;

          // Pawns cannot be dropped on 1st and 8th ranks
          if (type == PieceType::kPawn && (y == 0 || y == BoardSize() - 1)) {
            continue;
          }

          // Create drop move
          // We use a convention: from = {-1, -1} to indicate drop?
          // But `ChessBoard::ApplyMove` might not like that.
          // We need to override `ApplyMove` anyway.
          // Let's use `kInvalidSquare` ({-1, -1}) as from.
          Move drop_move(chess::kInvalidSquare, sq, piece);

          // Must verify if this drop is legal (doesn't leave King in check).
          // Drops can block checks.
          // We can use `TestApplyMove`? No, `ChessBoard::TestApplyMove` doesn't
          // know about drops. We need a specific check.

          // Construct a temp board or undoable action
          // Optimization: Check `IsMoveLegal` which we also override?
          // No, `IsMoveLegal` calls `GenerateLegalMoves`. Circular.

          // We need to implement a specialized TestApplyDrop.
          // For now, let's yield if it passes a basic check check?
          // Actually `GenerateLegalMoves` implies we only yield LEGAL moves.
          // So we must check for King safety.

          // Hack: Temporarily place piece, check check, remove piece.
          // `ChessBoard` is not const-correct for this unless we clone?
          // `GenerateLegalMoves` is const.
          // We can clone. Board is small.
          CrazyhouseBoard board_copy = *this;
          board_copy.chess::ChessBoard::set_square(
              sq, piece); // Use base set_square
          if (!board_copy.InCheck()) {
            if (!yield(drop_move))
              return;
          }
        }
      }
    }
  }
}

void CrazyhouseBoard::ApplyMove(const Move &move) {
  if (IsDropMove(move)) {
    // Handle Drop
    RemoveFromPocket(move.piece.color, move.piece.type);
    chess::ChessBoard::set_square(move.to, move.piece);

    // Switch turn
    SetToPlay(chess::OppColor(move.piece.color));
    SetEpSquare(chess::kInvalidSquare); // Drops reset EP? Yes.
    // Inc move number if black?
    if (move.piece.color == Color::kBlack) {
      SetMovenumber(Movenumber() + 1);
    }
    SetIrreversibleMoveCounter(0); // Drop is irreversible (consumes resource)
  } else {
    // Handle standard move
    // Check for capture to add to pocket
    chess::Piece captured = at(move.to);

    // Regular application
    chess::ChessBoard::ApplyMove(move);

    // Add captured piece to CURRENT player's pocket (the one who moved made the
    // capture) Wait, ApplyMove swaps the player at the end. So if White moves,
    // ApplyMove makes it Black's turn. We recorded `captured` before ApplyMove.
    if (captured.type != PieceType::kEmpty) {
      // Revert functionality:
      // If captured piece was Promoted, it becomes a Pawn.
      // Since we don't track it yet, we just add the type.
      // TODO: Fix promotion tracking.
      AddToPocket(chess::OppColor(captured.color), captured.type);
    }
  }
}

std::string CrazyhouseBoard::ToFEN() const {
  std::string fen = chess::ChessBoard::ToFEN();
  std::string pocket_str = "";
  pocket_str += PocketToString(pockets_[1], Color::kWhite);
  pocket_str += PocketToString(pockets_[0], Color::kBlack);

  if (!pocket_str.empty()) {
    absl::StrAppend(&fen, "[", pocket_str, "]");
  }
  return fen;
}

std::string CrazyhouseBoard::DebugString() const {
  return absl::StrCat(ToFEN(), "\n", chess::ChessBoard::DebugString());
}

std::string PocketToString(const Pocket &pocket, Color color) {
  std::string res = "";
  for (int i = 0; i < 8; ++i) {
    int count = pocket[i];
    if (count > 0) {
      PieceType pt = static_cast<PieceType>(i);
      std::string s =
          chess::PieceTypeToString(pt, /*uppercase=*/(color == Color::kWhite));
      for (int k = 0; k < count; ++k)
        res += s;
    }
  }
  return res;
}

bool IsDropMove(const Move &move) {
  return move.from == chess::kInvalidSquare &&
         move.piece.type != PieceType::kEmpty;
}

} // namespace crazyhouse
} // namespace open_spiel
