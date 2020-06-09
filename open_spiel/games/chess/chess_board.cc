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

#include "open_spiel/games/chess/chess_board.h"

#include <cctype>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chess {

bool IsMoveCharacter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

std::pair<std::string, std::string> SplitAnnotations(const std::string &move) {
  for (int i = 0; i < move.size(); ++i) {
    if (!IsMoveCharacter(move[i])) {
      return {move.substr(0, i), std::string(absl::ClippedSubstr(move, i))};
    }
  }
  return {move, ""};
}

std::string ColorToString(Color c) {
  switch (c) {
    case Color::kBlack:
      return "black";
    case Color::kWhite:
      return "white";
    case Color::kEmpty:
      return "empty";
    default:
      SpielFatalError(absl::StrCat("Unknown color: ", c));
      return "This will never return.";
  }
}

absl::optional<PieceType> PieceTypeFromChar(char c) {
  switch (toupper(c)) {
    case 'P':
      return PieceType::kPawn;
    case 'N':
      return PieceType::kKnight;
    case 'B':
      return PieceType::kBishop;
    case 'R':
      return PieceType::kRook;
    case 'Q':
      return PieceType::kQueen;
    case 'K':
      return PieceType::kKing;
    default:
      std::cerr << "Invalid piece type: " << c << std::endl;
      return std::nullopt;
  }
}

std::string PieceTypeToString(PieceType p, bool uppercase) {
  switch (p) {
    case PieceType::kEmpty:
      return " ";
    case PieceType::kPawn:
      return uppercase ? "P" : "p";
    case PieceType::kKnight:
      return uppercase ? "N" : "n";
    case PieceType::kBishop:
      return uppercase ? "B" : "b";
    case PieceType::kRook:
      return uppercase ? "R" : "r";
    case PieceType::kQueen:
      return uppercase ? "Q" : "q";
    case PieceType::kKing:
      return uppercase ? "K" : "k";
    default:
      SpielFatalError("Unknown piece.");
      return "This will never return.";
  }
}

std::string Piece::ToUnicode() const {
  switch (color) {
    case Color::kBlack:
      switch (type) {
        case PieceType::kEmpty:
          return " ";
        case PieceType::kPawn:
          return "♟";
        case PieceType::kKnight:
          return "♞";
        case PieceType::kBishop:
          return "♝";
        case PieceType::kRook:
          return "♜";
        case PieceType::kQueen:
          return "♛";
        case PieceType::kKing:
          return "♚";
        default:
          SpielFatalError("Unknown piece.");
          return "This will never return.";
      }
    case Color::kWhite:
      switch (type) {
        case PieceType::kEmpty:
          return " ";
        case PieceType::kPawn:
          return "♙";
        case PieceType::kKnight:
          return "♘";
        case PieceType::kBishop:
          return "♗";
        case PieceType::kRook:
          return "♖";
        case PieceType::kQueen:
          return "♕";
        case PieceType::kKing:
          return "♔";
        default:
          SpielFatalError("Unknown piece type.");
          return "This will never return.";
      }
    case Color::kEmpty:
      return " ";
    default:
      SpielFatalError("Unknown color.");
      return "This will never return.";
  }
}

std::string Piece::ToString() const {
  std::string base = PieceTypeToString(type);
  return color == Color::kWhite ? absl::AsciiStrToUpper(base)
                                : absl::AsciiStrToLower(base);
}

absl::optional<Square> SquareFromString(const std::string &s) {
  if (s.size() != 2) return InvalidSquare();

  auto file = ParseFile(s[0]);
  auto rank = ParseRank(s[1]);
  if (file && rank) return Square{*file, *rank};
  return std::nullopt;
}

std::string Move::ToString() const {
  std::string extra;
  if (promotion_type != PieceType::kEmpty) {
    absl::StrAppend(&extra, ", promotion to ",
                    PieceTypeToString(promotion_type));
  }
  if (is_castling) {
    absl::StrAppend(&extra, " (castle)");
  }
  return absl::StrCat(piece.ToString(), " ", SquareToString(from), " to ",
                      SquareToString(to), extra);
}

std::string Move::ToLAN() const {
  std::string promotion;
  if (promotion_type != PieceType::kEmpty) {
    promotion = PieceTypeToString(promotion_type, false);
  }
  return absl::StrCat(SquareToString(from), SquareToString(to), promotion);
}

std::string Move::ToSAN(const StandardChessBoard &board) const {
  std::string move_text;
  PieceType piece_type = board.at(from).type;
  if (is_castling) {
    if (from.x < to.x) {
      move_text = "O-O";
    } else {
      move_text = "O-O-O";
    }
  } else {
    switch (piece_type) {
      case PieceType::kKing:
      case PieceType::kQueen:
      case PieceType::kRook:
      case PieceType::kBishop:
      case PieceType::kKnight:
        move_text += PieceTypeToString(piece_type);
        break;
      case PieceType::kPawn:
        // No piece type required.
        break;
      case PieceType::kEmpty:
        std::cerr << "Move doesn't have a piece type" << std::endl;
    }

    // Now we generate all moves from this position, and see if our file and
    // rank are unique.
    bool file_unique = true;
    bool rank_unique = true;
    bool disambiguation_required = false;

    board.GenerateLegalMoves([&](const Move &move) -> bool {
      if (move.to != to) {
        return true;
      }
      if (move.from == from) {
        // This is either us, or a promotion to a different type. We don't count
        // them as ambiguous in either case.
        return true;
      }
      disambiguation_required = true;
      if (move.from.x == from.x) {
        file_unique = false;
      } else if (move.from.y == from.y) {
        rank_unique = false;
      }
      return true;
    });

    bool file_required = false;
    bool rank_required = false;

    if (piece_type == PieceType::kPawn && from.x != to.x) {
      // Pawn captures always need file, and they will never require rank dis-
      // ambiguation.
      file_required = true;
    } else if (disambiguation_required) {
      if (file_unique) {
        // This includes when both will disambiguate, in which case we have to
        // use file. [FIDE Laws of Chess (2018): C.10.3].
        file_required = true;
      } else if (rank_unique) {
        rank_required = true;
      } else {
        // We have neither unique file nor unique rank. This is only possible
        // with 3 or more pieces of the same type.
        file_required = true;
        rank_required = true;
      }
    }

    if (file_required) {
      absl::StrAppend(&move_text, FileToString(from.x));
    }

    if (rank_required) {
      absl::StrAppend(&move_text, RankToString(from.y));
    }

    // We have a capture if either 1) the destination square has a piece, or
    // 2) we are making a diagonal pawn move (which can also be an en-passant
    // capture, where the destination square would not have a piece).
    auto piece_at_to_square = board.at(to);
    if ((piece_at_to_square.type != PieceType::kEmpty) ||
        (piece_type == PieceType::kPawn && from.x != to.x)) {
      absl::StrAppend(&move_text, "x");
    }

    // Destination square is always fully encoded.
    absl::StrAppend(&move_text, SquareToString(to));

    // Encode the promotion type if we have a promotion.
    switch (promotion_type) {
      case PieceType::kEmpty:
        break;
      case PieceType::kQueen:
      case PieceType::kRook:
      case PieceType::kBishop:
      case PieceType::kKnight:
        absl::StrAppend(&move_text, "=", PieceTypeToString(promotion_type));
        break;
      case PieceType::kKing:
      case PieceType::kPawn:
        std::cerr << "Cannot promote to " << PieceTypeToString(promotion_type)
                  << "! Only Q, R, B, N are allowed" << std::endl;
        break;
    }
  }

  // Figure out if this is a check / checkmating move or not.
  auto board_copy = board;
  board_copy.ApplyMove(*this);
  if (board_copy.InCheck()) {
    bool has_escape = false;
    board_copy.GenerateLegalMoves([&](const Move &) -> bool {
      has_escape = true;
      return false;  // No need to keep generating moves.
    });

    if (has_escape) {
      // Check.
      absl::StrAppend(&move_text, "+");
    } else {
      // Checkmate.
      absl::StrAppend(&move_text, "#");
    }
  }

  return move_text;
}

template <uint32_t kBoardSize>
ChessBoard<kBoardSize>::ChessBoard()
    : to_play_(Color::kWhite),
      ep_square_(InvalidSquare()),
      irreversible_move_counter_(0),
      move_number_(1),
      castling_rights_{{true, true}, {true, true}},
      zobrist_hash_(0) {
  board_.fill(kEmptyPiece);
}

template <uint32_t kBoardSize>
/*static*/ absl::optional<ChessBoard<kBoardSize>>
ChessBoard<kBoardSize>::BoardFromFEN(const std::string &fen) {
  /* An FEN string includes a board position, side to play, castling
   * rights, ep square, 50 moves clock, and full move number. In that order.
   *
   * Eg. start position is:
   * rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
   *
   * Board is described from rank 8 to rank 1, and files from a to h. Empty
   * squares are encoded as number of consecutive empty squares.
   *
   * Many FEN strings don't have the last two fields.
   */
  ChessBoard board;

  for (auto color : {Color::kBlack, Color::kWhite}) {
    for (auto dir : {CastlingDirection::kLeft, CastlingDirection::kRight}) {
      board.SetCastlingRight(color, dir, false);
    }
  }

  std::vector<std::string> fen_parts = absl::StrSplit(fen, ' ');

  if (fen_parts.size() != 6 && fen_parts.size() != 4) {
    std::cerr << "Invalid FEN: " << fen << std::endl;
    return std::nullopt;
  }

  std::string &piece_configuration = fen_parts[0];
  std::string &side_to_move = fen_parts[1];
  std::string &castling_rights = fen_parts[2];
  std::string &ep_square = fen_parts[3];

  // These are defaults if the FEN string doesn't have these fields.
  std::string fifty_clock = "0";
  std::string move_number = "1";

  if (fen_parts.size() == 6) {
    fifty_clock = fen_parts[4];
    move_number = fen_parts[5];
  }

  std::vector<std::string> piece_config_by_rank =
      absl::StrSplit(piece_configuration, '/');

  for (int8_t current_y = kBoardSize - 1; current_y >= 0; --current_y) {
    std::string &rank = piece_config_by_rank[kBoardSize - current_y - 1];
    int8_t current_x = 0;
    for (char c : rank) {
      if (current_x >= kBoardSize) {
        std::cerr << "Too many things on FEN rank: " << rank << std::endl;
        return std::nullopt;
      }

      if (c >= '1' && c <= '8') {
        current_x += c - '0';
      } else {
        auto piece_type = PieceTypeFromChar(c);
        if (!piece_type) {
          std::cerr << "Invalid piece type in FEN: " << c << std::endl;
          return std::nullopt;
        }

        Color color = isupper(c) ? Color::kWhite : Color::kBlack;
        board.set_square(Square{current_x, current_y},
                         Piece{color, *piece_type});

        ++current_x;
      }
    }
  }

  if (side_to_move == "b") {
    board.SetToPlay(Color::kBlack);
  } else if (side_to_move == "w") {
    board.SetToPlay(Color::kWhite);
  } else {
    std::cerr << "Invalid side to move in FEN: " << side_to_move << std::endl;
    return std::nullopt;
  }

  if (castling_rights.find('K') != std::string::npos) {
    board.SetCastlingRight(Color::kWhite, CastlingDirection::kRight, true);
  }

  if (castling_rights.find('Q') != std::string::npos) {
    board.SetCastlingRight(Color::kWhite, CastlingDirection::kLeft, true);
  }

  if (castling_rights.find('k') != std::string::npos) {
    board.SetCastlingRight(Color::kBlack, CastlingDirection::kRight, true);
  }

  if (castling_rights.find('q') != std::string::npos) {
    board.SetCastlingRight(Color::kBlack, CastlingDirection::kLeft, true);
  }

  if (ep_square != "-") {
    auto maybe_ep_square = SquareFromString(ep_square);
    if (!maybe_ep_square) {
      std::cerr << "Invalid en passant square in FEN: " << ep_square
                << std::endl;
      return std::nullopt;
    }
    board.SetEpSquare(*maybe_ep_square);
  }

  board.SetIrreversibleMoveCounter(std::stoi(fifty_clock));
  board.SetMovenumber(std::stoi(move_number));

  return board;
}

template <uint32_t kBoardSize>
Square ChessBoard<kBoardSize>::find(const Piece &piece) const {
  for (int8_t y = 0; y < kBoardSize; ++y) {
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Square sq{x, y};
      if (at(sq) == piece) {
        return sq;
      }
    }
  }

  return InvalidSquare();
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::GenerateLegalMoves(
    const MoveYieldFn &yield) const {
  auto king_square = find(Piece{to_play_, PieceType::kKing});

  GeneratePseudoLegalMoves([this, &king_square, &yield](const Move &move) {
    // See if the move is legal by applying, checking whether the king is
    // under attack, and undoing the move.
    // TODO: Optimize this.
    auto board_copy = *this;
    board_copy.ApplyMove(move);

    auto ks = at(move.from).type == PieceType::kKing ? move.to : king_square;

    if (board_copy.UnderAttack(ks, to_play_)) {
      return true;
    } else {
      return yield(move);
    }
  });
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::GeneratePseudoLegalMoves(
    const MoveYieldFn &yield) const {
  bool generating = true;

#define YIELD(move)     \
  if (!yield(move)) {   \
    generating = false; \
  }

  for (int8_t y = 0; y < kBoardSize && generating; ++y) {
    for (int8_t x = 0; x < kBoardSize && generating; ++x) {
      Square sq{x, y};
      auto &piece = at(sq);
      if (piece.type != PieceType::kEmpty && piece.color == to_play_) {
        switch (piece.type) {
          case PieceType::kKing:
            GenerateKingDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            GenerateCastlingDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece, PieceType::kEmpty, true));
                });
            break;
          case PieceType::kQueen:
            GenerateQueenDestinations_(
                sq, to_play_,
                [&yield, &sq, &piece, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kRook:
            GenerateRookDestinations_(
                sq, to_play_,
                [&yield, &sq, &piece, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kBishop:
            GenerateBishopDestinations_(
                sq, to_play_,
                [&yield, &sq, &piece, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kKnight:
            GenerateKnightDestinations_(
                sq, to_play_,
                [&yield, &sq, &piece, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kPawn:
            GeneratePawnDestinations_(
                sq, to_play_,
                [&yield, &sq, &piece, &generating](const Square &to) {
                  if (IsPawnPromotionRank(to)) {
                    YIELD(Move(sq, to, piece, PieceType::kQueen));
                    YIELD(Move(sq, to, piece, PieceType::kRook));
                    YIELD(Move(sq, to, piece, PieceType::kBishop));
                    YIELD(Move(sq, to, piece, PieceType::kKnight));
                  } else {
                    YIELD(Move(sq, to, piece));
                  }
                });
            GeneratePawnCaptureDestinations_(
                sq, to_play_, true, /* include enpassant */
                [&yield, &sq, &piece, &generating](const Square &to) {
                  if (IsPawnPromotionRank(to)) {
                    YIELD(Move(sq, to, piece, PieceType::kQueen));
                    YIELD(Move(sq, to, piece, PieceType::kRook));
                    YIELD(Move(sq, to, piece, PieceType::kBishop));
                    YIELD(Move(sq, to, piece, PieceType::kKnight));
                  } else {
                    YIELD(Move(sq, to, piece));
                  }
                });
            break;
          default:
            std::cerr << "Unknown piece type: " << static_cast<int>(piece.type)
                      << std::endl;
        }
      }
    }
  }

#undef YIELD
}

template <uint32_t kBoardSize>
bool ChessBoard<kBoardSize>::HasSufficientMaterial() const {
  // Try to detect these 4 conditions.
  // 1. K vs K
  // 2. K+B vs K
  // 3. K+N vs K
  // 4. K+B* vs K+B* (all bishops on same coloured squares)

  // Indexed by colour.
  int knights[2] = {0, 0};
  int dark_bishops[2] = {0, 0};
  int light_bishops[2] = {0, 0};

  for (int8_t y = 0; y < kBoardSize; ++y) {
    for (int8_t x = 0; x < kBoardSize; ++x) {
      const auto &piece = at(Square{x, y});
      // If we have a queen, rook, or pawn, we have sufficient material.
      // This is early exit for almost all positions. We check rooks first
      // because they tend to appear on the corners of boards.
      if (piece.color != Color::kEmpty) {
        if (piece.type == PieceType::kRook || piece.type == PieceType::kPawn ||
            piece.type == PieceType::kQueen) {
          return true;
        }

        // We don't care about kings.
        if (piece.type == PieceType::kKing) {
          continue;
        }

        if (piece.type == PieceType::kKnight) {
          ++knights[static_cast<size_t>(piece.color)];
        }

        if (piece.type == PieceType::kBishop) {
          bool is_dark = ((x + y) % 2 == 0);
          if (is_dark) {
            ++dark_bishops[static_cast<size_t>(piece.color)];
          } else {
            ++light_bishops[static_cast<size_t>(piece.color)];
          }
        }
      }
    }
  }

  // Having two knights allows helpmate.
  if (knights[0] > 1 || knights[1] > 1) {
    return true;
  }

  if (knights[0] == 1) {
    // If we have anything else, mate is possible.
    if (light_bishops[0] > 0 || dark_bishops[0] > 0) {
      return true;
    } else {
      // If one side only has a knight, the other side must have something (#3).
      return knights[1] > 0 || dark_bishops[1] > 0 || light_bishops[1] > 0;
    }
  }

  if (knights[1] == 1) {
    // If we have anything else, mate is possible.
    if (light_bishops[1] > 0 || dark_bishops[1] > 0) {
      return true;
    } else {
      // If one side only has a knight, the other side must have something (#3).
      return knights[0] > 0 || dark_bishops[0] > 0 || light_bishops[0] > 0;
    }
  }

  // Now we only have bishops and kings. We must have two bishops on opposite
  // coloured squares (from either side) to not be a draw.
  // This covers #1, #2, and #4.
  bool dark_bishop_exists = (dark_bishops[0] + dark_bishops[1]) > 0;
  bool light_bishop_exists = (light_bishops[0] + light_bishops[1]) > 0;
  return dark_bishop_exists && light_bishop_exists;
}

template <uint32_t kBoardSize>
absl::optional<Move> ChessBoard<kBoardSize>::ParseMove(
    const std::string &move) const {
  // First see if they are in the long form -
  // "anan" (eg. "e2e4") or "anana" (eg. "f7f8q")
  // SAN moves will never have this form because an SAN move that starts with
  // a lowercase letter must be a pawn move, and pawn moves will never require
  // rank disambiguation (meaning the second character will never be a number).
  auto lan_move = ParseLANMove(move);
  if (lan_move) {
    return lan_move;
  }

  auto san_move = ParseSANMove(move);
  if (san_move) {
    return san_move;
  }

  return std::nullopt;
}

template <uint32_t kBoardSize>
absl::optional<Move> ChessBoard<kBoardSize>::ParseSANMove(
    const std::string &move_str) const {
  std::string move = move_str;

  if (move.empty()) return std::nullopt;

  if (absl::StartsWith(move, "O-O-O")) {
    // Queenside / left castling.
    std::vector<Move> candidates;
    GenerateLegalMoves([&candidates](const Move &move) {
      if (move.is_castling && move.to.x == 2) {
        candidates.push_back(move);
      }
      return true;
    });
    if (candidates.size() == 1) return candidates[0];
    std::cerr << "Invalid O-O-O" << std::endl;
    return std::nullopt;
  }

  if (absl::StartsWith(move, "O-O")) {
    // Kingside / right castling.
    std::vector<Move> candidates;
    GenerateLegalMoves([&candidates](const Move &move) {
      if (move.is_castling && move.to.x == 6) {
        candidates.push_back(move);
      }
      return true;
    });
    if (candidates.size() == 1) return candidates[0];
    std::cerr << "Invalid O-O" << std::endl;
    return std::nullopt;
  }

  auto move_annotation = SplitAnnotations(move);
  move = move_annotation.first;
  SPIEL_CHECK_FALSE(move.empty());

  auto annotation = move_annotation.second;

  // A move starts with a single letter identifying the piece. This may be
  // omitted for pawns.
  PieceType piece_type = PieceType::kPawn;
  std::string pieces = "PNBRQK";
  if (pieces.find(move[0]) != std::string::npos) {
    auto maybe_piece_type = PieceTypeFromChar(move[0]);
    if (!maybe_piece_type) {
      std::cerr << "Invalid piece type: " << move[0] << std::endl;
      return std::nullopt;
    }
    piece_type = *maybe_piece_type;
    move = std::string(absl::ClippedSubstr(move, 1));
  }

  // A move always ends with the destination square.
  if (move.size() < 2) {
    std::cerr << "Missing destination square" << std::endl;
    return std::nullopt;
  }
  auto destination = std::string(absl::ClippedSubstr(move, move.size() - 2));
  move = move.substr(0, move.size() - 2);

  auto dest_file = ParseFile(destination[0]);
  auto dest_rank = ParseRank(destination[1]);

  if (!dest_file || !dest_rank) {
    std::cerr << "Failed to parse destination square: " << destination
              << std::endl;
    return std::nullopt;
  }

  Square destination_square{*dest_file, *dest_rank};

  // Captures are indicated by a 'x' immediately preceding the destination.
  // This is irrelevant for parsing, so we just drop it.
  if (!move.empty() && move[move.size() - 1] == 'x') {
    move = move.substr(0, move.size() - 1);
  }

  // If necessary, source rank and/or file are also included for
  // disambiguation.
  absl::optional<int8_t> source_file, source_rank;
  if (!move.empty()) {
    source_file = ParseFile(move[0]);
    if (source_file) {
      move = std::string(absl::ClippedSubstr(move, 1));
    }
  }
  if (!move.empty()) {
    source_rank = ParseRank(move[0]);
    if (source_rank) {
      move = std::string(absl::ClippedSubstr(move, 1));
    }
  }

  SPIEL_CHECK_TRUE(move.empty());

  // Pawn promations are annotated with =Q to indicate the promotion type.
  absl::optional<PieceType> promotion_type;
  if (!annotation.empty() && annotation[0] == '=') {
    SPIEL_CHECK_GE(annotation.size(), 2);
    auto maybe_piece = PieceTypeFromChar(annotation[1]);
    if (!maybe_piece) return absl::optional<Move>();
    promotion_type = maybe_piece;
  }

  std::vector<Move> candidates;
  GenerateLegalMoves([&candidates, destination_square, piece_type, source_file,
                      source_rank, promotion_type, this](const Move &move) {
    PieceType moving_piece_type = at(move.from).type;
    if (move.to == destination_square && moving_piece_type == piece_type &&
        (!source_file || move.from.x == *source_file) &&
        (!source_rank || move.from.y == *source_rank) &&
        (!promotion_type || move.promotion_type == *promotion_type)) {
      candidates.push_back(move);
    }
    return true;
  });

  if (candidates.size() == 1) return candidates[0];
  std::cerr << "expected exactly one matching move, got " << candidates.size()
            << std::endl;
  return absl::optional<Move>();
}

template <uint32_t kBoardSize>
absl::optional<Move> ChessBoard<kBoardSize>::ParseLANMove(
    const std::string &move) const {
  SPIEL_CHECK_FALSE(move.empty());

  // Long algebraic notation moves (of the variant we care about) are in one of
  // two forms -
  // "anan" (eg. "e2e4") or "anana" (eg. "f7f8q")
  if (move.size() == 4 || move.size() == 5) {
    if (move[0] < 'a' || move[0] >= ('a' + kBoardSize) || move[1] < '1' ||
        move[1] >= ('1' + kBoardSize) || move[2] < 'a' ||
        move[2] >= ('a' + kBoardSize) || move[3] < '1' ||
        move[3] >= ('1' + kBoardSize)) {
      return std::nullopt;
    }

    if (move.size() == 5 && move[4] != 'q' && move[4] != 'r' &&
        move[4] != 'b' && move[4] != 'n') {
      return std::nullopt;
    }

    auto from = SquareFromString(move.substr(0, 2));
    auto to = SquareFromString(std::string(absl::ClippedSubstr(move, 2, 2)));
    if (from && to) {
      absl::optional<PieceType> promotion_type;
      if (move.size() == 5) {
        promotion_type = PieceTypeFromChar(move[4]);
        if (!promotion_type) {
          std::cerr << "Invalid promotion type" << std::endl;
          return std::nullopt;
        }
      }

      std::vector<Move> candidates;
      GenerateLegalMoves(
          [&to, &from, &promotion_type, &candidates](const Move &move) {
            if (move.from == *from && move.to == *to &&
                (!promotion_type || (move.promotion_type == *promotion_type))) {
              candidates.push_back(move);
            }
            return true;
          });

      if (candidates.empty()) {
        std::cerr << "Illegal move - " << move << " on " << ToUnicodeString()
                  << std::endl;
        return Move();
      } else if (candidates.size() > 1) {
        std::cerr << "Multiple matches (is promotion type missing?) - " << move
                  << std::endl;
        return Move();
      }

      return candidates[0];
    }
  } else {
    return std::nullopt;
  }
  SpielFatalError("All conditionals failed; this is a bug.");
  return std::nullopt;
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::ApplyMove(const Move &move) {
  // Most moves are simple - we remove the moving piece from the original
  // square, and put it on the destination square, overwriting whatever was
  // there before, update the 50 move counter, and update castling rights.
  //
  // There are a few exceptions - castling, en passant, promotions, double pawn
  // pushes. They require special adjustments in addition to those things. We
  // do them after the basic apply move.

  Piece moving_piece = at(move.from);
  Piece destination_piece = at(move.to);

  // We have to do it in this order because in Chess960 the king can castle
  // in-place! That's the only possibility for move.from == move.to.
  set_square(move.from, kEmptyPiece);
  set_square(move.to, moving_piece);

  // Whether the move is irreversible for the purpose of the 50-moves rule. Note
  // that although castling (and losing castling rights) should be irreversible,
  // it is counted as reversible here.
  // Irreversible moves are pawn moves and captures. We don't have to make a
  // special case for en passant, since they are pawn moves anyways.
  bool irreversible = (moving_piece.type == PieceType::kPawn) ||
                      (destination_piece.type != PieceType::kEmpty);

  if (irreversible) {
    SetIrreversibleMoveCounter(0);
  } else {
    SetIrreversibleMoveCounter(IrreversibleMoveCounter() + 1);
  }

  // Castling rights can be lost in a few different ways -
  // 1. The king moves (loses both rights), including castling.
  // 2. A rook moves (loses the right on that side).
  // 3. Captures an opponent rook (OPPONENT loses the right on that side).
  if (moving_piece.type == PieceType::kKing) {
    SetCastlingRight(to_play_, CastlingDirection::kLeft, false);
    SetCastlingRight(to_play_, CastlingDirection::kRight, false);
  }
  if (moving_piece.type == PieceType::kRook) {
    // TODO(author12): Fix this for Chess960, which requires storing initial
    // positions of rooks.
    if ((to_play_ == Color::kWhite && move.from == Square{0, 0}) ||
        (to_play_ == Color::kBlack && move.from == Square{0, 7})) {
      SetCastlingRight(to_play_, CastlingDirection::kLeft, false);
    } else if ((to_play_ == Color::kWhite && move.from == Square{7, 0}) ||
               (to_play_ == Color::kBlack && move.from == Square{7, 7})) {
      SetCastlingRight(to_play_, CastlingDirection::kRight, false);
    }
  }
  if (destination_piece.type == PieceType::kRook) {
    if ((to_play_ == Color::kWhite && move.to == Square{0, 7}) ||
        (to_play_ == Color::kBlack && move.to == Square{0, 0})) {
      SetCastlingRight(OppColor(to_play_), CastlingDirection::kLeft, false);
    } else if ((to_play_ == Color::kWhite && move.to == Square{7, 7}) ||
               (to_play_ == Color::kBlack && move.to == Square{7, 0})) {
      SetCastlingRight(OppColor(to_play_), CastlingDirection::kRight, false);
    }
  }

  // Special cases that require adjustment -
  // 1. Castling
  if (move.is_castling) {
    SPIEL_CHECK_EQ(moving_piece.type, PieceType::kKing);
    // We can tell which side we are castling to using "to" square.
    if (to_play_ == Color::kWhite) {
      if (move.to == Square{2, 0}) {
        // left castle
        // TODO(author12): In Chess960, rooks can be anywhere, so delete the
        // correct squares.
        set_square(Square{0, 0}, kEmptyPiece);
        set_square(Square{2, 0}, Piece{Color::kWhite, PieceType::kKing});
        set_square(Square{3, 0}, Piece{Color::kWhite, PieceType::kRook});
      } else if (move.to == Square{6, 0}) {
        // right castle
        set_square(Square{7, 0}, kEmptyPiece);
        set_square(Square{6, 0}, Piece{Color::kWhite, PieceType::kKing});
        set_square(Square{5, 0}, Piece{Color::kWhite, PieceType::kRook});
      } else {
        std::cerr << "Trying to castle but destination is not valid."
                  << std::endl;
      }
    } else {
      if (move.to == Square{2, 7}) {
        // left castle
        set_square(Square{0, 7}, kEmptyPiece);
        set_square(Square{2, 7}, Piece{Color::kBlack, PieceType::kKing});
        set_square(Square{3, 7}, Piece{Color::kBlack, PieceType::kRook});
      } else if (move.to == Square{6, 7}) {
        // right castle
        set_square(Square{7, 7}, kEmptyPiece);
        set_square(Square{6, 7}, Piece{Color::kBlack, PieceType::kKing});
        set_square(Square{5, 7}, Piece{Color::kBlack, PieceType::kRook});
      } else {
        std::cerr << "Trying to castle but destination is not valid.";
      }
    }
  }

  // 2. En-passant
  if (moving_piece.type == PieceType::kPawn && move.from.x != move.to.x &&
      destination_piece.type == PieceType::kEmpty) {
    if (move.to != EpSquare()) {
      std::cerr << "We are trying to capture an empty square "
                << "with a pawn, but the square is not the en passant square:\n"
                << *this << "\n"
                << "Move: " << move.ToString() << std::endl;
      SpielFatalError("Trying to apply an invalid move");
    }
    Square captured_pawn_square = move.to;
    if (to_play_ == Color::kWhite) {
      --captured_pawn_square.y;
    } else {
      ++captured_pawn_square.y;
    }
    SPIEL_CHECK_EQ(at(captured_pawn_square),
                   (Piece{OppColor(to_play_), PieceType::kPawn}));
    set_square(captured_pawn_square, kEmptyPiece);
  }

  // 3. Promotions
  if (moving_piece.type == PieceType::kPawn && IsPawnPromotionRank(move.to)) {
    set_square(move.to, Piece{at(move.to).color, move.promotion_type});
  }

  // 4. Double push
  if (moving_piece.type == PieceType::kPawn &&
      abs(move.from.y - move.to.y) == 2) {
    SetEpSquare(Square{move.from.x,
                       static_cast<int8_t>((move.from.y + move.to.y) / 2)});
  } else {
    SetEpSquare(InvalidSquare());
  }

  if (to_play_ == Color::kBlack) {
    ++move_number_;
  }

  SetToPlay(OppColor(to_play_));
}

template <uint32_t kBoardSize>
bool ChessBoard<kBoardSize>::TestApplyMove(const Move &move) {
  Color color = to_play_;
  ApplyMove(move);
  return !UnderAttack(find(Piece{color, PieceType::kKing}), color);
}

template <uint32_t kBoardSize>
bool ChessBoard<kBoardSize>::UnderAttack(const Square &sq,
                                         Color our_color) const {
  SPIEL_CHECK_NE(sq, InvalidSquare());

  bool under_attack = false;
  Color opponent_color = OppColor(our_color);

  // We do this by pretending we are a piece of different types, and seeing if
  // we can attack opponent pieces. Eg. if we pretend we are a knight, and can
  // attack an opponent knight, that means the knight can also attack us.

  // King moves (this is possible because we use this function for checking
  // whether we are moving into check, and we can be trying to move the king
  // into a square attacked by opponent king).
  GenerateKingDestinations_(
      sq, our_color, [this, &under_attack, &opponent_color](const Square &to) {
        if (at(to) == Piece{opponent_color, PieceType::kKing}) {
          under_attack = true;
        }
      });
  if (under_attack) {
    return true;
  }

  // Rook moves (for rooks and queens)
  GenerateRookDestinations_(
      sq, our_color, [this, &under_attack, &opponent_color](const Square &to) {
        if ((at(to) == Piece{opponent_color, PieceType::kRook}) ||
            (at(to) == Piece{opponent_color, PieceType::kQueen})) {
          under_attack = true;
        }
      });
  if (under_attack) {
    return true;
  }

  // Bishop moves (for bishops and queens)
  GenerateBishopDestinations_(
      sq, our_color, [this, &under_attack, &opponent_color](const Square &to) {
        if ((at(to) == Piece{opponent_color, PieceType::kBishop}) ||
            (at(to) == Piece{opponent_color, PieceType::kQueen})) {
          under_attack = true;
        }
      });
  if (under_attack) {
    return true;
  }

  // Knight moves
  GenerateKnightDestinations_(
      sq, our_color, [this, &under_attack, &opponent_color](const Square &to) {
        if (at(to) == Piece{opponent_color, PieceType::kKnight}) {
          under_attack = true;
        }
      });
  if (under_attack) {
    return true;
  }

  // Pawn captures.
  GeneratePawnCaptureDestinations_(
      sq, our_color, false /* no ep */,
      [this, &under_attack, &opponent_color](const Square &to) {
        if (at(to) == Piece{opponent_color, PieceType::kPawn}) {
          under_attack = true;
        }
      });
  if (under_attack) {
    return true;
  }

  return false;
}

template <uint32_t kBoardSize>
std::string ChessBoard<kBoardSize>::DebugString() const {
  std::string s;
  s = absl::StrCat("FEN: ", ToFEN(), "\n");
  absl::StrAppend(&s, "\n  ---------------------------------\n");
  for (int8_t y = kBoardSize - 1; y >= 0; --y) {
    // Rank label.
    absl::StrAppend(&s, RankToString(y), " ");

    // Pieces on the rank.
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Square sq{x, y};
      absl::StrAppend(&s, "| ", at(sq).ToString(), " ");
    }
    absl::StrAppend(&s, "|\n");
    absl::StrAppend(&s, "  ---------------------------------\n");
  }

  // File labels.
  absl::StrAppend(&s, "    ");
  for (int8_t x = 0; x < kBoardSize; ++x) {
    absl::StrAppend(&s, FileToString(x), "   ");
  }
  absl::StrAppend(&s, "\n");

  absl::StrAppend(&s, "To play: ", to_play_ == Color::kWhite ? "W" : "B", "\n");
  absl::StrAppend(&s, "En passant square: ", SquareToString(EpSquare()), "\n");
  absl::StrAppend(&s, "50-moves clock: ", IrreversibleMoveCounter(), "\n");
  absl::StrAppend(&s, "Move number: ", move_number_, "\n\n");

  absl::StrAppend(&s, "Castling rights:\n");
  absl::StrAppend(&s, "White left (queen-side): ",
                  CastlingRight(Color::kWhite, CastlingDirection::kLeft), "\n");
  absl::StrAppend(&s, "White right (king-side): ",
                  CastlingRight(Color::kWhite, CastlingDirection::kRight),
                  "\n");
  absl::StrAppend(&s, "Black left (queen-side): ",
                  CastlingRight(Color::kBlack, CastlingDirection::kLeft), "\n");
  absl::StrAppend(&s, "Black right (king-side): ",
                  CastlingRight(Color::kBlack, CastlingDirection::kRight),
                  "\n");
  absl::StrAppend(&s, "\n");

  return s;
}

// King moves without castling.
template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateKingDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  static const std::array<Offset, 8> kOffsets = {
      {{1, 0}, {1, 1}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}}};

  for (const auto &offset : kOffsets) {
    Square dest = sq + offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateCastlingDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  // There are 8 conditions for castling -
  // 1. The rook involved must not have moved.
  // 2. The king must not have moved.
  // 3. The rook involved must still be alive.
  // 4. All squares king jumps over must be empty.
  // 5. All squares the rook jumps over must be empty.
  // 6. The squares the king jumps over must not be under attack.
  // 7. The king must not be in check.
  // (8). The square the king ends up in must not be under attack.
  //
  // We don't check for (8) here because this is not unique to castling, and
  // we will check for it later.
  //
  // We use the generalized definition of castling from Chess960, instead of
  // hard-coding starting squares.
  // By Chess960 rules, the king and rook end up in the same positions as in
  // standard chess, but they can start from any squares.
  //
  // Castling to one side doesn't necessarily mean the king will move towards
  // that side.
  // Eg.
  // |RK...R..| + long castle (to the left) =>
  // |..KR.R..|

  // Whether all squares between sq1 and sq2 exclusive are empty, and
  // optionally safe (not under attack).
  const auto check_squares_between = [this, &color](const Square &sq1,
                                                    const Square &sq2,
                                                    bool check_safe) -> bool {
    SPIEL_CHECK_EQ(sq1.y, sq2.y);

    if (sq1.x <= sq2.x) {
      for (Square test_square = sq1 + Offset{1, 0}; test_square != sq2;
           test_square += Offset{1, 0}) {
        if (!IsEmpty(test_square) ||
            (check_safe && UnderAttack(test_square, color))) {
          return false;
        }
      }
    } else {
      for (Square test_square = sq1 + Offset{-1, 0}; test_square != sq2;
           test_square += Offset{-1, 0}) {
        if (!IsEmpty(test_square) ||
            (check_safe && UnderAttack(test_square, color))) {
          return false;
        }
      }
    }

    return true;
  };

  const auto check_castling_conditions =
      [this, &sq, &color, &check_squares_between](int8_t direction) -> bool {
    // First we need to find the rook.
    Square rook_sq = sq + Offset{direction, 0};
    bool rook_found = false;

    // Yes, we do actually have to check colour -
    // https://github.com/official-stockfish/Stockfish/issues/356
    for (; InBoardArea(rook_sq); rook_sq.x += direction) {
      if (at(rook_sq) == Piece{color, PieceType::kRook}) {
        rook_found = true;
        break;
      }
    }

    if (!rook_found) {
      std::cerr << "Where did our rook go?" << *this << "\n"
                << "Square: " << SquareToString(sq) << std::endl;
      SpielFatalError("Rook not found");
    }

    static_assert(kBoardSize == 8,
                  "This is not boardsize-independent! What does castling "
                  "mean for other boardsizes?");
    int8_t rook_final_x = direction == -1 ? 3 /* d-file */ : 5 /* f-file */;
    Square rook_final_sq = Square{rook_final_x, sq.y};
    int8_t king_final_x = direction == -1 ? 2 /* c-file */ : 6 /* g-file */;
    Square king_final_sq = Square{king_final_x, sq.y};

    // 4. 5. 6. All squares the king and rook jump over, including the final
    // squares, must be empty. Squares king jumps over must additionally be
    // safe.
    if (!IsEmpty(rook_final_sq) || !IsEmpty(king_final_sq) ||
        !check_squares_between(rook_sq, rook_final_sq, false) ||
        !check_squares_between(sq, king_final_sq, true)) {
      return false;
    }

    return true;
  };

  // 1. 2. 3. Moving the king, moving the rook, or the rook getting captured
  // will reset the flag.
  bool can_left_castle = CastlingRight(color, CastlingDirection::kLeft) &&
                         check_castling_conditions(-1);
  bool can_right_castle = CastlingRight(color, CastlingDirection::kRight) &&
                          check_castling_conditions(1);

  if (can_left_castle || can_right_castle) {
    // 7. No castling to escape from check.
    if (UnderAttack(sq, color)) {
      return;
    }

    static_assert(kBoardSize == 8,
                  "This is not boardsize-independent! What does castling "
                  "mean for other boardsizes?");
    if (can_left_castle) {
      yield(Square{static_cast<int8_t>(2), sq.y});
    }

    if (can_right_castle) {
      yield(Square{static_cast<int8_t>(6), sq.y});
    }
  }
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateQueenDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  GenerateRookDestinations_(sq, color, yield);
  GenerateBishopDestinations_(sq, color, yield);
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateRookDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  GenerateRayDestinations_(sq, color, {1, 0}, yield);
  GenerateRayDestinations_(sq, color, {-1, 0}, yield);
  GenerateRayDestinations_(sq, color, {0, 1}, yield);
  GenerateRayDestinations_(sq, color, {0, -1}, yield);
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateBishopDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  GenerateRayDestinations_(sq, color, {1, 1}, yield);
  GenerateRayDestinations_(sq, color, {-1, 1}, yield);
  GenerateRayDestinations_(sq, color, {1, -1}, yield);
  GenerateRayDestinations_(sq, color, {-1, -1}, yield);
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateKnightDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  for (const auto &offset : kKnightOffsets) {
    Square dest = sq + offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

// Pawn moves without captures.
template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GeneratePawnDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  int8_t y_direction = color == Color::kWhite ? 1 : -1;
  Square dest = sq + Offset{0, y_direction};
  if (InBoardArea(dest) && IsEmpty(dest)) {
    yield(dest);

    // Test for double move.
    if (IsPawnStartingRank(sq, color)) {
      dest = sq + Offset{0, static_cast<int8_t>(2 * y_direction)};
      if (IsEmpty(dest)) {
        yield(dest);
      }
    }
  }
}

// Pawn capture destinations, with or without en passant.
template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GeneratePawnCaptureDestinations_(
    Square sq, Color color, bool include_ep, const YieldFn &yield) const {
  int8_t y_direction = color == Color::kWhite ? 1 : -1;
  Square dest = sq + Offset{1, y_direction};
  if (InBoardArea(dest) &&
      (IsEnemy(dest, color) || (include_ep && dest == EpSquare()))) {
    yield(dest);
  }

  dest = sq + Offset{-1, y_direction};
  if (InBoardArea(dest) &&
      (IsEnemy(dest, color) || (include_ep && dest == EpSquare()))) {
    yield(dest);
  }
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void ChessBoard<kBoardSize>::GenerateRayDestinations_(
    Square sq, Color color, Offset offset_step, const YieldFn &yield) const {
  for (Square dest = sq + offset_step; InBoardArea(dest); dest += offset_step) {
    if (IsEmpty(dest)) {
      yield(dest);
    } else if (IsEnemy(dest, color)) {
      yield(dest);
      break;
    } else {
      // We have a friendly piece.
      break;
    }
  }
}

template <uint32_t kBoardSize>
std::string ChessBoard<kBoardSize>::ToUnicodeString() const {
  std::string out = "\n";
  for (int8_t rank = kBoardSize - 1; rank >= 0; --rank) {
    out += std::to_string(rank + 1);
    for (int8_t file = 0; file < kBoardSize; ++file) {
      out += at(Square{file, rank}).ToUnicode();
    }
    out += "\n";
  }
  out += ' ';
  for (int8_t file = 0; file < kBoardSize; ++file) {
    out += ('a' + file);
  }
  out += '\n';
  return out;
}

template <uint32_t kBoardSize>
std::string ChessBoard<kBoardSize>::ToFEN() const {
  // Example FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
  std::string fen;

  // 1. encode the board.
  for (int8_t rank = kBoardSize - 1; rank >= 0; --rank) {
    int num_empty = 0;
    for (int8_t file = 0; file < kBoardSize; ++file) {
      auto piece = at(Square{file, rank});
      if (piece == kEmptyPiece) {
        ++num_empty;
      } else {
        if (num_empty > 0) {
          fen += std::to_string(num_empty);
          num_empty = 0;
        }
        fen += piece.ToString();
      }
    }
    if (num_empty > 0) {
      fen += std::to_string(num_empty);
    }
    if (rank > 0) {
      fen += "/";
    }
  }

  // 2. color to play.
  fen +=
      " " + (to_play_ == Color::kWhite ? std::string("w") : std::string("b"));

  // 3. by castling rights.
  fen += " ";
  std::string castling_rights;
  if (CastlingRight(Color::kWhite, CastlingDirection::kRight)) {
    castling_rights += "K";
  }
  if (CastlingRight(Color::kWhite, CastlingDirection::kLeft)) {
    castling_rights += "Q";
  }
  if (CastlingRight(Color::kBlack, CastlingDirection::kRight)) {
    castling_rights += "k";
  }
  if (CastlingRight(Color::kBlack, CastlingDirection::kLeft)) {
    castling_rights += "q";
  }
  fen += castling_rights.empty() ? std::string("-") : castling_rights;

  // 4. en passant square
  fen += " ";
  fen += (EpSquare() == InvalidSquare()) ? std::string("-")
                                         : SquareToString(EpSquare());

  // 5. half-move clock for 50-move rule
  fen += " " + std::to_string(irreversible_move_counter_);

  // 6. full-move clock
  fen += " " + std::to_string(move_number_);

  return fen;
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::set_square(Square sq, Piece piece) {
  static const ZobristTableU64<kBoardSize * kBoardSize, 3, 7> kZobristValues(
      /*seed=*/2765481);

  // First, remove the current piece from the hash.
  auto position = SquareToIndex_(sq);
  auto current_piece = at(sq);
  zobrist_hash_ ^=
      kZobristValues[position][static_cast<int>(current_piece.color)]
                    [static_cast<int>(current_piece.type)];

  // Then add the new piece
  zobrist_hash_ ^= kZobristValues[position][static_cast<int>(piece.color)]
                                 [static_cast<int>(piece.type)];

  board_[position] = piece;
}

template <uint32_t kBoardSize>
bool ChessBoard<kBoardSize>::CastlingRight(Color side,
                                           CastlingDirection direction) const {
  switch (direction) {
    case CastlingDirection::kLeft:
      return castling_rights_[ToInt(side)].left_castle;
    case CastlingDirection::kRight:
      return castling_rights_[ToInt(side)].right_castle;
    default:
      SpielFatalError("Unknown direction.");
      return -1;
  }
}

int ToInt(CastlingDirection direction) {
  switch (direction) {
    case CastlingDirection::kLeft:
      return 0;
    case CastlingDirection::kRight:
      return 1;
    default:
      SpielFatalError("Unknown direction.");
      return -1;
  }
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::SetCastlingRight(Color side,
                                              CastlingDirection direction,
                                              bool can_castle) {
  static const ZobristTableU64<2, 2, 2> kZobristValues(/*seed=*/876387212);

  // Remove old value from hash.
  zobrist_hash_ ^= kZobristValues[ToInt(side)][ToInt(direction)]
                                 [CastlingRight(side, direction)];

  // Then add the new value.
  zobrist_hash_ ^= kZobristValues[ToInt(side)][ToInt(direction)][can_castle];
  switch (direction) {
    case CastlingDirection::kLeft:
      castling_rights_[ToInt(side)].left_castle = can_castle;
      break;
    case CastlingDirection::kRight:
      castling_rights_[ToInt(side)].right_castle = can_castle;
      break;
  }
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::SetToPlay(Color c) {
  static const ZobristTableU64<2> kZobristValues(/*seed=*/284628);

  // Remove old color and add new to play.
  zobrist_hash_ ^= kZobristValues[ToInt(to_play_)];
  zobrist_hash_ ^= kZobristValues[ToInt(c)];
  to_play_ = c;
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::SetIrreversibleMoveCounter(int c) {
  irreversible_move_counter_ = c;
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::SetMovenumber(int move_number) {
  move_number_ = move_number;
}

template <uint32_t kBoardSize>
void ChessBoard<kBoardSize>::SetEpSquare(Square sq) {
  static const ZobristTableU64<kBoardSize, kBoardSize> kZobristValues(
      /*seed=*/837261);

  if (EpSquare() != InvalidSquare()) {
    // Remove en passant square if there was one.
    zobrist_hash_ ^= kZobristValues[EpSquare().x][EpSquare().y];
  }
  if (sq != InvalidSquare()) {
    zobrist_hash_ ^= kZobristValues[sq.x][sq.y];
  }

  ep_square_ = sq;
}

// Explicit instantiations for all board sizes we care about.
template class ChessBoard<8>;

StandardChessBoard MakeDefaultBoard() {
  auto maybe_board = StandardChessBoard::BoardFromFEN(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}

}  // namespace chess
}  // namespace open_spiel
