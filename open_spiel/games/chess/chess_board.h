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

#ifndef OPEN_SPIEL_GAMES_IMPL_CHESS_CHESS_BOARD_H_
#define OPEN_SPIEL_GAMES_IMPL_CHESS_CHESS_BOARD_H_

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chess {

using chess_common::kInvalidSquare;  // NOLINT
using chess_common::Offset;
using chess_common::Square;
using chess_common::SquareToString;  // NOLINT

template <std::size_t... Dims>
using ZobristTableU64 = chess_common::ZobristTable<uint64_t, Dims...>;

enum class Color : int8_t { kBlack = 0, kWhite = 1, kEmpty = 2 };

inline int ToInt(Color color) { return color == Color::kWhite ? 1 : 0; }

inline Color OppColor(Color color) {
  return color == Color::kWhite ? Color::kBlack : Color::kWhite;
}

std::string ColorToString(Color c);

inline std::ostream& operator<<(std::ostream& stream, Color c) {
  return stream << ColorToString(c);
}

enum class CastlingDirection { kLeft, kRight };

int ToInt(CastlingDirection dir);

enum class PieceType : int8_t {
  kEmpty = 0,
  kKing = 1,
  kQueen = 2,
  kRook = 3,
  kBishop = 4,
  kKnight = 5,
  kPawn = 6
};

static inline constexpr std::array<PieceType, 6> kPieceTypes = {
    {PieceType::kKing, PieceType::kQueen, PieceType::kRook, PieceType::kBishop,
     PieceType::kKnight, PieceType::kPawn}};

// In case all the pieces are represented in the same plane, these values are
// used to represent each piece type.
static inline constexpr std::array<float, 6> kPieceRepresentation = {
    {1, 0.8, 0.6, 0.4, 0.2, 0.1}};

// Tries to parse piece type from char ('K', 'Q', 'R', 'B', 'N', 'P').
// Case-insensitive.
absl::optional<PieceType> PieceTypeFromChar(char c);

// Converts piece type to one character strings - "K", "Q", "R", "B", "N", "P".
// p must be one of the enumerator values of PieceType.
std::string PieceTypeToString(PieceType p, bool uppercase = true);

struct Piece {
  bool operator==(const Piece& other) const {
    return type == other.type && color == other.color;
  }

  bool operator!=(const Piece& other) const { return !(*this == other); }

  std::string ToUnicode() const;
  std::string ToString() const;

  Color color;
  PieceType type;
};

static inline constexpr Piece kEmptyPiece =
    Piece{Color::kEmpty, PieceType::kEmpty};

inline std::ostream& operator<<(std::ostream& stream, const Piece& p) {
  return stream << p.ToString();
}

inline absl::optional<int8_t> ParseRank(char c) {
  if (c >= '1' && c <= '8') return c - '1';
  return absl::nullopt;
}

inline absl::optional<int8_t> ParseFile(char c) {
  if (c >= 'a' && c <= 'h') return c - 'a';
  return absl::nullopt;
}

// Maps y = [0, 7] to rank ["1", "8"].
inline std::string RankToString(int8_t rank) {
  return std::string(1, '1' + rank);
}

// Maps x = [0, 7] to file ["a", "h"].
inline std::string FileToString(int8_t file) {
  return std::string(1, 'a' + file);
}

// Offsets for all possible knight moves.
inline constexpr std::array<Offset, 8> kKnightOffsets = {
    {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {2, -1}, {2, 1}, {1, -2}, {1, 2}}};

absl::optional<Square> SquareFromString(const std::string& s);

bool IsLongDiagonal(const chess::Square& from_sq, const chess::Square& to_sq,
                    int board_size);

// Forward declare ChessBoard here because it's needed in Move::ToSAN.
class ChessBoard;

struct Move {
  Square from;
  Square to;
  Piece piece;
  PieceType promotion_type;

  // We have to record castling here, because in Chess960 we may not be able to
  // tell just from "from" and "to" squares.
  bool is_castling = false;

  Move() : is_castling(false) {}
  Move(const Square& from, const Square& to, const Piece& piece,
       PieceType promotion_type = PieceType::kEmpty, bool is_castling = false)
      : from(from),
        to(to),
        piece(piece),
        promotion_type(promotion_type),
        is_castling(is_castling) {}

  std::string ToString() const;

  // Converts to long algebraic notation, as required by the UCI protocol.
  std::string ToLAN() const;

  // Converts to standard algebraic notation, as required by portable game
  // notation (PGN). It is a chess move notation that is designed to be
  // human-readable and concise.
  //
  // Unlike the LAN format, generating a SAN string requires the board the move
  // is generated from.
  //
  // There are 3 types of SAN moves -
  // 1. O-O (short castle)
  // 2. O-O-O (long castle)
  // 3. [piece type][from file][from rank][x][to square][=Promo][annotations]
  //
  // [piece type] is omitted for pawns
  // [from file] is only included if 1) move is a pawn capture, or 2) it's
  //     required for disambiguation (see below).
  // [from rank] is only included if it's required for disambiguation.
  // [x] is only included for captures
  // [to square] is always included
  // [=Promo] is only included for promotions ("=N", "=B", "=R", "=Q" depending
  //     on type promoting to).
  // [annotations] are a list of 0 or more characters added with different
  //     meanings. The ones we care about are '+' for check, and '#' for
  //     checkmate. All others are optional.
  //
  // Disambiguation:
  // If a move is not uniquely-identified otherwise, file and/or rank of the
  // from square is inserted to disambiguate. When either one will disambiguate,
  // file should be used. If file is unique, file is used. Otherwise if rank is
  // unique, rank is used. If neither is unique (this happens rarely, usually
  // after under-promoting to a minor piece with both original pieces still
  // intact, or double queen promotions with original queen still intact), both
  // are used.
  //
  // Examples:
  // * e4 (pawn to e4)
  // * exd5 (pawn on file e capture the piece on d5)
  // * Nf3 (knight to f3)
  // * Nxd5 (knight captures piece on d5)
  // * Bed5 (bishop on file e to d5)
  // * B5xc3 (bishop on rank 5 capture piece on c3)
  // * Ne5f7 (knight on e5 to f7, when there are 3 knights on the board, one on
  //          e file, and one on 5th rank)
  // * exd8=N#!! (pawn on e file capture piece on d8 and promote to knight
  //              resulting in checkmate in a surprisingly good move)
  // * O-O-O!!N+/- (a surprisingly good long castle that is a theoretical
  //                novelty that gives white a clear but not winning advantage)
  std::string ToSAN(const ChessBoard& board) const;

  bool operator==(const Move& other) const {
    return from == other.from && to == other.to && piece == other.piece &&
           promotion_type == other.promotion_type &&
           is_castling == other.is_castling;
  }
};

inline std::ostream& operator<<(std::ostream& stream, const Move& m) {
  return stream << m.ToString();
}

bool IsMoveCharacter(char c);

std::pair<std::string, std::string> SplitAnnotations(const std::string& move);

inline constexpr int kMaxBoardSize = 8;
inline constexpr int kDefaultBoardSize = 8;
inline constexpr int k2dMaxBoardSize = kMaxBoardSize * kMaxBoardSize;
inline const std::string kDefaultStandardFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
inline const std::string kDefaultSmallFEN = "r1kr/pppp/PPPP/R1KR w - - 0 1";

using ObservationTable = std::array<bool, k2dMaxBoardSize>;

// Specifies policy for pseudo legal moves generation.
enum PseudoLegalMoveSettings {
  // Standard legal moves (do not allow to move past enemy pieces).
  kAcknowledgeEnemyPieces,
  // Pseudo-legal moves, where a piece can move anywhere (according to the rules
  // for that piece), except if it was blocked from doing so by other player's
  // pieces. This is used in games, where the player may not know the position
  // of an enemy piece (like Kriegspiel or RBC) and it can try to move past the
  // enemy (for example a rook can try to move the other side of the board, even
  // if it is in fact blocked by an unseen opponent's pawn).
  kBreachEnemyPieces,
};

// Some chess variants (RBC) allow a "pass" action/move
inline constexpr open_spiel::Action kPassAction = 0;
inline const chess::Move kPassMove =
    Move(Square{-1, -1}, Square{-1, -1},
         Piece{Color::kEmpty, PieceType::kEmpty});

class ChessBoard {
 public:
  ChessBoard(int board_size = kDefaultBoardSize,
             bool king_in_check_allowed = false,
             bool allow_pass_move = false);

  // Constructs a chess board at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  static absl::optional<ChessBoard> BoardFromFEN(
      const std::string& fen, int board_size = 8,
      bool king_in_check_allowed = false,
      bool allow_pass_move = false);

  const Piece& at(Square sq) const { return board_[SquareToIndex_(sq)]; }

  void set_square(Square sq, Piece p);

  const std::array<Piece, k2dMaxBoardSize>& pieces() const { return board_; }

  Color ToPlay() const { return to_play_; }
  void SetToPlay(Color c);

  Square EpSquare() const { return ep_square_; }
  void SetEpSquare(Square sq);

  int32_t IrreversibleMoveCounter() const { return irreversible_move_counter_; }
  int32_t Movenumber() const { return move_number_; }

  bool CastlingRight(Color side, CastlingDirection direction) const;
  void SetCastlingRight(Color side, CastlingDirection direction,
                        bool can_castle);

  // Find the location of any one piece of the given type, or kInvalidSquare.
  Square find(const Piece& piece) const;

  // Pseudo-legal moves are moves that may leave the king in check, but are
  // otherwise legal.
  // The generation functions call yield(move) for each move generated.
  // The yield function should return whether generation should continue.
  // For performance reasons, we do not guarantee that no more moves will be
  // generated if yield returns false. It is only for optimization.
  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const {
    GenerateLegalMoves(yield, to_play_);
  }
  void GenerateLegalMoves(const MoveYieldFn& yield, Color color) const;
  void GeneratePseudoLegalMoves(
      const MoveYieldFn& yield, Color color,
      PseudoLegalMoveSettings settings =
          PseudoLegalMoveSettings::kAcknowledgeEnemyPieces) const;

  // Optimization for computing number of pawn tries for kriegspiel
  void GenerateLegalPawnCaptures(const MoveYieldFn& yield, Color color) const;
  void GeneratePseudoLegalPawnCaptures(
      const MoveYieldFn& yield, Color color,
      PseudoLegalMoveSettings settings =
          PseudoLegalMoveSettings::kAcknowledgeEnemyPieces) const;

  bool HasLegalMoves() const {
    bool found = false;
    GenerateLegalMoves([&found](const Move&) {
      found = true;
      return false;  // We don't need any more moves.
    });
    return found;
  }

  bool IsMoveLegal(const Move& tested_move) const {
    bool found = false;
    GenerateLegalMoves([&found, &tested_move](const Move& found_move) {
      if (tested_move == found_move) {
        found = true;
        return false;  // We don't need any more moves.
      }
      return true;
    });
    return found;
  }

  // Does either side have sufficient material to mate?
  // FIDE rules say it must be impossible to mate even with "most unskilled"
  // counterplay. This would technically include things like pawns blocking
  // either side from making progress.
  // Eg. "8/4k3/8/p1p1p1p1/P1P1P1P1/8/4K3/8 w - -".
  // However, detecting all such positions will require solving chess... so
  // we detect a more generally-accepted subset of positions - those with the
  // following material combinations:
  // 1. K vs K
  // 2. K+B vs K
  // 3. K+N vs K
  // 4. K+B* vs K+B* (all bishops on same coloured squares)
  bool HasSufficientMaterial() const;

  // Parses a move in standard algebraic notation or long algebraic notation
  // (see below). Returns absl::nullopt on failure.
  absl::optional<Move> ParseMove(const std::string& move) const;

  // Parses a move in standard algebraic notation as defined by FIDE.
  // https://en.wikipedia.org/wiki/Algebraic_notation_(chess).
  // Returns absl::nullopt on failure.
  absl::optional<Move> ParseSANMove(const std::string& move) const;

  // Parses a move in long algebraic notation.
  // Long algebraic notation is not standardized and there are many variants,
  // but the one we care about is of the form "e2e4" and "f7f8q". This is the
  // form used by chess engine text protocols that are of interest to us.
  // Returns absl::nullopt on failure.
  absl::optional<Move> ParseLANMove(const std::string& move) const;

  void ApplyMove(const Move& move);

  // Applies a pseudo-legal move and returns whether it's legal. This avoids
  // applying and copying the whole board once for legality testing, and once
  // for actually applying the move.
  bool TestApplyMove(const Move& move);

  bool InBoardArea(const Square& sq) const {
    return sq.x >= 0 && sq.x < board_size_ && sq.y >= 0 && sq.y < board_size_;
  }

  bool IsEmpty(const Square& sq) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.type == PieceType::kEmpty;
  }

  bool IsEnemy(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.type != PieceType::kEmpty && piece.color != our_color;
  }

  bool IsFriendly(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.color == our_color;
  }

  bool IsEmptyOrEnemy(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.color != our_color;
  }

  /* Whether the square is on the pawn starting rank for our_color. */
  bool IsPawnStartingRank(const Square& sq, Color our_color) const {
    return ((our_color == Color::kWhite && sq.y == 1) ||
            (our_color == Color::kBlack && sq.y == (board_size_ - 2)));
  }

  bool IsPawnPromotionRank(const Square& sq) const {
    // No need to test for color here because a pawn can't be on the "wrong"
    // promotion rank.
    return sq.y == 0 || sq.y == (board_size_ - 1);
  }

  /* Whether the sq is under attack by the opponent. */
  bool UnderAttack(const Square& sq, Color our_color) const;

  bool InCheck() const {
    return UnderAttack(find(Piece{to_play_, PieceType::kKing}), to_play_);
  }

  int BoardSize() const { return board_size_; }

  bool KingInCheckAllowed() const { return king_in_check_allowed_; }

  bool AllowPassMove() const { return allow_pass_move_; }

  uint64_t HashValue() const { return zobrist_hash_; }

  std::string DebugString() const;

  std::string ToUnicodeString() const;

  // Constructs a string describing the chess board position in Forsyth-Edwards
  // Notation. https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  std::string ToFEN() const;

  /* Constructs a string describing the dark chess board position in a notation
   * similar to Forsyth-Edwards Notation.
   * https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
   *
   * There are several key differences to FEN:
   * 1) Only observable squares are shown. Squares invisible to the observer are
   *    represented by the character '?'
   * 2) Only observer's castling rights are shown
   * 3) en passant square is only shown if the observer is capable of performing
   *    an en passant capture
   *
   */
  std::string ToDarkFEN(const ObservationTable& observability_table,
                        Color color) const;

  bool IsBreachingMove(Move move) const;
  void BreachingMoveToCaptureMove(Move* move) const;

 private:
  size_t SquareToIndex_(Square sq) const { return sq.y * board_size_ + sq.x; }

  /* Generate*Destinations functions call yield(sq) for every potential
   * destination generated.
   * Eg.
   * std::vector<Move> knight_moves;
   * board.GenerateKnightDestinations(Square{3, 3},
   *                                  [](const Square& sq) {
   *                                    Move move{Square{3, 3}, sq,
   *                                              Piece{kWhite, kKnight}};
   *                                    knight_moves.push_back(move);
   *                                  });
   */

  /* All the Generate*Destinations functions work in the same slightly strange
   * way -
   * They assume there's a piece of the type in question at sq, and generate
   * potential destinations. Potential destinations may include moves that
   * will leave the king exposed, and are therefore illegal.
   * This strange semantic is to support reusing these functions for checking
   * whether one side is in check, which would otherwise require an almost-
   * duplicate move generator.
   */

  // King moves without castling.
  template <typename YieldFn>
  void GenerateKingDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateCastlingDestinations_(Square sq, Color color,
                                     PseudoLegalMoveSettings settings,
                                     const YieldFn& yield) const;
  bool CanCastle(Square king_sq, Color color,
                 PseudoLegalMoveSettings settings) const;
  bool CanCastleBetween(Square sq1, Square sq2,
                        bool check_safe_from_opponent,
                        PseudoLegalMoveSettings settings) const;

  template <typename YieldFn>
  void GenerateQueenDestinations_(Square sq, Color color,
                                  PseudoLegalMoveSettings settings,
                                  const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateRookDestinations_(Square sq, Color color,
                                 PseudoLegalMoveSettings settings,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateBishopDestinations_(Square sq, Color color,
                                   PseudoLegalMoveSettings settings,
                                   const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateKnightDestinations_(Square sq, Color color,
                                   const YieldFn& yield) const;

  template <typename YieldFn>
  // Pawn moves without captures.
  void GeneratePawnDestinations_(Square sq, Color color,
                                 PseudoLegalMoveSettings settings,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  // Pawn diagonal capture destinations, with or without en passant.
  void GeneratePawnCaptureDestinations_(Square sq, Color color,
                                        PseudoLegalMoveSettings settings,
                                        bool include_ep,
                                        const YieldFn& yield) const;

  // Helper function.
  template <typename YieldFn>
  void GenerateRayDestinations_(Square sq, Color color,
                                PseudoLegalMoveSettings settings,
                                Offset offset_step, const YieldFn& yield) const;

  void SetIrreversibleMoveCounter(int c);
  void SetMovenumber(int move_number);

  int board_size_;
  bool king_in_check_allowed_;
  bool allow_pass_move_;

  std::array<Piece, k2dMaxBoardSize> board_;
  Color to_play_;
  Square ep_square_;
  int32_t irreversible_move_counter_;

  // This starts at 1, and increments after each black move (a "full move" in
  // chess is a "half move" by white followed by a "half move" by black).
  int32_t move_number_;

  struct {
    bool left_castle;   // -x direction, AKA long castle
    bool right_castle;  // +x direction, AKA short castle
  } castling_rights_[2];

  uint64_t zobrist_hash_;
};

inline std::ostream& operator<<(std::ostream& stream, const ChessBoard& board) {
  return stream << board.DebugString();
}

inline std::ostream& operator<<(std::ostream& stream, const PieceType& pt) {
  return stream << PieceTypeToString(pt);
}

ChessBoard MakeDefaultBoard();
std::string DefaultFen(int board_size);

}  // namespace chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_IMPL_CHESS_CHESS_BOARD_H_
