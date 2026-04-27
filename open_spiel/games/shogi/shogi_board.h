// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_SHOGI_SHOGI_BOARD_H_
#define OPEN_SPIEL_GAMES_SHOGI_SHOGI_BOARD_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace shogi {

constexpr int kBoardSize = 9;
constexpr int kNumSquares = 81;
constexpr int kNumBoardMoves = 81 * 81 * 2;
constexpr int kNumDropMoves = 7 * 81;
constexpr int kNumPieceTypes = 14;  // not counting kempty
constexpr int kNumPocketPieces = 7;

struct Offset {
  int8_t x_offset;
  int8_t y_offset;

  bool operator==(const Offset& other) const {
    return x_offset == other.x_offset && y_offset == other.y_offset;
  }
};

// Shogi coordinates are twisted and backwards from chess:
// files (columns)  are numbers with 9 on the left
// ranks (rows) are letters with 'i' at the bottom
// black (first player, sente) is going 'up'
// The number (column) is written first.

inline absl::optional<int8_t> ParseRank(char c) {
  if (c >= 'a' && c <= 'i') return 'i' - c;
  return absl::nullopt;
}

inline absl::optional<int8_t> ParseFile(char c) {
  if (c >= '1' && c <= '9') return '9' - c;
  return absl::nullopt;
}

// Maps y = [0, 8] to rank ["1", "9"].
inline std::string RankToString(int8_t rank) {
  return std::string(1, 'i' - rank);
}

// Maps x = [0, 8] to file ["a", "i"].
inline std::string FileToString(int8_t file) {
  return std::string(1, '9' - file);
}

// x corresponds to file (column / letter)
// y corresponds to rank (row / number).
struct Square {
  int8_t x;
  int8_t y;

  Square& operator+=(const Offset& offset) {
    x = static_cast<int8_t>(x + offset.x_offset);
    y = static_cast<int8_t>(y + offset.y_offset);
    return *this;
  }

  bool operator==(const Square& other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const Square& other) const { return !(*this == other); }

  // Required by std::set.
  bool operator<(const Square& other) const {
    if (x != other.x) {
      return x < other.x;
    } else {
      return y < other.y;
    }
  }

  std::string ToString() const { return FileToString(x) + RankToString(y); }

  int Index() const { return y * kBoardSize + x; }
};

inline Square operator+(const Square& sq, const Offset& offset) {
  int8_t x = sq.x + offset.x_offset;
  int8_t y = sq.y + offset.y_offset;
  return Square{x, y};
}

constexpr Square kInvalidSquare{-1, -1};

inline std::ostream& operator<<(std::ostream& stream, const Square& sq) {
  return stream << sq.ToString();
}

absl::optional<Square> SquareFromString(const std::string& s);

template <typename T, std::size_t InnerDim, std::size_t... OtherDims>
class ZobristTable {
 public:
  using Generator = std::mt19937_64;
  using NestedTable = ZobristTable<T, OtherDims...>;

  explicit ZobristTable(Generator::result_type seed) {
    Generator generator(seed);
    absl::uniform_int_distribution<Generator::result_type> dist;
    data_.reserve(InnerDim);
    for (std::size_t i = 0; i < InnerDim; ++i) {
      data_.emplace_back(dist(generator));
    }
  }

  const NestedTable& operator[](std::size_t inner_index) const {
    return data_[inner_index];
  }

 private:
  std::vector<NestedTable> data_;
};

// 1-dimensional array of uniform random numbers.
template <typename T, std::size_t InnerDim>
class ZobristTable<T, InnerDim> {
 public:
  using Generator = std::mt19937_64;

  explicit ZobristTable(Generator::result_type seed) : data_(InnerDim) {
    Generator generator(seed);
    absl::uniform_int_distribution<T> dist;
    for (auto& field : data_) {
      field = dist(generator);
    }
  }

  T operator[](std::size_t index) const { return data_[index]; }

 private:
  std::vector<T> data_;
};

template <std::size_t... Dims>
using ZobristTableU64 = ZobristTable<uint64_t, Dims...>;

enum class Color : int8_t { kBlack = 0, kWhite = 1, kEmpty = 2 };

inline int ToInt(Color color) { return color == Color::kWhite ? 1 : 0; }

inline int8_t Forward(Color color) { return color == Color::kBlack ? 1 : -1; }

inline Color OppColor(Color color) {
  return color == Color::kWhite ? Color::kBlack : Color::kWhite;
}

std::string ColorToString(Color c);

inline std::ostream& operator<<(std::ostream& stream, Color c) {
  return stream << ColorToString(c);
}

enum class PieceType : int8_t {
  kEmpty = 0,
  kKing = 1,
  kLance = 2,
  kKnight = 3,
  kSilver = 4,
  kGold = 5,
  kPawn = 6,
  kBishop = 7,
  kRook = 8,
  kLanceP = 9,
  kKnightP = 10,
  kSilverP = 11,
  kPawnP = 12,
  kBishopP = 13,
  kRookP = 14
};

static inline constexpr std::array<PieceType, kNumPieceTypes> kPieceTypes = {
    {PieceType::kKing, PieceType::kLance, PieceType::kKnight,
     PieceType::kSilver, PieceType::kGold, PieceType::kPawn, PieceType::kBishop,
     PieceType::kRook, PieceType::kLanceP, PieceType::kKnightP,
     PieceType::kSilverP, PieceType::kPawnP, PieceType::kBishopP,
     PieceType::kRookP}};

PieceType PromotedType(PieceType type);
PieceType UnpromotedType(PieceType type);
bool IsPromoted(PieceType type);
int PieceValue(PieceType pt);

// Case-insensitive.
absl::optional<PieceType> PieceTypeFromChar(char c);

// one character for unpromoted types, appen + for promoted
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

struct Move {
  Square from;
  Square to;
  Piece piece;
  bool promote = false;
  bool drop = false;

  Move() {}

  Move(const Square& from, const Square& to, const Piece& piece,
       bool promote = false, bool drop = false)
      : from(from), to(to), piece(piece), promote(promote), drop(drop) {}

  std::string ToString() const;
  bool IsDropMove() const { return drop; }

  bool operator==(const Move& other) const {
    return from == other.from && to == other.to && piece == other.piece &&
           promote == other.promote && drop == other.drop;
  }
};

inline std::ostream& operator<<(std::ostream& stream, const Move& m) {
  return stream << m.ToString();
}

bool IsMoveCharacter(char c);

std::pair<std::string, std::string> SplitAnnotations(const std::string& move);

inline const std::string kDefaultStandardSFEN =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
using ObservationTable = std::array<bool, kNumSquares>;

class Pocket {
 public:
  // Iteration support
  static constexpr std::array<PieceType, kNumPocketPieces> PieceTypes() {
    return {PieceType::kPawn,   PieceType::kLance, PieceType::kKnight,
            PieceType::kSilver, PieceType::kGold,  PieceType::kBishop,
            PieceType::kRook};
  }

  // Modifiers
  void Increment(PieceType piece);
  void Decrement(PieceType piece);

  // Accessor
  int Count(PieceType piece) const;

  // Move encoding
  static int Index(PieceType ptype);

  static PieceType PocketPieceType(int index);

  bool Empty() const;

 private:
  // Internal storage: Pawn, Lance
  std::array<int, kNumPocketPieces> counts_{};
};

class ShogiBoard {
 public:
  // Constructs a shogi board at the given position with the given SFEN.
  ShogiBoard();
  static absl::optional<ShogiBoard> BoardFromSFEN(const std::string& fen);

  const Piece& at(Square sq) const { return board_[SquareToIndex_(sq)]; }

  void set_square(Square sq, Piece p);

  const std::array<Piece, kNumSquares>& pieces() const { return board_; }

  Color ToPlay() const { return to_play_; }
  void SetToPlay(Color c);

  int32_t Movenumber() const { return move_number_; }

  void AddToPocket(Color owner, PieceType piece);

  void RemoveFromPocket(Color owner, PieceType piece);

  // Find the location of any one piece of the given type, or kInvalidSquare.
  Square find(const Piece& piece) const;

  using MoveYieldFn = std::function<bool(const Move&)>;
  void GeneratePseudoLegalMoves(const MoveYieldFn& yield, Color color,
                                bool skip_drops = false) const;

  void GenerateLegalMoves(const MoveYieldFn& yield,
                          bool skip_drops = false) const {
    GenerateLegalMoves(yield, to_play_, skip_drops);
  }

  void GenerateLegalMoves(const MoveYieldFn& yield, Color color,
                          bool skip_drops = false) const;

  Pocket white_pocket_;  // counts of pocket pieces by type
  Pocket black_pocket_;

  bool HasLegalMoves(bool skip_drops = false) const {
    bool found = false;
    GenerateLegalMoves(
        [&found](const Move&) {
          found = true;
          return false;  // We don't need any more moves.
        },
        skip_drops);
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

  // Parses a move in standard algebraic notation or long algebraic notation
  // (see below). Returns absl::nullopt on failure.
  absl::optional<Move> ParseMove(const std::string& move) const;

  // We first check for a drop move with syntax like N@d4
  // All drop moves are shown with a drop syntax, so Nd4 always mean a knight
  // on the board moved.
  absl::optional<Move> ParseDropMove(const std::string& move) const;
  // Parses a move in standard algebraic notation.
  // Ranks are letters with a at the top and i at te bottom
  // Files are numbers with 1 at the right.
  // Drops are signified by a piece letter and a *
  // Returns absl::nullopt on failure.

  absl::optional<Move> ParseLANMove(const std::string& move) const;

  void ApplyMove(const Move& move);

  // Applies a pseudo-legal move and returns whether it's legal. This avoids
  // applying and copying the whole board once for legality testing, and once
  // for actually applying the move.
  bool TestApplyMove(const Move& move);

  bool InBoardArea(const Square& sq) const {
    return sq.x >= 0 && sq.x < kBoardSize && sq.y >= 0 && sq.y < kBoardSize;
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

  /* Whether the sq is under attack by the opponent. */
  bool UnderAttack(const Square& sq, Color our_color) const;

  bool InCheck() const {
    return UnderAttack(find(Piece{to_play_, PieceType::kKing}), to_play_);
  }

  bool KingInEnemyCamp(Color player) const;

  uint64_t HashValue() const { return zobrist_hash_; }

  std::string DebugString(bool shredder_fen = false) const;

  int MaterialPoints(Color player) const;

  // Constructs a string describing the shogi board position in Forsyth-Edwards
  // Notation. https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  // Modified to support promoted and pocket pieces.
  std::string ToSFEN() const;

 private:
  size_t SquareToIndex_(const Square& sq) const {
    return sq.y * kBoardSize + sq.x;
  }

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

  template <typename YieldFn>
  void GenerateKingDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateRookDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateRookPDestinations_(Square sq, Color color,
                                  const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateBishopDestinations_(Square sq, Color color,
                                   const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateBishopPDestinations_(Square sq, Color color,
                                    const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateLanceDestinations_(Square sq, Color color,
                                  const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateKnightDestinations_(Square sq, Color color,
                                   const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateSilverDestinations_(Square sq, Color color,
                                   const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateGoldDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  template <typename YieldFn>
  void GeneratePawnDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  // Helper function.
  template <typename YieldFn>
  void GenerateRayDestinations_(Square sq, Color color, Offset offset_step,
                                const YieldFn& yield) const;

  template <typename YieldFn>
  void GenerateDropDestinations_(Color player, const YieldFn& yield) const;

  void SetMovenumber(int move_number);

  std::array<Piece, kNumSquares> board_;
  Color to_play_;

  // This starts at 1, and increments after each black move (a "full move" in
  // shogi is a "half move" by black followed by a "half move" by white).
  int32_t move_number_;

  uint64_t zobrist_hash_;
};

inline std::ostream& operator<<(std::ostream& stream, const ShogiBoard& board) {
  return stream << board.DebugString();
}

inline std::ostream& operator<<(std::ostream& stream, const PieceType& pt) {
  return stream << PieceTypeToString(pt);
}

bool StuckPiece(Color player, PieceType ptype, int8_t y);

bool InPromoZone(Color player, int8_t y);

}  // namespace shogi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SHOGI_SHOGI_BOARD_H_
