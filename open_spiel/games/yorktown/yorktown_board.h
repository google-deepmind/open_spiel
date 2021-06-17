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

#ifndef OPEN_SPIEL_GAMES_IMPL_YORKTOWN_BOARD_H_
#define OPEN_SPIEL_GAMES_IMPL_YORKTOWN_BOARD_H_

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

// This board method is based on the chess_board version of open_spiel. I uses the same square representation
// as well as the methods to calculate indices and destinations

namespace open_spiel {
namespace yorktown {

using chess_common::Offset;
using chess_common::Square;

// This method is similar to chess_common::SquareToString. The only difference
// is that we add 1 to x in case of j to get k -> a-k without j which is a common notation for Stratego
inline std::string SquareToStrategoString(const Square& square){
  if (square == chess_common::kInvalidSquare) {
    return "None";
  } else {
    std::string s;
    s.push_back(square.x == 9 ? 'a' + square.x + 1 : 'a' + square.x);
    s.push_back('1' + square.y);
    return s;
  }
}

template <std::size_t... Dims>
using ZobristTableU64 = chess_common::ZobristTable<uint64_t, Dims...>;

// The Colors of the pieces/players...
enum class Color : int8_t { kRed = 0, kBlue = 1, kEmpty = 2};

// Cast the color to 0 (Red) and 1 (Blue). The case of kEmpty is not 
// expected
inline int ToInt(Color color) { return color == Color::kRed ? 0 : 1; }

// Gives you the color of the opponent
inline Color OppColor(Color color) {
  return color == Color::kRed ? Color::kBlue : Color::kRed;
}

// Gives you the color as a string
std::string ColorToString(Color c);

inline std::ostream& operator<<(std::ostream& stream, Color c) {
  return stream << ColorToString(c);
}

// All possible PieceTypes (the 12 pieces + kEmpty and kLake)
enum class PieceType : int8_t {
  kEmpty = 0,
  kFlag = 1,
  kSpy = 2,
  kScout = 3,
  kMiner = 4,
  kSergeant = 5,
  kLieutenant = 6,
  kCaptain = 7,
  kMajor = 8,
  kColonel = 9,
  kGeneral = 10,
  kMarshal = 11,
  kBomb = 12,
  kLake = -1
};

// An array with the 12 PieceTypes of all real pieces
static inline constexpr std::array<PieceType, 12> kPieceTypes = {
    {PieceType::kFlag, PieceType::kSpy, PieceType::kScout, PieceType::kMiner,
     PieceType::kSergeant, PieceType::kLieutenant, PieceType::kCaptain, 
     PieceType::kMajor, PieceType::kColonel, PieceType::kGeneral,
     PieceType::kMarshal, PieceType::kBomb}};


// In case all the pieces are represented in the same plane, these values are
// used to represent each piece type. 
// Note that this numbers do not indicate value or probability.
static inline constexpr std::array<float, 12> kPieceRepresentation = {
    {1, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05}};


// Tries to parse piece type from char ('F', 'B', 'S', '2', '3', '4', '5', '6', '7', '8', '9', '0').
// Case-insensitive.
absl::optional<PieceType> PieceTypeFromChar(char c);
absl::optional<PieceType> PieceTypeFromStraDos(char c);

// Converts piece type to character strings - "F", "B", "S", "1" - "0".
// p must be one of the enumerator values of PieceType.
std::string PieceTypeToString(PieceType p);
std::string PieceTypeToStradosString(PieceType p, Color c);

// default struct for pieces with one color and type and bool giving information
// about the visibility of this piece
struct Piece {
  bool operator==(const Piece& other) const {
    return type == other.type && color == other.color && isVisible == other.isVisible;
  }

  bool operator!=(const Piece& other) const { return !(*this == other); }

  // A human readable string representation like "B7" for "Blue Captain (Rank 7)"
  // visibility is indicated by the case of the letter -> "B" means hidden, "b" open/visible
  std::string ToString(Color c = Color::kEmpty) const;
  // A string representation from the StaDos notation scheme "N" for "Blue Captain"
  std::string ToStraDosString(Color c = Color::kEmpty) const;

  Color color;
  PieceType type;
  bool isVisible = false;
  bool hasMoved = false;

};

// a static constant/struct for an empty piece
static inline constexpr Piece kEmptyPiece =
    Piece{Color::kEmpty, PieceType::kEmpty, true};


inline std::ostream& operator<<(std::ostream& stream, const Piece& p) {
  return stream << p.ToString();
}

// My Board is defined 10x10 and has ranks 1 to :(10)
// This gives you the rank from 0 to 9
inline absl::optional<int8_t> ParseRank(char c) {
  if (c >= '1' && c <= ':') return c - '1';
  return std::nullopt;
}

// My Board is defined 10x10 and has file A to K without J
// This gives you the file from 0 to 9 
inline absl::optional<int8_t> ParseFile(char c) {
  if (c == 'k') c = c-1;
  if (c >= 'a' && c <= 'j') return c - 'a';
  return std::nullopt;
}

// Maps y = [0, 9] to rank ["1", ":"].
inline std::string RankToString(int8_t rank) {
  return std::string(1, '1' + rank);
}

// Maps x = [0, 9] to file ["A", "K"].
inline std::string FileToString(int8_t file) {
  if (file == 9) file = file + 1;
  return std::string(1, 'a' + file);
}

// get a square of the board from a string
absl::optional<Square> SquareFromString(const std::string& s);

// Forward declare ChessBoard here because it's needed in Move::ToComplexLAN.
template <uint32_t kBoardSize>
class YorktownBoard;

// standart definition of a move (in the style of chess)
struct Move {
  Square from;
  Square to;
  Piece piece;

  Move(const Square& from, const Square& to, const Piece& piece)
      : from(from),
        to(to),
        piece(piece) {}

  Move(const Square& from, const Square& to)
      : from(from),
        to(to) {}

  // Gives you a string representation of the Move
  std::string ToString() const;
  // Gives you a string representation in form of "a4a5". It is based on the
  // simple long algebraic notation used in chess
  std::string ToLANMove() const;
  // Gives you a more complex string representation in form of "a4-a5". It is based on the
  // long algebraic notation used in chess with more informations
  std::string ToComplexLANMove(const YorktownBoard<10>& board) const;
  // Gives you a string representation used to generate xml files, used by Strados2
  // Hint: Do not use this representation of a move, it is not really readable
  std::string ToStraDosMove() const;

  bool operator==(const Move& other) const {
    return from == other.from && to == other.to && piece == other.piece;
  }
};

inline std::ostream& operator<<(std::ostream& stream, const Move& m) {
  return stream << m.ToString();
}

template <uint32_t kBoardSize>
class YorktownBoard {

 public:
  YorktownBoard();

  /* This method generates a YorktownBoard from a StraDos3 string
  * StraDos3 is a new notation scheme for Stratego/Yorktown matches. It is a combination
  * of the Strados2 system used by the website Gravon and the fen notation used in chess. 
  * A Strados3 string has 3 parts, the piece configuration of the current board situation
  * as a string of 100 characters each representing a square. The second part is the current player
  * and the third part the ply count or the current situation.
  * E.g. "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0" 
  */
  static absl::optional<YorktownBoard> BoardFromStraDos3(const std::string& straDos);

  // "at" is a small helping method getting a piece from a given square
  const Piece& at(Square sq) const { return board_[SquareToIndex_(sq)]; }
  
  // Setter method for a square
  void set_square(Square sq, Piece p);

  // Gives you the board representations as a 1-dimensional array
  const std::array<Piece, kBoardSize * kBoardSize>& pieces() const {
    return board_;
  }

  // gives you the active player/color
  Color ToPlay() const { return to_play_; }

  // set the active player/color to c
  void SetToPlay(Color c);
 
  // setter for the movenumber/ply count
  int32_t Movenumber() const { return move_number_; }

  // Find the locations of any piece of the given type, color and visibility, or InvalidSquare().
  std::vector<Square> find(const Piece& piece) const;

  // given a square, gives you an array with the 4 neighbour pieces.
  std::array<Piece, 4>  neighbours(const Square &square) const;

  // The generation functions call yield(move) for each move generated.
  // The yield function should return whether generation should continue.
  // For performance reasons, we do not guarantee that no more moves will be
  // generated if yield returns false. It is only for optimization.
  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const;
  
  // This method checks if their are anymore legal moves at the given moment
  // This uses the information of the board to get the active player
  bool HasLegalMoves() const {
    bool found = false;
    GenerateLegalMoves([&found](const Move&) {
      found = true;
      return false;  // We don't need any more moves.
    });
    return found;
  }

  // Parses a move from a notation scheme of choice, e.g. LAN, complex LAN or StraDos string (
  // see below).
  absl::optional<Move> ParseMove(const std::string& move) const;
  absl::optional<Move> ParseLANMove(const std::string& move) const;
  absl::optional<Move> ParseComplexLANMove(const std::string& move) const;
  absl::optional<Move> ParseStraDosMove(const std::string& move) const;

  // Applies a move on the board. Checks if it is an attack move and handles it
  // Sets visibility if a piece attacks or is attacked.

  // Applies the given move
  void ApplyMove(const Move& move);

  // Checks if a given square is in the range of the board
  static bool InBoardArea(const Square& sq) {
    return sq.x >= 0 && sq.x < kBoardSize && sq.y >= 0 && sq.y < kBoardSize;
  }

  // Checks Visibility of a given square (empty fields are visible/open)
  bool IsVisible(const Square& sq) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.isVisible;
  }

  // Checks if a given square is empty
  bool IsEmpty(const Square& sq) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.type == PieceType::kEmpty;
  }

  // Checks if a given square is a lake
  bool IsLake(const Square& sq) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.type == PieceType::kLake;
  }

  // Checks if the given square is captured by an opponents piece
  bool IsEnemy(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.type != PieceType::kEmpty && piece.type != PieceType::kLake && piece.color != our_color;
  }

  // Checks if the given square is captured by an own piece 
  bool IsFriendly(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return piece.color == our_color;
  }

  // Combination of IsEnemy and IsEmpty 
  bool IsEmptyOrEnemy(const Square& sq, Color our_color) const {
    const Piece& piece = board_[SquareToIndex_(sq)];
    return IsEmpty(sq) == true || IsEnemy(sq, our_color) == true;
  }

  // The boardsize, we can only describe quadratic fields (Boardsize*Boardsize)
  int BoardSize() const { return kBoardSize; }

  uint64_t HashValue() const { return zobrist_hash_; }

  // Arrays of all livingPieces (all pieces still on Board) and all captured pieces for the current match
  std::array<int, kPieceTypes.size()*2> CapturedPieces() const { return capturedPieces_; }
  std::array<int, kPieceTypes.size()*2> LivingPieces() const { return livingPieces_; }
  
  // setter for the LivingPieces array
  void SetLivingPieces(std::array<int, kPieceTypes.size()*2> amounts){
    livingPieces_ = amounts;
  }

  // This method gets two parameters color and onlyBoard. The first parameter defines
  // which player perspective is shown. If the given color is Empty the observers
  // perspective without hidden information will be shown. OnlyBoard reduces the 
  // return string to the board without information like whos turn or which move it is.
  std::string DebugString(Color c = Color::kEmpty, bool onlyBoard = false) const;

  // The same method as DebugString but with Strados notation instead of a better human readable notation scheme
  std::string DebugStringStraDos(Color c = Color::kEmpty, bool onlyBoard = false) const;

  // Return the piece-configuration as a string from player/color c's perspective
  std::string ToString(Color C = Color::kEmpty) const;

  // Returns a FEN-like string with piece-configuration, move number and active player
  std::string ToStraDos3(Color C = Color::kEmpty) const;

 private:
  // Given a square it return the Index from 0 to 99
  static size_t SquareToIndex_(Square sq) { return sq.y * kBoardSize + sq.x; }

  // Given a square and a color, it returns all possible moves for a standard piece,
  // which can only move one field in the four standard directions. It checks if the field
  // is free, not occupied by a friendly piece or a lake
  template <typename YieldFn>
  void GenerateStandardDestinations_(Square sq, Color color,
                                 const YieldFn& yield) const;

  // Similiar to GenerateStandardDestinations only for the scout piece, which moves
  // like a rook in chess. 
  template <typename YieldFn>
  void GenerateScoutDestinations_(Square sq, Color color,
                                  const YieldFn& yield) const;

  // Helper function for the scout moves finding the end of a row/line
  template <typename YieldFn>
  void GenerateRayDestinations_(Square sq, Color color, Offset offset_step,
                                const YieldFn& yield) const;


  // Setter for the move_number
  void SetMovenumber(int move_number);

  // updates capturedPieces as well as livingPieces
  void capturePiece(Piece piece);


  // A Board is defined as an array of size 100
  std::array<Piece, kBoardSize * kBoardSize> board_;
  std::array<int, kPieceTypes.size()*2> capturedPieces_ = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  std::array<int, kPieceTypes.size()*2> livingPieces_ = {1,1,8,5,4,4,4,3,2,1,1,6,1,1,8,5,4,4,4,3,2,1,1,6};

  
  // The active player/color
  Color to_play_;
 
  // This starts at 1, and increments after each ply other then in chess
  int32_t move_number_;
  uint64_t zobrist_hash_;
};

template <uint32_t kBoardSize>
inline std::ostream& operator<<(std::ostream& stream,
                                const YorktownBoard<kBoardSize>& board) {
  return stream << board.DebugString();
}

inline std::ostream& operator<<(std::ostream& stream, const PieceType& pt) {
  return stream << PieceTypeToString(pt);
}

// A little method generating a "default" or more specific a fixed board to test on
YorktownBoard<10> MakeDefaultBoard();
YorktownBoard<10> MakeDefaultBoard(std::string strados3);

}  // namespace yorktown
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_IMPL_YORKTOWN_BOARD_H_
