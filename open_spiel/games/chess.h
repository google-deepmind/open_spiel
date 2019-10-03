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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_CHESS_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_CHESS_H_

#include <array>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of chess:
// https://en.wikipedia.org/wiki/Chess
//
// Parameters: none

namespace open_spiel {
namespace chess {

// Constants.
constexpr int NumPlayers() { return 2; }
constexpr double LossUtility() { return -1; }
constexpr double DrawUtility() { return 0; }
constexpr double WinUtility() { return 1; }
constexpr int BoardSize() { return 8; }

// See action encoding below.
constexpr int NumDistinctActions() { return (1 << 15); }

// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
constexpr int MaxGameLength() { return 17695; }

inline const std::vector<int>& InformationStateNormalizedVectorShape() {
  static std::vector<int> shape{
      13 /* piece types * colours + empty */ + 1 /* repetition count */ +
          1 /* side to move */ + 1 /* irreversible move counter */ +
          4 /* castling rights */,
      BoardSize(), BoardSize()};
  return shape;
}

class ChessGame;

inline int ColorToPlayer(Color c) {
  if (c == Color::kBlack) {
    return 0;
  } else if (c == Color::kWhite) {
    return 1;
  } else {
    SpielFatalError("Unknown color");
  }
}

inline int OtherPlayer(Player player) { return player == Player{0} ? 1 : 0; }

// Action encoding (must be changed to support larger boards):
// bits 0-5: from square (0-64)
// bits 6-11: to square (0-64)
// bits 12-14: promotion type (0 if not promotion)
//
// Promotion type:
enum class PromotionTypeEncoding {
  kNotPromotion = 0,
  kQueen = 1,
  kRook = 2,
  kBishop = 3,
  kKnight = 4
};

// Reads a bitfield within action, with LSB at offset, and length bits long (up
// to 8).
inline uint8_t GetField(Action action, int offset, int length) {
  return (action >> offset) & ((1ULL << length) - 1);
}

// Sets a bitfield within action, with LSB at offset, and length bits long (up
// to 8) to value.
inline void SetField(int offset, int length, uint8_t value, Action* action) {
  uint32_t mask = ((1ULL << length) - 1) << offset;
  *action &= ~mask;
  *action |= static_cast<Action>(value) << offset;
}

// Returns index (0 ... BoardSize*BoardSize-1) of a square
// ({0, 0} ... {BoardSize-1, BoardSize-1}).
inline uint8_t SquareToIndex(const Square& square) {
  return square.y * BoardSize() + square.x;
}

// Returns square ({0, 0} ... {BoardSize-1, BoardSize-1}) from an index
// (0 ... BoardSize*BoardSize-1).
inline Square IndexToSquare(uint8_t index) {
  return Square{static_cast<int8_t>(index % BoardSize()),
                static_cast<int8_t>(index / BoardSize())};
}

inline Action MoveToAction(const Move& move) {
  Action action = 0;

  SetField(0, 6, SquareToIndex(move.from), &action);
  SetField(6, 6, SquareToIndex(move.to), &action);

  uint8_t promo_encoded = 0;
  switch (move.promotion_type) {
    case PieceType::kQueen:
      promo_encoded = static_cast<uint8_t>(PromotionTypeEncoding::kQueen);
      break;
    case PieceType::kRook:
      promo_encoded = static_cast<uint8_t>(PromotionTypeEncoding::kRook);
      break;
    case PieceType::kBishop:
      promo_encoded = static_cast<uint8_t>(PromotionTypeEncoding::kBishop);
      break;
    case PieceType::kKnight:
      promo_encoded = static_cast<uint8_t>(PromotionTypeEncoding::kKnight);
      break;
    default:
      promo_encoded = 0;
  }

  SetField(12, 3, promo_encoded, &action);

  return action;
}

inline Move ActionToMove(const Action& action) {
  PieceType promo_type;
  switch (static_cast<PromotionTypeEncoding>(GetField(action, 12, 3))) {
    case PromotionTypeEncoding::kQueen:
      promo_type = PieceType::kQueen;
      break;
    case PromotionTypeEncoding::kRook:
      promo_type = PieceType::kRook;
      break;
    case PromotionTypeEncoding::kBishop:
      promo_type = PieceType::kBishop;
      break;
    case PromotionTypeEncoding::kKnight:
      promo_type = PieceType::kKnight;
      break;
    case PromotionTypeEncoding::kNotPromotion:
      promo_type = PieceType::kEmpty;
      break;
    default:
      SpielFatalError("Unknown promotion type encoding");
  }
  return Move(IndexToSquare(GetField(action, 0, 6)),
              IndexToSquare(GetField(action, 6, 6)), promo_type);
}

// State of an in-play game.
class ChessState : public State {
 public:
  // Constructs a chess state at the standard start position.
  ChessState();

  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  ChessState(const std::string& fen);
  ChessState(const ChessState&) = default;

  ChessState& operator=(const ChessState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(Board().ToPlay());
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override {
    return static_cast<bool>(MaybeFinalReturns());
  }

  std::vector<double> Returns() const override;

  std::string InformationState(Player player) const override;
  void InformationStateAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // Current board.
  StandardChessBoard& Board() { return current_board_; }
  const StandardChessBoard& Board() const { return current_board_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Draw can be claimed under the FIDE 3-fold repetition rule (the current
  // board position has already appeared twice in the history).
  bool IsRepetitionDraw() const;

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<Move> moves_history_;
  // We store the start board for history to support games not starting
  // from the start position.
  StandardChessBoard start_board_;
  // We store the current board position as an optimization.
  StandardChessBoard current_board_;

  // RepetitionTable records how many times the given hash exists in the history
  // stack (including the current board).
  // We are already indexing by board hash, so there is no need to hash that
  // hash again, so we use a custom passthrough hasher.
  class PassthroughHash {
   public:
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
  };
  using RepetitionTable = std::unordered_map<uint64_t, int, PassthroughHash>;
  RepetitionTable repetitions_;
};

// Game object.
class ChessGame : public Game {
 public:
  explicit ChessGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return chess::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ChessState());
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  double UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new ChessGame(*this));
  }
  std::vector<int> InformationStateNormalizedVectorShape() const override {
    return chess::InformationStateNormalizedVectorShape();
  }
  int MaxGameLength() const override { return chess::MaxGameLength(); }
};

}  // namespace chess
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_CHESS_H_
