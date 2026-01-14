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

#ifndef OPEN_SPIEL_GAMES_CHESS_H_
#define OPEN_SPIEL_GAMES_CHESS_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of chess:
// https://en.wikipedia.org/wiki/Chess
//
// Parameters:
//       "chess960"    bool      is it a Fischer Random game?  (default: false)
//

namespace open_spiel {
namespace chess {

// Constants.
inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

inline constexpr int NumDistinctActions() { return 4674; }
inline constexpr int kLeftCastlingAction = 4672;
inline constexpr int kRightCastlingAction = 4673;

// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
inline constexpr int MaxGameLength() { return 17695; }

inline const std::vector<int>& ObservationTensorShape() {
  static std::vector<int> shape{
      13 /* piece types * colours + empty */ + 1 /* repetition count */ +
          1 /* side to move */ + 1 /* irreversible move counter */ +
          4 /* castling rights */,
      kMaxBoardSize, kMaxBoardSize};
  return shape;
}

constexpr bool kDefaultChess960 = false;

// Returns a list of all possible starting positions in chess960.
std::vector<std::string> Chess960StartingPositions();

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

inline constexpr std::array<PieceType, 3> kUnderPromotionIndexToType = {
    PieceType::kRook, PieceType::kBishop, PieceType::kKnight};
inline constexpr std::array<Offset, 3> kUnderPromotionDirectionToOffset = {
    {{0, 1}, {1, 1}, {-1, 1}}};
inline constexpr int kNumUnderPromotions =
    kUnderPromotionIndexToType.size() * kUnderPromotionDirectionToOffset.size();

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
inline uint8_t SquareToIndex(const Square& square, int board_size) {
  return square.y * board_size + square.x;
}

// Returns square ({0, 0} ... {BoardSize-1, BoardSize-1}) from an index
// (0 ... BoardSize*BoardSize-1).
inline Square IndexToSquare(uint8_t index, int board_size) {
  return Square{static_cast<int8_t>(index % board_size),
                static_cast<int8_t>(index / board_size)};
}

int EncodeMove(const Square& from_square, int destination_index, int board_size,
               int num_actions_destinations);

inline constexpr int kNumActionDestinations = 73;

int8_t ReflectRank(Color to_play, int board_size, int8_t rank);

Color PlayerToColor(Player p);

std::pair<Square, int> ActionToDestination(int action, int board_size,
                                           int num_actions_destinations);

Action MoveToAction(const Move& move, int board_size = kDefaultBoardSize);

Move ActionToMove(const Action& action, const ChessBoard& board);

// State of an in-play game.
class ChessState : public State {
 public:
  // Constructs a chess state at the standard start position.
  ChessState(std::shared_ptr<const Game> game);

  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  ChessState(std::shared_ptr<const Game> game, const std::string& fen);
  ChessState(const ChessState&) = default;

  ChessState& operator=(const ChessState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  ActionsAndProbs ChanceOutcomes() const override;  // for chess960

  bool IsTerminal() const override {
    return static_cast<bool>(MaybeFinalReturns());
  }

  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // Current board.
  ChessBoard& Board() { return current_board_; }
  const ChessBoard& Board() const { return current_board_; }
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  ChessBoard& StartBoard() { return start_board_; }
  const ChessBoard& StartBoard() const { return start_board_; }

  std::vector<Move>& MovesHistory() { return moves_history_; }
  const std::vector<Move>& MovesHistory() const { return moves_history_; }

  // A prettier board string.
  std::string DebugString() const;

  // Returns an action parsed from standard algebraic notation or long
  // algebraic notation (using ChessBoard::ParseMove), or kInvalidAction if
  // the parsing fails.
  Action ParseMoveToAction(const std::string& move_str) const;

  std::string Serialize() const override;

  // Draw can be claimed under the FIDE 3-fold repetition rule (the current
  // board position has already appeared twice in the history).
  bool IsRepetitionDraw() const;

  // Returns the number of times the specified state has appeared in the
  // history.
  int NumRepetitions(const ChessState& state) const;

  // Get the FEN for this move and the list of moves in UCI format.
  std::pair<std::string, std::vector<std::string>> ExtractFenAndMaybeMoves()
      const;

  const ChessGame* ParentGame() const {
    return down_cast<const ChessGame*>(GetGame().get());
  }

  std::string StartFEN() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Calculates legal actions and caches them. This is separate from
  // LegalActions() as there are a number of other methods that need the value
  // of LegalActions. This is a separate method as it's called from
  // IsTerminal(), which is also called by LegalActions().
  void MaybeGenerateLegalActions() const;

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<Move> moves_history_;
  // We store the start board for history to support games not starting
  // from the start position.
  ChessBoard start_board_;
  // We store the current board position as an optimization.
  ChessBoard current_board_;
  // This FEN string is used only when NewInitialState is called with a specific
  // initial FEN.
  std::string specific_initial_fen_;

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
  using RepetitionTable = absl::flat_hash_map<uint64_t, int, PassthroughHash>;
  RepetitionTable repetitions_;
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

// Game object.
class ChessGame : public Game {
 public:
  explicit ChessGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return chess::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState(
      const std::string& fen) const override {
    return absl::make_unique<ChessState>(shared_from_this(), fen);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<ChessState>(shared_from_this());
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  absl::optional<double> UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    return chess::ObservationTensorShape();
  }
  int MaxGameLength() const override { return chess::MaxGameLength(); }
  int MaxChanceOutcomes() const override;  // for chess960

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  bool IsChess960() const { return chess960_; }

  std::string Chess960LookupFEN(int index) const {
    SPIEL_CHECK_GE(index, 0);
    SPIEL_CHECK_LT(index, initial_fens_.size());
    return initial_fens_[index];
  }

 private:
  bool chess960_;
  std::vector<std::string> initial_fens_;  // Used for chess960.
};

}  // namespace chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHESS_H_
