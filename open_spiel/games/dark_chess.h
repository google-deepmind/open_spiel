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

#ifndef OPEN_SPIEL_GAMES_DARK_CHESS_H_
#define OPEN_SPIEL_GAMES_DARK_CHESS_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Dark chess - imperfect information version of chess:
// https://en.wikipedia.org/wiki/Dark_chess
//
// Parameters: none

namespace open_spiel {
namespace dark_chess {

// Constants.
inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

// See action encoding below.
inline constexpr int NumDistinctActions() { return 4672; }

// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
inline constexpr int MaxGameLength() { return 17695; }

inline const std::vector<int> &ObservationTensorShape() {
  static std::vector<int> shape{
      13 /* piece types * colours + empty */ + 1 /* repetition count */ +
      1 /* side to move */ + 1 /* irreversible move counter */ +
      4 /* castling rights */,
      chess::kMaxBoardSize, chess::kMaxBoardSize};
  return shape;
}

class DarkChessGame;

// Action encoding (must be changed to support larger boards):
// bits 0-5: from square (0-64)
// bits 6-11: to square (0-64)
// bits 12-14: promotion type (0 if not promotion)
// bits 15: is castling (we need to record this because just from and to squares
//   can be ambiguous in chess960).
//
// Promotion type:
enum class PromotionTypeEncoding {
  kNotPromotion = 0,
  kQueen = 1,
  kRook = 2,
  kBishop = 3,
  kKnight = 4
};

inline constexpr std::array<chess::PieceType, 3> kUnderPromotionIndexToType = {
    chess::PieceType::kRook, chess::PieceType::kBishop, chess::PieceType::kKnight};
inline constexpr std::array<chess::Offset, 3> kUnderPromotionDirectionToOffset = {
    {{0, 1}, {1, 1}, {-1, 1}}};
inline constexpr int kNumUnderPromotions =
    kUnderPromotionIndexToType.size() * kUnderPromotionDirectionToOffset.size();

inline constexpr int kNumActionDestinations = 73;

std::pair<chess::Square, int> ActionToDestination(int action, int board_size,
                                           int num_actions_destinations);


// State of an in-play game.
class DarkChessState : public State {
 public:

  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  DarkChessState(std::shared_ptr<const Game> game, int board_size, const std::string& fen);
  DarkChessState(const DarkChessState&) = default;

  DarkChessState& operator=(const DarkChessState&) = default;

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
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // Current board.
  chess::ChessBoard& Board() { return current_board_; }
  const chess::ChessBoard& Board() const { return current_board_; }

  // Starting board.
  chess::ChessBoard& StartBoard() { return start_board_; }
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  std::vector<chess::Move>& MovesHistory() { return moves_history_; }
  const std::vector<chess::Move>& MovesHistory() const { return moves_history_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Draw can be claimed under the FIDE 3-fold repetition rule (the current
  // board position has already appeared twice in the history).
  bool IsRepetitionDraw() const;

  // Calculates legal actions and caches them. This is separate from
  // LegalActions() as there are a number of other methods that need the value
  // of LegalActions. This is a separate method as it's called from
  // IsTerminal(), which is also called by LegalActions().
  void MaybeGenerateLegalActions() const;

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<chess::Move> moves_history_;
  // We store the start board for history to support games not starting
  // from the start position.
  chess::ChessBoard start_board_;
  // We store the current board position as an optimization.
  chess::ChessBoard current_board_;

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
class DarkChessGame : public Game {
 public:
  explicit DarkChessGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return chess::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<DarkChessState>(shared_from_this(), board_size_, fen_);
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  double UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    return chess::ObservationTensorShape();
  }
  int MaxGameLength() const override { return chess::MaxGameLength(); }

 private:
  const int board_size_;
  const std::string fen_;
};

}  // namespace dark_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DARK_CHESS_H_
