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
//

#ifndef OPEN_SPIEL_GAMES_SHOGI_SHOGI_H_
#define OPEN_SPIEL_GAMES_SHOGI_SHOGI_H_

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
#include "open_spiel/games/shogi/shogi_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// An implementation of shogi
// https://en.wikipedia.org/wiki/Shogi
//
//

namespace open_spiel {
namespace shogi {

// Constants.

inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

inline constexpr int NumDistinctActions() {
  return kNumBoardMoves + kNumPocketPieces * kNumSquares;
}

// Keep the same value as for chess, nobody cares
inline constexpr int MaxGameLength() { return 17695; }

inline const std::vector<int>& ObservationTensorShape() {
  static std::vector<int> shape{2 * kNumPieceTypes + 1 /* empty */ +
                                    1   /* repetition count */
                                    + 1 /* side to move */
                                    + 2 * kNumPocketPieces /* pockets */,
                                kBoardSize, kBoardSize};
  return shape;
}

class ShogiGame;

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

// Returns index (0 ... kNumSquares - 1) of a square
// ({0, 0} ... {kBoardSize-1, kBoardSize-1}).
inline uint8_t SquareToIndex(const Square& square) {
  return square.y * kBoardSize + square.x;
}

// Returns square ({0, 0} ... {kBoardSize-1, kBoardSize-1}) from an index
// (0 ... kNumSquares - 1).
inline Square IndexToSquare(uint8_t index) {
  return Square{static_cast<int8_t>(index % kBoardSize),
                static_cast<int8_t>(index / kBoardSize)};
}

int EncodeMove(const Square& from_square, int destination_index, int kBoardSize,
               int num_actions_destinations);

inline constexpr int kNumActionDestinations = 73;

int8_t ReflectRank(Color to_play, int kBoardSize, int8_t rank);

Color PlayerToColor(Player p);

std::pair<Square, int> ActionToDestination(int action, int kBoardSize,
                                           int num_actions_destinations);

Action MoveToAction(const Move& move);

Move ActionToMove(Action action, const ShogiBoard& board);

// State of an in-play game.
class ShogiState : public State {
 public:
  // Constructs a shogi state at the standard start position.
  explicit ShogiState(std::shared_ptr<const Game> game);

  // Constructs a shogi state at the given position in SFEN Notation.
  // SFEN is similar to FEN for chess but supports pocket pieces and
  // promoted pieces.
  ShogiState(std::shared_ptr<const Game> game, const std::string& sfen);
  ShogiState(const ShogiState&) = default;

  Player CurrentPlayer() const override;
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

  bool InCheck() const { return current_board_.InCheck(); }

  // Current board.
  ShogiBoard& Board() { return current_board_; }
  const ShogiBoard& Board() const { return current_board_; }

  // Starting board.
  ShogiBoard& StartBoard() { return start_board_; }
  const ShogiBoard& StartBoard() const { return start_board_; }

  std::vector<Move>& MovesHistory() { return moves_history_; }
  const std::vector<Move>& MovesHistory() const { return moves_history_; }

  // A prettier board string.
  std::string DebugString() const;

  // Returns an action parsed from standard algebraic notation or long
  // algebraic notation (using ShogiBoard::ParseMove), or kInvalidAction if
  // the parsing fails.
  Action ParseMoveToAction(const std::string& move_str) const;

  std::string Serialize() const override;

  // Shogi ends the game at the 4th position of a position.
  // If the repetition was a result of perpetual check,
  // the checking player loses.
  bool IsRepetitionEnd() const;

  // Returns the number of times the specified state has appeared in the
  // history.
  int NumRepetitions(const ShogiState& state) const;

  int MaterialPoints(Color player) const;

  // Get the SFEN for this move and the list of moves in UCI format.
  std::pair<std::string, std::vector<std::string>> ExtractSFenAndMaybeMoves()
      const;

  const ShogiGame* ParentGame() const {
    return down_cast<const ShogiGame*>(GetGame().get());
  }

  std::string StartSFEN() const;

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
  ShogiBoard start_board_;
  // We store the current board position as an optimization.
  ShogiBoard current_board_;
  // This SFEN string is used only when NewInitialState is called
  // with a specific initial SFEN.
  std::string specific_initial_sfen_;

  std::array<int, 2> check_count_ = {0, 0};

  // RepetitionTable records how many times the given hash exists in the history
  // stack (including the current board).
  // We are already indexing by board hash, so there is no need to hash that
  // hash again, so we use a custom passthrough hasher.
  class PassthroughHash {
   public:
    PassthroughHash() = default;
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
  };
  using RepetitionTable = absl::flat_hash_map<uint64_t, int, PassthroughHash>;
  RepetitionTable repetitions_;
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

// Game object.
class ShogiGame : public Game {
 public:
  explicit ShogiGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return shogi::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState(
      const std::string& sfen) const override {
    return absl::make_unique<ShogiState>(shared_from_this(), sfen);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<ShogiState>(shared_from_this());
  }
  int NumPlayers() const override { return shogi::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  absl::optional<double> UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    return shogi::ObservationTensorShape();
  }
  int MaxGameLength() const override { return shogi::MaxGameLength(); }

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;
};

}  // namespace shogi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SHOGI_SHOGI_H_
