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

#ifndef OPEN_SPIEL_GAMES_DARK_CHESS_H_
#define OPEN_SPIEL_GAMES_DARK_CHESS_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Dark chess - imperfect information version of chess:
// https://en.wikipedia.org/wiki/Dark_chess
//
// Parameters:
//   "board_size"  int     Number of squares in each row and column (default: 8)
//   "fen"         string  String describing the chess board position in
//                         Forsyth-Edwards Notation. The FEN has to match
//                         the board size. Default values are available for
//                         board sizes 4 and 8.

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

class DarkChessGame;
class DarkChessObserver;

// State of an in-play game.
class DarkChessState : public State {
 public:
  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  DarkChessState(std::shared_ptr<const Game> game, int board_size,
                 const std::string& fen);
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
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // Current board.
  chess::ChessBoard& Board() { return current_board_; }
  const chess::ChessBoard& Board() const { return current_board_; }
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  chess::ChessBoard& StartBoard() { return start_board_; }
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  std::vector<chess::Move>& MovesHistory() { return moves_history_; }
  const std::vector<chess::Move>& MovesHistory() const {
    return moves_history_;
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  friend class DarkChessObserver;

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
    return absl::make_unique<DarkChessState>(shared_from_this(), board_size_,
                                             fen_);
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  double UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    std::vector<int> shape{
        (13 +  // public boards:  piece types * colours + empty
         14)   // private boards: piece types * colours + empty + unknown
            * board_size_ * board_size_ +
        3 +    // public: repetitions count, one-hot encoding
        2 +    // public: side to play
        1 +    // public: irreversible move counter -- a fraction of $n over 100
        2 * 2  // private: left/right castling rights, one-hot encoded.
    };
    return shape;
  }
  int MaxGameLength() const override { return chess::MaxGameLength(); }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;

  std::shared_ptr<DarkChessObserver> default_observer_;

 private:
  const int board_size_;
  const std::string fen_;
};

}  // namespace dark_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DARK_CHESS_H_
