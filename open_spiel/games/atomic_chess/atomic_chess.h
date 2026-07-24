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

#ifndef OPEN_SPIEL_GAMES_ATOMIC_CHESS_H_
#define OPEN_SPIEL_GAMES_ATOMIC_CHESS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of atomic chess:
// https://en.wikipedia.org/wiki/Atomic_chess
//
// Standard chess rules apply, with one change: any capture triggers an
// "explosion" centered on the destination square. The explosion removes the
// capturing piece, the captured piece, and every NON-PAWN piece on the (up to)
// eight surrounding squares. Pawns are immune to explosions unless they are the
// directly-captured piece.
//
// Consequences of the explosion rule:
//   * Kings may not capture (a king capture would blow up the king itself).
//   * A player wins by exploding the opponent's king, OR by regular checkmate.
//     Exploding the enemy king wins immediately, even while one's own king is
//     in check.
//   * A move that would explode one's own king is illegal (this includes
//     capturing a piece adjacent to one's own king).
//   * When the two kings are on adjacent squares, no check can exist: the
//     opponent can never capture a king without exploding its own king.
//   * Stalemate is a draw (as in standard chess).
//
// This follows the canonical lichess ruleset. The action encoding, board
// representation, and observation tensor are reused directly from the standard
// chess implementation.

namespace open_spiel {
namespace atomic_chess {

// Hardcoding board size to standard 8x8.
inline constexpr int kBoardSize = 8;
inline constexpr int k2dBoardSize = kBoardSize * kBoardSize;

class AtomicChessGame;

// State of an in-play game.
class AtomicChessState : public State {
 public:
  // Constructs an atomic chess state at the standard start position.
  explicit AtomicChessState(std::shared_ptr<const Game> game);

  // Constructs an atomic chess state at the given position in Forsyth-Edwards
  // Notation: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  AtomicChessState(std::shared_ptr<const Game> game, const std::string& fen);
  AtomicChessState(const AtomicChessState&) = default;

  AtomicChessState& operator=(const AtomicChessState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId
                        : chess::ColorToPlayer(Board().ToPlay());
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
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  chess::ChessBoard& StartBoard() { return start_board_; }
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  std::vector<chess::Move>& MovesHistory() { return moves_history_; }
  const std::vector<chess::Move>& MovesHistory() const {
    return moves_history_;
  }

  // A prettier board string.
  std::string DebugString() const;

  // Returns an action parsed from standard algebraic notation or long
  // algebraic notation (using chess::ChessBoard::ParseMove), or kInvalidAction
  // if the parsing fails.
  Action ParseMoveToAction(const std::string& move_str) const;

  std::string Serialize() const override;

  // Draw can be claimed under the 3-fold repetition rule (the current board
  // position has already appeared twice in the history).
  bool IsRepetitionDraw() const;

  // Get the FEN for this move and the list of moves in UCI format.
  std::pair<std::string, std::vector<std::string>> ExtractFenAndMaybeMoves()
      const;

  const AtomicChessGame* ParentGame() const {
    return down_cast<const AtomicChessGame*>(GetGame().get());
  }

  std::string StartFEN() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
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
class AtomicChessGame : public Game {
 public:
  explicit AtomicChessGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return chess::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState(
      const std::string& fen) const override {
    return absl::make_unique<AtomicChessState>(shared_from_this(), fen);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<AtomicChessState>(shared_from_this());
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return chess::LossUtility(); }
  absl::optional<double> UtilitySum() const override {
    return chess::DrawUtility();
  }
  double MaxUtility() const override { return chess::WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    static std::vector<int> shape{
        13 /* piece types * colours + empty */ + 1 /* repetition count */ +
            1 /* side to move */ + 1 /* irreversible move counter */,
        kBoardSize, kBoardSize};
    return shape;
  }
  int MaxGameLength() const override { return chess::MaxGameLength(); }

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;
};

}  // namespace atomic_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ATOMIC_CHESS_H_
