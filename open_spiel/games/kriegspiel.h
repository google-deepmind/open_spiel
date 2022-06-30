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

#ifndef OPEN_SPIEL_GAMES_KRIEGSPIEL_H_
#define OPEN_SPIEL_GAMES_KRIEGSPIEL_H_

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

// Kriegspiel - imperfect information version of chess:
// https://en.wikipedia.org/wiki/Kriegspiel
// The implementation follows ICC rules (with a few exceptions):
// https://www.chessclub.com/help/Kriegspiel
// One of the exceptions is that ICC does not notify opponent about player's
// illegal move. Here he has to be notified about it because tests don't allow
// player to not recognise the difference between states with different move
// number. And Illegal attempt is considered a move.
// Other exceptions are 50-move rule and threefold repetition, which under ICC
// rules are not automatically enforced, but can be claimed by the player. This
// implementation does not support claiming or offering draws so these rules'
// automatic enforcement can be turned on and off
//
// Parameters:
//   "board_size"           int     Number of squares in each row and column
//                                  (default: 8)
//   "fen"                  string  String describing the chess board position
//                                  in Forsyth-Edwards Notation. The FEN has to
//                                  match the board size. Default values are
//                                  available for board sizes 4 and 8.
//   "threefold_repetition" bool    Whether threefold repetition rule should be
//                                  automatically enforced (default: true)
//   "50_move_rule"         bool    Whether 50 move rule should be automatically
//                                  enforced (default: true)

namespace open_spiel {
namespace kriegspiel {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr double kLossUtility = -1;
inline constexpr double kDrawUtility = 0;
inline constexpr double kWinUtility = 1;

// See action encoding below.
inline constexpr int kNumDistinctActions = 4672;

// This is max length of a FIDE chess game. Kriegspiel can be longer. It can
// last forever when the three fold repetition and 50-move rule are turned off.
// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
inline constexpr int kMaxGameLength = 17695;

enum KriegspielCaptureType { kNoCapture = 0, kPawn = 1, kPiece = 2 };

std::string CaptureTypeToString(KriegspielCaptureType capture_type);

enum KriegspielCheckType {
  kNoCheck = 0,
  kFile = 1,
  kRank = 2,
  kLongDiagonal = 3,
  kShortDiagonal = 4,
  kKnight = 5
};

std::string CheckTypeToString(KriegspielCheckType check_type);

std::pair<KriegspielCheckType, KriegspielCheckType> GetCheckType(
    const chess::ChessBoard& board);

struct KriegspielUmpireMessage {
  bool illegal = false;
  KriegspielCaptureType capture_type = KriegspielCaptureType::kNoCapture;
  chess::Square square = chess::kInvalidSquare;
  // there can be max two checks at a time so a pair is enough
  std::pair<KriegspielCheckType, KriegspielCheckType> check_types = {
      KriegspielCheckType::kNoCheck, KriegspielCheckType::kNoCheck};
  chess::Color to_move = chess::Color::kEmpty;
  int pawn_tries = 0;

  std::string ToString() const;

  bool operator==(KriegspielUmpireMessage& other) const {
    return illegal == other.illegal && capture_type == other.capture_type &&
           square == other.square && check_types == other.check_types &&
           to_move == other.to_move && pawn_tries == other.pawn_tries;
  }
};

KriegspielUmpireMessage GetUmpireMessage(const chess::ChessBoard& chess_board,
                                         const chess::Move& move);

bool GeneratesUmpireMessage(const chess::ChessBoard& chess_board,
                            const chess::Move& move,
                            const KriegspielUmpireMessage& orig_msg);

class KriegspielGame;
class KriegspielObserver;

// State of an in-play game.
class KriegspielState : public State {
 public:
  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  KriegspielState(std::shared_ptr<const Game> game, int board_size,
                  const std::string& fen, bool threefold_repetition,
                  bool rule_50_move);
  KriegspielState(const KriegspielState&) = default;

  KriegspielState& operator=(const KriegspielState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(Board().ToPlay());
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override { return MaybeFinalReturns().has_value(); }

  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  friend class KriegspielObserver;

  // Current board.
  chess::ChessBoard& Board() { return current_board_; }
  const chess::ChessBoard& Board() const { return current_board_; }
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  // History of moves and umpire messages.
  const std::vector<std::pair<chess::Move, KriegspielUmpireMessage>>&
  MoveMsgHistory() const {
    return move_msg_history_;
  }

  // Draw can be claimed under the FIDE threefold repetition rule (the current
  // board position has already appeared twice in the history).
  bool IsThreefoldRepetitionDraw() const;

  // Calculates legal actions and caches them. This is separate from
  // LegalActions() as there are a number of other methods that need the value
  // of LegalActions. This is a separate method as it's called from
  // IsTerminal(), which is also called by LegalActions().
  void MaybeGenerateLegalActions() const;

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<std::pair<chess::Move, KriegspielUmpireMessage>>
      move_msg_history_;
  // We store this info as an optimisation so that we don't have to compute it
  // from move_msg_history for observations
  absl::optional<KriegspielUmpireMessage> last_umpire_msg_{};
  // Moves that the player tried and were illegal. We don't let player try them
  // again on the same board because they are clearly still illegal;
  std::vector<chess::Move> illegal_tried_moves_;
  // We store the start board for history to support games not starting
  // from the start position.
  chess::ChessBoard start_board_;
  // We store the current board position as an optimization.
  chess::ChessBoard current_board_;

  bool threefold_repetition_;
  bool rule_50_move_;

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
class KriegspielGame : public Game {
 public:
  explicit KriegspielGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<KriegspielState>(shared_from_this(), board_size_,
                                              fen_, threefold_repetition_,
                                              rule_50_move_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kLossUtility; }
  double UtilitySum() const override { return kDrawUtility; }
  double MaxUtility() const override { return kWinUtility; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return kMaxGameLength; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;

  std::shared_ptr<KriegspielObserver> default_observer_;

 private:
  mutable std::vector<int> observation_tensor_shape_;
  const int board_size_;
  const std::string fen_;
  const bool threefold_repetition_;
  const bool rule_50_move_;
};

}  // namespace kriegspiel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_KRIEGSPIEL_H_
