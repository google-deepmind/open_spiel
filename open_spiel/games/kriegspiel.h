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

#ifndef OPEN_SPIEL_GAMES_KRIEGSPIEL_H_
#define OPEN_SPIEL_GAMES_KRIEGSPIEL_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <open_spiel/fog/observation_history.h>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Kriegpsiel - imperfect information version of chess:
// https://en.wikipedia.org/wiki/Kriegspiel
//
// Parameters:
//   "board_size"  int     Number of squares in each row and column (default: 8)
//   "fen"         string  String describing the chess board position in
//                         Forsyth-Edwards Notation. The FEN has to match
//                         the board size. Default values are available for
//                         board sizes 4 and 8.

namespace open_spiel {
namespace kriegspiel {

// Constants.
inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

// See action encoding below.
inline constexpr int NumDistinctActions() { return 4672; }

// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
inline constexpr int MaxGameLength() { return 17695; }

enum KriegspielCaptureType : int8_t {
  kNoCapture = 0,
  kPawn = 1,
  kPiece = 2
};

std::string CaptureTypeToString(KriegspielCaptureType capture_type) {
  if (capture_type == KriegspielCaptureType::kNoCapture) {
    return "No Piece";
  }
  if (capture_type == KriegspielCaptureType::kPawn) {
    return "Pawn";
  }
  return "Piece";
}

enum KriegspielCheckType : int8_t {
  kNoCheck = 0,
  kFile = 1,
  kRank = 2,
  kLongDiagonal = 3,
  kShortDiagonal = 4,
  kKnight = 5
};

std::string CheckTypeToString(KriegspielCheckType check_type) {
  switch (check_type) {
    case KriegspielCheckType::kNoCheck:
      return "No";
    case KriegspielCheckType::kFile:
      return "File";
    case KriegspielCheckType::kRank:
      return "Rank";
    case KriegspielCheckType::kLongDiagonal:
      return "Long-diagonal";
    case KriegspielCheckType::kShortDiagonal:
      return "Short-diagonal";
    case KriegspielCheckType::kKnight:
      return "Knight";
    default:
      SpielFatalError("kNoCheck does not have a string representation");
  }
}

struct KriegspielUmpireMessage {

  bool illegal_ = false;
  KriegspielCaptureType capture_type_ = KriegspielCaptureType::kNoCapture;
  chess::Square square_ = chess::InvalidSquare();
  std::pair<KriegspielCheckType, KriegspielCheckType> check_types_ =
      {KriegspielCheckType::kNoCheck, KriegspielCheckType::kNoCheck};
  chess::Color to_move_ = chess::Color::kEmpty;
  int pawn_tries_ = 0;

  std::string ToString() const;
};

KriegspielUmpireMessage GetUmpireMessage(const chess::ChessBoard &chess_board,
                                         const chess::Move &move) {
  KriegspielUmpireMessage msg {};
  if (!chess_board.IsMoveLegal(move)) {
    // If the move is illegal, the player is notified about it and can play again
    msg.illegal_ = true;
    return msg;
  }
  msg.illegal_ = false;

  chess::PieceType capture_type = chess_board.at(move.to).type;
  switch (capture_type) {
    case chess::PieceType::kEmpty:
      msg.capture_type_ = KriegspielCaptureType::kNoCapture;
      msg.square_ = chess::InvalidSquare();
      break;
    case chess::PieceType::kPawn:
      msg.capture_type_ = KriegspielCaptureType::kPawn;
      msg.square_ = move.to;
      break;
    default:
      msg.capture_type_ = KriegspielCaptureType::kPiece;
      msg.square_ = move.to;
  }

  // todo optimze when undo is optimized
  chess::ChessBoard board_copy = chess_board;
  board_copy.ApplyMove(move);

  chess::Square king_sq = board_copy.find(
      chess::Piece{board_copy.ToPlay(), chess::PieceType::kKing});

  std::pair<KriegspielCheckType, KriegspielCheckType> check_type_pair =
      {KriegspielCheckType::kNoCheck, KriegspielCheckType::kNoCheck};

  board_copy.GeneratePseudoLegalMoves([&king_sq, &check_type_pair, &board_copy](const chess::Move &move) {
    if (move.to != king_sq) {
      return true;
    }
    KriegspielCheckType check_type;
    if (move.piece.type == chess::PieceType::kKnight)
      check_type = KriegspielCheckType::kKnight;
    else if (move.from.x == move.to.x)
      check_type = KriegspielCheckType::kFile;
    else if (move.from.y == move.to.y)
      check_type = KriegspielCheckType::kRank;
    else if (chess::IsLongDiagonal(move.from, move.to, board_copy.BoardSize()))
      check_type = KriegspielCheckType::kLongDiagonal;
    else
      check_type = KriegspielCheckType::kShortDiagonal;

    if (check_type_pair.first != KriegspielCheckType::kNoCheck) {
      // There cannot be more than two checks at the same time
      check_type_pair.second = check_type;
      return false;
    }
    else check_type_pair.first = check_type;

    return true;
  }, chess_board.ToPlay(), false);
  msg.check_types_ = check_type_pair;

  int pawnTries = 0;
  board_copy.GenerateLegalMoves([&board_copy, &pawnTries](const chess::Move &move) {
    if (move.piece.type == chess::PieceType::kPawn
        && board_copy.at(move.to).type != chess::PieceType::kEmpty) {
      pawnTries++;
    }
    return true;
  });
  msg.pawn_tries_ = pawnTries;
  msg.to_move_ = board_copy.ToPlay();

  return msg;
}

class KriegspielGame;
class KriegspielObserver;

// State of an in-play game.
class KriegspielState : public State {
 public:

  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  KriegspielState(std::shared_ptr<const Game> game,
                 int board_size, const std::string& fen);
  KriegspielState(const KriegspielState&) = default;

  KriegspielState& operator=(const KriegspielState&) = default;

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
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  chess::ChessBoard& StartBoard() { return start_board_; }
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  std::vector<std::pair<chess::Move, KriegspielUmpireMessage>>& MoveMsgHistory() {
    return move_msg_history_;
  }
  const std::vector<std::pair<chess::Move, KriegspielUmpireMessage>>& MoveMsgHistory() const {
    return move_msg_history_;
  }

 protected:
  void DoApplyAction(Action action) override;

 private:

  friend class KriegspielObserver;

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
  std::vector<std::pair<chess::Move, KriegspielUmpireMessage>> move_msg_history_;
  // We store this info as an optimisation so that we don't have to compute it
  // from move_msg_history for observations
  std::optional<KriegspielUmpireMessage> last_public_msg{};
  std::optional<KriegspielUmpireMessage> before_last_public_msg{};
  // Moves that the player tried and were illegal. We don't let player try them
  // again on the same board because they are clearly still illegal;
  std::set<chess::Move> illegal_tried_moves_;
  // We store the start board for history to support games not starting
  // from the start position.
  chess::ChessBoard start_board_;
  // We store the current board position as an optimization.
  chess::ChessBoard current_board_;

  // cached ActionObservationHistory for each player
  std::vector<open_spiel::ActionObservationHistory> aohs_;

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
  int NumDistinctActions() const override {
    return kriegspiel::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<KriegspielState>(shared_from_this(),
                                             board_size_, fen_);
  }
  int NumPlayers() const override { return kriegspiel::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  double UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    std::vector<int> shape{
        14 // private boards: piece types * colours + empty + unknown
         * board_size_ * board_size_ +
        3 + // public: repetitions count, one-hot encoding
        2 + // public: side to play
        1 + // public: irreversible move counter -- a fraction of $n over 100
        2 * ( // public: last two umpire messages
            3 + // capture type
            6 + // check type one
            6 + // check type two
            3 + // player to move
            16 + // pawn tries
            board_size_ * board_size_) + // capture square
        2*2 + // private: left/right castling rights, one-hot encoded.
        2 + // private: whether last move was illegal
        2 * (board_size_ * board_size_)// private: last move (from, two)
          + 6 // private: last move (promotion type)
    };
    return shape;
  }
  int MaxGameLength() const override { return kriegspiel::MaxGameLength(); }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;


  std::shared_ptr<KriegspielObserver> default_observer_;
  std::shared_ptr<KriegspielObserver> info_state_observer_;

 private:
  const int board_size_;
  const std::string fen_;
};

}  // namespace kriegspiel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_KRIEGSPIEL_H_
