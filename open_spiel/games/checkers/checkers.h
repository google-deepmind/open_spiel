// Copyright 2022 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_CHECKERS_H_
#define OPEN_SPIEL_GAMES_CHECKERS_H_

// Implementation of the board game Checkers.
// https://en.wikipedia.org/wiki/Checkers
//
// Some notes about this implementation:
// - Capturing:
//     When capturing an opponent's piece is possible, capturing is mandatory
//     in this implementation.
// - Drawing:
//     Game is drawn if no pieces have been removed in 40 moves
//     http://www.flyordie.com/games/help/checkers/en/games_rules_checkers.html
// - Custom board dimensions:
//     Dimensions of the board can be customised by calling the
//     CheckersState(rows, columns) constructer with the desired
//     number of rows and columns

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace checkers {

constexpr int kNumPlayers = 2;
constexpr int kDefaultRows = 8;
constexpr int kDefaultColumns = 8;
constexpr int kMaxMovesWithoutCapture = 40;
// Empty, White, WhiteKing, Black and BlackKing.
constexpr int kCellStates = 5;
constexpr int kNoMultipleJumpsPossible = -1;

// State of a cell.
enum class CellState {
  kEmpty,      // Represented by ' '.
  kWhite,      // Represented by 'o'.
  kBlack,      // Represented by '+'.
  kWhiteKing,  // Represented by '8'.
  kBlackKing,  // Represented by '*'.
};

struct CheckersAction {
  int row;
  int column;
  int direction;
  int move_type;
  CheckersAction(int _row, int _column, int _direction, int _move_type)
      : row(_row),
        column(_column),
        direction(_direction),
        move_type(_move_type) {}
};

// Types of moves.
enum MoveType {
  kNormal = 0,
  kCapture = 1,
};

// Types of pieces.
enum PieceType {
  kMan = 0,
  kKing = 1,
};

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  Action action;
  Player player;
  // set to kMan if not a capture move
  PieceType captured_piece_type;
  PieceType player_piece_type;
  TurnHistoryInfo(Action _action, Player _player,
                  PieceType _captured_piece_type, PieceType _player_piece_type)
      : action(_action),
        player(_player),
        captured_piece_type(_captured_piece_type),
        player_piece_type(_player_piece_type) {}
};

// State of an in-play game.
class CheckersState : public State {
 public:
  explicit CheckersState(std::shared_ptr<const Game> game, int rows,
                         int columns);
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new CheckersState(*this));
  }
  void UndoAction(Player player, Action action) override;
  bool InBounds(int row, int column) const;
  void SetCustomBoard(const std::string board_string);
  CellState CrownStateIfLastRowReached(int row, CellState state);
  CheckersAction SpielActionToCheckersAction(Action action) const;
  Action CheckersActionToSpielAction(CheckersAction move) const;
  void SetBoard(int row, int column, CellState state) {
    board_[row * columns_ + column] = state;
  }
  CellState BoardAt(int row, int column) const {
    return board_[row * columns_ + column];
  }
  std::vector<Action> LegalActions() const override;
  int ObservationPlane(CellState state, Player player) const;
  int GetRow() const { return rows_; }
  int GetCollumn() const { return columns_; }
  int GetCellState() const { return kCellStates; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  Player current_player_ = 0;  // Player zero (White, 'o') goes first.
  Player outcome_ = kInvalidPlayer;
  // Piece in the board who can do multiple jump.
  // Represented by row * rows_ + column
  int multiple_jump_piece_ = kNoMultipleJumpsPossible;
  int rows_;
  int columns_;
  int moves_without_capture_;
  std::vector<CellState> board_;
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
};

// Game object.
class CheckersGame : public Game {
 public:
  explicit CheckersGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<CheckersState>(shared_from_this(), rows_,
                                            columns_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, rows_, columns_};
  }
  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

 private:
  int rows_;
  int columns_;
};

std::ostream& operator<<(std::ostream& stream, const CellState& state);

}  // namespace checkers
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHECKERS_H_
