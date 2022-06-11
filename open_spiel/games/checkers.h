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

#ifndef OPEN_SPIEL_GAMES_CHECKERS_H_
#define OPEN_SPIEL_GAMES_CHECKERS_H_

// Implementation of the board game Checkers.
// https://en.wikipedia.org/wiki/Checkers
//
// Some notes about this implementation:
// - Drawing: 
//     Game is drawn if no pieces have been removed in 40 moves

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace checkers {

inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultRows = 8;
inline constexpr int kDefaultColumns = 8;
inline constexpr int kMaxMovesWithoutCapture = 40;
inline constexpr int kCellStates = 5;  // Empty, White, WhiteCrowned, Black and BlackCrowned.

// State of a cell.
enum class CellState {
  kEmpty,         // Represented by ' '.
  kWhite,         // Represented by 'o'.
  kBlack,         // Represented by '+'.
  kWhiteCrowned,  // Represented by '8'.
  kBlackCrowned,  // Represented by '*'.
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
  void SetBoard(int row, int column, CellState state) {    
    board_[row * columns_ + column] = state;
  }
  CellState BoardAt(int row, int column) const {
    return board_[row * columns_ + column];
  }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  int ObservationPlane(CellState state, Player player) const;

  Player current_player_ = 0;  // Player zero (White, 'o') goes first.
  Player outcome_ = kInvalidPlayer;
  int rows_;
  int columns_;
  int moves_without_capture_;
  std::vector<CellState> board_;
};

// Game object.
class CheckersGame : public Game {
 public:
  explicit CheckersGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<CheckersState>(shared_from_this(), rows_, columns_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
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
