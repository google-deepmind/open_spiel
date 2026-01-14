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

#ifndef OPEN_SPIEL_GAMES_MNK_MNK_H_
#define OPEN_SPIEL_GAMES_MNK_MNK_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

// m,n,k-game, also known as k-in-a-row game on an m-by-n board:
// https://en.wikipedia.org/wiki/M,n,k-game
//
// Parameters:
//  "m"  int  width of the board (i.e., number of columns)  (default = 15)
//  "n"  int  height of the board (i.e., number of rows)    (default = 15)
//  "k"  int  k-in-a-row win condition                      (default = 5)

namespace open_spiel {
namespace mnk {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.
inline constexpr int kDefaultNumRows = 15;
inline constexpr int kDefaultNumCols = 15;
inline constexpr int kDefaultNumInARow = 5;

// State of a cell.
enum class CellState {
  kEmpty,
  kNought,  // O
  kCross,   // X
};

// State of an in-play game.
class MNKState : public State {
 public:
  MNKState(std::shared_ptr<const Game> game);  // NOLINT

  MNKState(const MNKState&) = default;
  MNKState& operator=(const MNKState&) = default;

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
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  CellState BoardAt(int cell) const {
    auto [row, column] = ActionToCoordinates(cell);
    return board_[row][column];
  }
  CellState BoardAt(int row, int column) const { return board_[row][column]; }
  Player outcome() const { return outcome_; }
  std::pair<int, int> ActionToCoordinates(Action move) const;
  int CoordinatesToAction(int row, int column) const;
  int NumRows() const;
  int NumCols() const;
  int NumCells() const;
  int NumInARow() const;

  // Only used by Ultimate Tic-Tac-Toe.
  void SetCurrentPlayer(Player player) { current_player_ = player; }

 protected:
  std::vector<std::vector<CellState>> board_;
  void DoApplyAction(Action move) override;

 private:
  bool HasLine(Player player) const;  // Does this player have a line?
  bool IsFull() const;                // Is the board full?
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
};

// Game object.
class MNKGame : public Game {
 public:
  explicit MNKGame(const GameParameters& params);
  int NumDistinctActions() const override { return NumCells(); }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new MNKState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, NumRows(), NumCols()};
  }
  int MaxGameLength() const override { return NumCells(); }
  std::string ActionToString(Player player, Action action_id) const override;
  int NumRows() const { return ParameterValue<int>("n"); }
  int NumCols() const { return ParameterValue<int>("m"); }
  int NumCells() const { return NumRows() * NumCols(); }
  int NumInARow() const { return ParameterValue<int>("k"); }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

// Does this player have a line?
bool BoardHasLine(const std::vector<std::vector<CellState>>& board,
                  const Player player, int k, int r, int c, int dr, int dc);

bool BoardHasLine(const std::vector<std::vector<CellState>>& board,
                  const Player player, int k);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace mnk
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MNK_MNK_H_
