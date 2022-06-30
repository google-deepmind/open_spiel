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

#ifndef OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_
#define OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters: none

namespace open_spiel {
namespace tic_tac_toe {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 3;
inline constexpr int kNumCols = 3;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.

// https://math.stackexchange.com/questions/485752/tictactoe-state-space-choose-calculation/485852
inline constexpr int kNumberStates = 5478;

// State of a cell.
enum class CellState {
  kEmpty,
  kNought,  // O
  kCross,   // X
};

// State of an in-play game.
class TicTacToeState : public State {
 public:
  TicTacToeState(std::shared_ptr<const Game> game);

  TicTacToeState(const TicTacToeState&) = default;
  TicTacToeState& operator=(const TicTacToeState&) = default;

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
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }
  Player outcome() const { return outcome_; }

  // Only used by Ultimate Tic-Tac-Toe.
  void SetCurrentPlayer(Player player) { current_player_ = player; }

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

 private:
  bool HasLine(Player player) const;  // Does this player have a line?
  bool IsFull() const;                // Is the board full?
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
};

// Game object.
class TicTacToeGame : public Game {
 public:
  explicit TicTacToeGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCells; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TicTacToeState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kNumCells; }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

// Does this player have a line?
bool BoardHasLine(const std::array<CellState, kNumCells>& board,
                  const Player player);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace tic_tac_toe
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_
