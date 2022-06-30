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

#ifndef OPEN_SPIEL_GAMES_CONNECT_FOUR_H_
#define OPEN_SPIEL_GAMES_CONNECT_FOUR_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Connect Four
// https://en.wikipedia.org/wiki/Connect_Four
//
// Minimax values (win/loss/draw) available for first 8 moves, here:
// https://archive.ics.uci.edu/ml/datasets/Connect-4
//
// Parameters: none

namespace open_spiel {
namespace connect_four {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kRows = 6;
inline constexpr int kCols = 7;
inline constexpr int kNumCells = kRows * kCols;
inline constexpr int kCellStates =
    1 + kNumPlayers;  // player 0, player 1, empty

// Outcome of the game.
enum class Outcome {
  kPlayer1 = 0,
  kPlayer2 = 1,
  kUnknown,
  kDraw,
};

// State of a cell.
enum class CellState {
  kEmpty,
  kNought,
  kCross,
};

// State of an in-play game.
class ConnectFourState : public State {
 public:
  ConnectFourState(std::shared_ptr<const Game>);
  explicit ConnectFourState(std::shared_ptr<const Game> game,
                            const std::string& str);
  ConnectFourState(const ConnectFourState& other) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override {
    return Clone();
  }

 protected:
  void DoApplyAction(Action move) override;

 private:
  CellState& CellAt(int row, int col);
  CellState CellAt(int row, int col) const;
  bool HasLine(Player player) const;  // Does this player have a line?
  bool HasLineFrom(Player player, int row, int col) const;
  bool HasLineFromInDirection(Player player, int row, int col, int drow,
                              int dcol) const;
  bool IsFull() const;         // Is the board full?
  Player current_player_ = 0;  // Player zero goes first
  Outcome outcome_ = Outcome::kUnknown;
  std::array<CellState, kNumCells> board_;
};

// Game object.
class ConnectFourGame : public Game {
 public:
  explicit ConnectFourGame(const GameParameters& params);
  int NumDistinctActions() const override { return kCols; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ConnectFourState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kRows, kCols};
  }
  int MaxGameLength() const override { return kNumCells; }
};

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  switch (state) {
    case CellState::kEmpty:
      return stream << "Empty";
    case CellState::kNought:
      return stream << "O";
    case CellState::kCross:
      return stream << "X";
    default:
      SpielFatalError("Unknown cell state");
  }
}

}  // namespace connect_four
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CONNECT_FOUR_H_
