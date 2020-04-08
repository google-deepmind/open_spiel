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

#ifndef OPEN_SPIEL_GAMES_OTHELLO_H_
#define OPEN_SPIEL_GAMES_OTHELLO_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Othello:
// https://en.wikipedia.org/wiki/Reversi
//
// Parameters: none

namespace open_spiel {
namespace othello {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 8;
inline constexpr int kNumCols = 8;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.
inline constexpr int passMove = kNumCells;

// State of a cell.
enum class CellState {
  kEmpty,
  kBlack,
  kWhite,
};

enum Direction {
  kUp,
  kDown,
  kLeft,
  kRight,
  kUpLeft,
  kUpRight,
  kDownLeft,
  kDownRight,
  kLast
};

// State of an in-play game.
class OthelloState : public State {
 public:
  OthelloState(std::shared_ptr<const Game> game);

  OthelloState(const OthelloState&) = default; // can default for reversi
  OthelloState& operator=(const OthelloState&) = default;

  Player CurrentPlayer() const override { return current_player_; }

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

 private: 
  int CountSteps(Player player, int row, int col, Direction dir) const;
  std::vector<Action> LegalRegularActions(Player p) const;  // list of regular (non-pass) actions
  bool ValidAction(Player player, int move) const; // check if a valid move
  int DiskCount(Player player) const;  // number of disk for each player
  bool CanCapture(Player player, int move) const;  // can capture by making move
  void Capture(Player player, int row, int col, Direction dir, int steps);  // capture row, col in direction
  bool IsFull() const;                // Is the board full?
  bool NoValidActions() const;  // no moves possible for either player
  inline bool OnBoard(int row, int col) const;  // is row and col on the boad?
  std::tuple<int, int> XYFromCode(int move) const;  // return (row, col) from move code
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
};

// Game object.
class OthelloGame : public Game {
 public:
  explicit OthelloGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCells + 1; }  // can pass
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new OthelloState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new OthelloGame(*this));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kNumCells; }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace othello
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OTHELLO_H_
