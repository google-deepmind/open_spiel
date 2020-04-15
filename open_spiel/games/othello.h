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
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"  // for c_fill
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
inline constexpr int kPassMove = kNumCells;

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
};

inline constexpr std::array<Direction, 8> kDirections = {
    kUp, kDown, kLeft, kRight, kUpLeft, kUpRight, kDownLeft, kDownRight};

struct Move {
  Move(int move) : row(move / kNumCols), col(move % kNumRows) {
    if (move < 0 || move >= kNumCells) {
      SpielFatalError(absl::StrCat("Move too large: ", move));
    }
  }

  Move(int row, int col) : row(row), col(col) {};

  inline int GetRow() const { return row; }
  inline int GetColumn() const { return col; }
  inline int GetAction() const { return row * kNumCols + col; }
  inline bool OnBoard() const;

  Move Next(Direction dir) const;

private:
  int row, col;
};

// State of an in-play game.
class OthelloState : public State {
 public:
  OthelloState(std::shared_ptr<const Game> game);

  OthelloState(const OthelloState&) = default;
  OthelloState& operator=(const OthelloState&) = default;

  Player CurrentPlayer() const override { return current_player_; }

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  std::string ToString(Player player) const;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 private:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

  CellState BoardAt(int row, int col) const {
    Move move(row, col);
    return board_[move.GetAction()];
  }

  CellState BoardAt(Move move) const { return BoardAt(move.GetRow(), move.GetColumn()); }

  std::string ToStringForPlayer(Player player) const;  // to string for a specific player

  // Returns a list of regular (non-pass) actions.
  std::vector<Action> LegalRegularActions(Player p) const;

  // Returns true if the move would be valid for player if it were their turn.
  bool ValidAction(Player player, int action) const;

  // Returns true if there are no actions available for either player.
  bool NoValidActions() const;

  // Returns the number of pieces on the board for the given player.
  int DiskCount(Player player) const;

  // Returns true if the specified move would result in a capture.
  bool CanCapture(Player player, int action) const;

  // Returns the number of capturable disks of the opponent in the given
  // direction from the given starting location.
  int CountSteps(Player player, int action, Direction dir) const;

  // Updates the board to reflect a capture move.
  void Capture(Player player, int action, Direction dir, int steps);

  Player current_player_ = 0;  // Player zero goes first
  Player outcome_ = kInvalidPlayer;
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
std::string StateToString(Player player, CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace othello
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OTHELLO_H_
