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
//
// Contributed by Wannes Meert, Giuseppe Marra, and Pieter Robberechts
// for the KU Leuven course Machine Learning: Project.

#ifndef OPEN_SPIEL_GAMES_DOTS_AND_BOXES_H_
#define OPEN_SPIEL_GAMES_DOTS_AND_BOXES_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Dots and Boxes:
// https://en.wikipedia.org/wiki/Dots_and_Boxes
//
// Parameters:
// - num_rows: Number of rows on the board
// - num_cols: Number of columns on the board
// - utility_margin: Return as payoff the margin achieved (if true) or
//                   return -1/0/1 to indicate win/tie/loss.

namespace open_spiel {
namespace dots_and_boxes {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultNumRows = 2;
inline constexpr int kDefaultNumCols = 2;
inline constexpr int kMaskSize = 10;
inline constexpr int kMask = (1 << kMaskSize) - 1;
inline constexpr bool kDefaultUtilityMargin = false;

// State of a cell.
enum class CellState {
  kEmpty,    // Not set
  kPlayer1,  // Set by player 1
  kPlayer2,  // Set by player 2
  kSet       // Set by default start state
};

enum class CellOrientation {
  kHorizontal,  // = 0
  kVertical,    // = 1
};

class Move {
 public:
  Move(void);
  Move(int row, int col, CellOrientation orientation, int rows, int cols);
  explicit Move(Action action, int rows, int cols);

  void SetRowsCols(int rows, int cols) {
    num_rows_ = rows;
    num_cols_ = cols;
  }
  void Set(int row, int col, CellOrientation orientation);
  int GetRow() const;
  int GetCol() const;
  CellOrientation GetOrientation() const;

  Action ActionId();
  int GetCell();
  int GetCellLeft();
  int GetCellRight();
  int GetCellAbove();
  int GetCellBelow();
  int GetCellAboveLeft();
  int GetCellAboveRight();
  int GetCellBelowLeft();
  int GetCellBelowRight();

 protected:
  int row_;
  int col_;
  CellOrientation orientation_;
  int num_rows_;
  int num_cols_;
};

// State of an in-play game.
class DotsAndBoxesState : public State {
 public:
  DotsAndBoxesState(std::shared_ptr<const Game> game, int num_rows,
                    int num_cols, bool utility_margin);
  DotsAndBoxesState(std::shared_ptr<const Game> game, int num_rows,
                    int num_cols, bool utility_margin, const std::string& dbn);
  DotsAndBoxesState(const DotsAndBoxesState&) = default;
  DotsAndBoxesState& operator=(const DotsAndBoxesState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string DbnString() const;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  Player outcome() const { return outcome_; }

  std::string StateToStringV(CellState state, int row, int col) const;
  std::string StateToStringH(CellState state, int row, int col) const;
  std::string StateToStringP(CellState state, int row, int col) const;

  void SetCurrentPlayer(Player player) { current_player_ = player; }

 protected:
  std::vector<CellState> v_;  // Who set the vertical line
  std::vector<CellState> h_;  // Who set the horizontal line
  std::vector<CellState> p_;  // Who won the cell
  void DoApplyAction(Action action) override;

 private:
  bool Wins(Player player) const;
  bool IsFull() const;
  Player current_player_ = 0;  // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  const int num_rows_;
  const int num_cols_;
  const int num_cells_;
  std::array<int, 2> points_;
  const bool utility_margin_;
};

// Game object.
class DotsAndBoxesGame : public Game {
 public:
  explicit DotsAndBoxesGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return (num_rows_ + 1) * num_cols_ + num_rows_ * (num_cols_ + 1);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DotsAndBoxesState(
        shared_from_this(), num_rows_, num_cols_, utility_margin_));
  }
  std::unique_ptr<State> NewInitialState(
      const std::string& str) const override {
    return std::make_unique<DotsAndBoxesState>(shared_from_this(), num_rows_,
                                               num_cols_, utility_margin_, str);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override;
  absl::optional<double> UtilitySum() const override;
  double MaxUtility() const override;
  std::vector<int> ObservationTensorShape() const override {
    return {3, num_cells_, 3};
  }
  int MaxGameLength() const override {
    return (num_rows_ + 1) * num_cols_ + num_cols_ * (num_rows_ + 1);
  }

 private:
  const int num_rows_;
  const int num_cols_;
  const int num_cells_;
  const bool utility_margin_;
};

// CellState PlayerToState(Player player);
std::string StateToString(CellState state);
std::string OrientationToString(CellOrientation orientation);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace dots_and_boxes
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOTS_AND_BOXES_H_
