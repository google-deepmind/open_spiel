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

#include "open_spiel/games/dots_and_boxes/dots_and_boxes.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace dots_and_boxes {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"dots_and_boxes",
    /*long_name=*/"Dots and Boxes",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_rows", GameParameter(kDefaultNumRows)},
     {"num_cols", GameParameter(kDefaultNumCols)},
     {"utility_margin", GameParameter(kDefaultUtilityMargin)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DotsAndBoxesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kPlayer1;
    case 1:
      return CellState::kPlayer2;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kPlayer1:
      return "1";
    case CellState::kPlayer2:
      return "2";
    default:
      SpielFatalError("Unknown state.");
  }
}

std::string OrientationToString(CellOrientation orientation) {
  switch (orientation) {
    case CellOrientation::kHorizontal:
      return "h";
    case CellOrientation::kVertical:
      return "v";
    default:
      SpielFatalError("Unknown orientation.");
  }
}

// Move Methods ================================================================

Move::Move(int row, int col, CellOrientation orientation, int rows, int cols) {
  row_ = row;
  col_ = col;
  orientation_ = orientation;
  num_rows_ = rows;
  num_cols_ = cols;
}

Move::Move() {
  row_ = 0;
  col_ = 0;
  orientation_ = CellOrientation::kVertical;
  num_rows_ = 0;
  num_cols_ = 0;
}

Move::Move(Action action, int rows, int cols) {
  num_rows_ = rows;
  num_cols_ = cols;
  int maxh = (num_rows_ + 1) * num_cols_;
  if (action < maxh) {
    // Horizontal
    orientation_ = CellOrientation::kHorizontal;
    row_ = action / num_cols_;
    col_ = action % num_cols_;
  } else {
    // Vertical
    action -= maxh;
    orientation_ = CellOrientation::kVertical;
    row_ = action / (num_cols_ + 1);
    col_ = action % (num_cols_ + 1);
  }
  SPIEL_CHECK_LT(row_, num_rows_ + 1);
  SPIEL_CHECK_LT(col_, num_cols_ + 1);
}

void Move::Set(int row, int col, CellOrientation orientation) {
  row_ = row;
  col_ = col;
  SPIEL_CHECK_LT(row_, num_rows_ + 1);
  SPIEL_CHECK_LT(col_, num_cols_ + 1);
  orientation_ = orientation;
}

int Move::GetRow() const { return row_; }
int Move::GetCol() const { return col_; }
CellOrientation Move::GetOrientation() const { return orientation_; }

Action Move::ActionId() {
  // First bit is horizontal (0) or vertical (1)
  Action action = 0;
  int maxh = (num_rows_ + 1) * num_cols_;
  if (orientation_ == CellOrientation::kHorizontal) {
    action = row_ * num_cols_ + col_;
  } else {
    action = maxh + row_ * (num_cols_ + 1) + col_;
  }
  return action;
}

int Move::GetCell() { return row_ * (num_cols_ + 1) + col_; }

int Move::GetCellLeft() {
  if (col_ == 0) {
    return -1;
  }
  return row_ * (num_cols_ + 1) + (col_ - 1);
}

int Move::GetCellRight() {
  if (col_ == num_cols_) {
    return -1;
  }
  return row_ * (num_cols_ + 1) + (col_ + 1);
}

int Move::GetCellAbove() {
  if (row_ == 0) {
    return -1;
  }
  return (row_ - 1) * (num_cols_ + 1) + col_;
}

int Move::GetCellBelow() {
  if (row_ == num_rows_) {
    return -1;
  }
  return (row_ + 1) * (num_cols_ + 1) + col_;
}

int Move::GetCellAboveLeft() {
  if (row_ == 0 || col_ == 0) {
    return -1;
  }
  return (row_ - 1) * (num_cols_ + 1) + (col_ - 1);
}

int Move::GetCellAboveRight() {
  if (row_ == 0 || col_ == num_cols_) {
    return -1;
  }
  return (row_ - 1) * (num_cols_ + 1) + (col_ + 1);
}

int Move::GetCellBelowLeft() {
  if (row_ == num_rows_ || col_ == 0) {
    return -1;
  }
  return (row_ + 1) * (num_cols_ + 1) + (col_ - 1);
}

int Move::GetCellBelowRight() {
  if (row_ == num_rows_ || col_ == num_cols_) {
    return -1;
  }
  return (row_ + 1) * (num_cols_ + 1) + (col_ + 1);
}

// DotsAndBoxesState Methods ===================================================

void DotsAndBoxesState::DoApplyAction(Action action) {
  Move move = Move(action, num_rows_, num_cols_);
  int cell = move.GetCell();
  bool won_cell = false;
  if (move.GetOrientation() == CellOrientation::kVertical) {
    SPIEL_CHECK_EQ(v_[cell], CellState::kEmpty);
    v_[cell] = PlayerToState(CurrentPlayer());

    // Left
    if (move.GetCol() > 0) {
      if (v_[move.GetCellLeft()] != CellState::kEmpty &&
          h_[move.GetCellLeft()] != CellState::kEmpty &&
          h_[move.GetCellBelowLeft()] != CellState::kEmpty) {
        won_cell = true;
        p_[move.GetCellLeft()] = PlayerToState(CurrentPlayer());
        points_[current_player_]++;
      }
    }

    // Right
    if (move.GetCol() < num_cols_) {
      if (v_[move.GetCellRight()] != CellState::kEmpty &&
          h_[move.GetCellBelow()] != CellState::kEmpty &&
          h_[cell] != CellState::kEmpty) {
        won_cell = true;
        p_[cell] = PlayerToState(CurrentPlayer());
        points_[current_player_]++;
      }
    }

  } else {  // move.GetOrientation() == kHorizontal
    SPIEL_CHECK_EQ(h_[cell], CellState::kEmpty);
    h_[cell] = PlayerToState(CurrentPlayer());

    // Above
    if (move.GetRow() > 0) {
      if (v_[move.GetCellAbove()] != CellState::kEmpty &&
          v_[move.GetCellAboveRight()] != CellState::kEmpty &&
          h_[move.GetCellAbove()] != CellState::kEmpty) {
        won_cell = true;
        p_[move.GetCellAbove()] = PlayerToState(CurrentPlayer());
        points_[current_player_]++;
      }
    }
    // Below
    if (move.GetRow() < num_rows_) {
      if (v_[cell] != CellState::kEmpty &&
          v_[move.GetCellRight()] != CellState::kEmpty &&
          h_[move.GetCellBelow()] != CellState::kEmpty) {
        won_cell = true;
        p_[cell] = PlayerToState(CurrentPlayer());
        points_[current_player_]++;
      }
    }
  }

  if (Wins(current_player_)) {
    outcome_ = current_player_;
  }
  if (!won_cell) {
    // If box is scored, current player keeps the turn
    current_player_ = 1 - current_player_;
  }
  num_moves_ += 1;
}

std::vector<Action> DotsAndBoxesState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> actions;
  int action = 0;
  Move move;
  move.SetRowsCols(num_rows_, num_cols_);
  int maxh = (num_rows_ + 1) * num_cols_;
  int maxv = num_rows_ * (num_cols_ + 1);
  // Horizontal lines
  for (int row = 0; row <= num_rows_; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      move.Set(row, col, CellOrientation::kHorizontal);
      if (h_[move.GetCell()] == CellState::kEmpty) {
        actions.push_back(action);
      } else {
      }
      action++;
    }
  }
  SPIEL_CHECK_EQ(action, maxh);
  // Vertical lines
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col <= num_cols_; ++col) {
      move.Set(row, col, CellOrientation::kVertical);
      if (v_[move.GetCell()] == CellState::kEmpty) {
        actions.push_back(action);
      } else {
      }
      // std::cout << action << std::endl;
      action++;
    }
  }
  SPIEL_CHECK_EQ(action, maxh + maxv);
  return actions;
}

std::string DotsAndBoxesState::DbnString() const {
  // A string representing which lines have been set.
  // This corresponds to an unscored state representation
  // (Barker and Korf 2012).
  // For a scored state, use the ObservationTensor function.
  std::string str;
  int cell = 0;
  int idx = 0;
  for (int row = 0; row < num_rows_ + 1; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      if (h_[cell] != CellState::kEmpty) {
        absl::StrAppend(&str, "1");
      } else {
        absl::StrAppend(&str, "0");
      }
      idx++;
      cell++;
    }
    cell++;
  }
  cell = 0;
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_ + 1; ++col) {
      if (v_[cell] != CellState::kEmpty) {
        absl::StrAppend(&str, "1");
      } else {
        absl::StrAppend(&str, "0");
      }
      idx++;
      cell++;
    }
  }
  return str;
}

std::string DotsAndBoxesState::ActionToString(Player player,
                                              Action action_id) const {
  Move move(action_id, num_rows_, num_cols_);
  return absl::StrCat("P", StateToString(PlayerToState(player)), "(",
                      OrientationToString(move.GetOrientation()), ",",
                      move.GetRow(), ",", move.GetCol(), ")");
}

bool DotsAndBoxesState::Wins(Player player) const {
  if (IsFull()) {
    // Game over
    if (PlayerToState(player) == CellState::kPlayer1) {
      return points_[0] > points_[1];
    } else {
      return points_[0] < points_[1];
    }
  }
  return false;
}

bool DotsAndBoxesState::IsFull() const {
  return num_moves_ ==
         (num_rows_ + 1) * num_cols_ + num_rows_ * (num_cols_ + 1);
}

DotsAndBoxesState::DotsAndBoxesState(std::shared_ptr<const Game> game,
                                     int num_rows, int num_cols,
                                     bool utility_margin)
    : State(game),
      num_rows_(num_rows),
      num_cols_(num_cols),
      num_cells_((1 + num_rows) * (1 + num_cols)),
      utility_margin_(utility_margin) {
  SPIEL_CHECK_GE(num_rows_, 1);
  SPIEL_CHECK_GE(num_cols_, 1);
  h_.resize(num_cells_);
  v_.resize(num_cells_);
  p_.resize(num_cells_);
  std::fill(begin(h_), end(h_), CellState::kEmpty);
  std::fill(begin(v_), end(v_), CellState::kEmpty);
  std::fill(begin(p_), end(p_), CellState::kEmpty);
  std::fill(begin(points_), end(points_), 0);
}

// Create initial board from the Dots-and-Boxes Notation.
// A vector with:
// [b | for r in [0,num_rows+1], for c in [0,num_cols]:
//      b=1 if horizontal line[r,c] set else 0] +
// [b | for r in [0,num_rows_], for c in [0,num_cols+1]:
//      b=1 if vertical line[r,c] set else 0]
DotsAndBoxesState::DotsAndBoxesState(std::shared_ptr<const Game> game,
                                     int num_rows, int num_cols,
                                     bool utility_margin,
                                     const std::string& dbn)
    : State(game),
      num_rows_(num_rows),
      num_cols_(num_cols),
      num_cells_((1 + num_rows) * (1 + num_cols)),
      utility_margin_(utility_margin) {
  /* std::cout << "Init dots and boxes state with dbn\n"; */
  SPIEL_CHECK_GE(num_rows_, 1);
  /* SPIEL_CHECK_LE(num_rows_, 1000); */
  SPIEL_CHECK_GE(num_cols_, 1);
  /* SPIEL_CHECK_LE(num_cols_, 1000); */
  h_.resize(num_cells_);
  v_.resize(num_cells_);
  p_.resize(num_cells_);
  std::fill(begin(h_), end(h_), CellState::kEmpty);
  std::fill(begin(v_), end(v_), CellState::kEmpty);
  std::fill(begin(p_), end(p_), CellState::kEmpty);
  std::fill(begin(points_), end(points_), 0);
  int cell = 0;
  int idx = 0;
  for (int row = 0; row < num_rows_ + 1; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      if (dbn[idx] == '1') {
        h_[cell] = CellState::kSet;
        num_moves_++;
      }
      idx++;
      cell++;
    }
    cell++;
  }
  cell = 0;
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_ + 1; ++col) {
      if (dbn[idx] == '1') {
        v_[cell] = CellState::kSet;
        num_moves_++;
      }
      idx++;
      cell++;
    }
  }
  int max_moves = (num_rows_ + 1) * num_cols_ + num_rows_ * (num_cols_ + 1);
  SPIEL_CHECK_LE(num_moves_, max_moves);
}

std::string DotsAndBoxesState::ToString() const {
  std::string str;
  int cell = 0;
  int cell_start = 0;
  for (int r = 0; r < num_rows_; ++r) {
    cell_start = cell;
    for (int c = 0; c <= num_cols_; ++c) {
      absl::StrAppend(&str, StateToStringH(h_[cell], r, c));
      cell++;
    }
    absl::StrAppend(&str, "\n");
    cell = cell_start;
    for (int c = 0; c < num_cols_; ++c) {
      absl::StrAppend(&str, StateToStringV(v_[cell], r, c));
      absl::StrAppend(&str, StateToStringP(p_[cell], r, c));
      cell++;
    }
    absl::StrAppend(&str, StateToStringV(v_[cell], r, num_cols_));
    cell++;
    absl::StrAppend(&str, "\n");
  }
  for (int c = 0; c <= num_cols_; ++c) {
    absl::StrAppend(&str, StateToStringH(h_[cell], num_rows_, c));
    cell++;
  }
  absl::StrAppend(&str, "\n");
  return str;
}

bool DotsAndBoxesState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> DotsAndBoxesState::Returns() const {
  if (utility_margin_) {
    if (IsTerminal()) {
      double margin = (double)(points_[0] - points_[1]);
      return {margin, -margin};
    } else {
      return {0.0, 0.0};
    }
  } else {
    if (Wins(Player{0})) {
      return {1.0, -1.0};
    } else if (Wins(Player{1})) {
      return {-1.0, 1.0};
    } else {
      // Game is not finished
      return {0.0, 0.0};
    }
  }
}

std::string DotsAndBoxesState::InformationStateString(Player player) const {
  // Cannot be used when starting from a non-empty initial state.
  // If the game is started from a non-empty initial state
  // there are no previous moves and thus the history is empty.
  // And moves cannot be inferred as different orderings can lead
  // to different scores for the players.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string DotsAndBoxesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void DotsAndBoxesState::ObservationTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 3-d tensor.
  TensorView<3> view(values,
                     {/*cellstates=*/3, num_cells_,
                      /*part of cell (h, v, p)=*/3},
                     true);
  for (int cell = 0; cell < num_cells_; ++cell) {
    view[{static_cast<int>(h_[cell]), cell, 0}] = 1.0;
    view[{static_cast<int>(v_[cell]), cell, 1}] = 1.0;
    view[{static_cast<int>(p_[cell]), cell, 2}] = 1.0;
  }
}

void DotsAndBoxesState::UndoAction(Player player, Action action) {
  Move move(action, num_rows_, num_cols_);
  int cell = move.GetCell();
  if (p_[cell] != CellState::kEmpty) {
    points_[current_player_]--;
  }
  h_[cell] = CellState::kEmpty;
  v_[cell] = CellState::kEmpty;
  p_[cell] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> DotsAndBoxesState::Clone() const {
  return std::unique_ptr<State>(new DotsAndBoxesState(*this));
}

std::string DotsAndBoxesState::StateToStringH(CellState state, int row,
                                              int col) const {
  if (row == 0 && col == 0) {
    if (state == CellState::kEmpty) {
      return "┌╴ ╶";
    } else {
      return "┌───";
    }
  }
  if (row == num_rows_ && col == 0) {
    if (state == CellState::kEmpty) {
      return "└╴ ╶";
    } else {
      return "└───";
    }
  }
  if (row == 0 && col == num_cols_) {
    return "┐";
  }
  if (row == num_rows_ && col == num_cols_) {
    return "┘";
  }
  if (col == num_cols_) {
    return "┤";
  }
  if (col == 0) {
    if (state == CellState::kEmpty) {
      return "├╴ ╶";
    } else {
      return "├───";
    }
  }
  if (row == 0) {
    if (state == CellState::kEmpty) {
      return "┬╴ ╶";
    } else {
      return "┬───";
    }
  }
  if (row == num_rows_) {
    if (state == CellState::kEmpty) {
      return "┴╴ ╶";
    } else {
      return "┴───";
    }
  }
  if (state == CellState::kEmpty) {
    return "┼╴ ╶";
  } else {
    return "┼───";
  }
}

std::string DotsAndBoxesState::StateToStringV(CellState state, int row,
                                              int col) const {
  if (state == CellState::kEmpty) {
    return " ";  // "┊";
  } else {
    return "│";
  }
}

std::string DotsAndBoxesState::StateToStringP(CellState state, int row,
                                              int col) const {
  if (state == CellState::kEmpty) {
    return "   ";
  }
  if (state == CellState::kPlayer1) {
    return " 1 ";
  }
  if (state == CellState::kPlayer2) {
    return " 2 ";
  }
  return " x ";
}

DotsAndBoxesGame::DotsAndBoxesGame(const GameParameters& params)
    : Game(kGameType, params),
      num_rows_(ParameterValue<int>("num_rows", kDefaultNumRows)),
      num_cols_(ParameterValue<int>("num_cols", kDefaultNumCols)),
      num_cells_((1 + ParameterValue<int>("num_rows", kDefaultNumRows)) *
                 (1 + ParameterValue<int>("num_cols", kDefaultNumCols))),
      utility_margin_(
          ParameterValue<bool>("utility_margin", kDefaultUtilityMargin)) {
}

double DotsAndBoxesGame::MinUtility() const {
  // If win/lose is the utility, this is -1.
  if (utility_margin_) {
    return -num_rows_ * num_cols_;
  } else {
    return -1;
  }
}

absl::optional<double> DotsAndBoxesGame::UtilitySum() const {
  return 0;
}

double DotsAndBoxesGame::MaxUtility() const {
  if (utility_margin_) {
    return num_rows_ * num_cols_;
  } else {
    return 1;
  }
}

}  // namespace dots_and_boxes
}  // namespace open_spiel
