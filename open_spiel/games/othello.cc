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

#include "open_spiel/games/othello.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <tuple>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace othello {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"othello",
    /*long_name=*/"Othello",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OthelloGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

std::tuple<int, int> GetNext(int row, int col, Direction dir) {
  switch (dir) {
    case Direction::kUp:
      return std::make_tuple(row - 1, col);
    case Direction::kDown:
      return std::make_tuple(row + 1, col);
    case Direction::kLeft:
      return std::make_tuple(row, col - 1);
    case Direction::kRight:
      return std::make_tuple(row, col + 1);
    case Direction::kUpRight:
      return std::make_tuple(row - 1, col + 1);
    case Direction::kUpLeft:
      return std::make_tuple(row - 1, col - 1);
    case Direction::kDownRight:
      return std::make_tuple(row + 1, col + 1);
    case Direction::kDownLeft:
      return std::make_tuple(row + 1, col - 1);
    default:
      SpielFatalError(absl::StrCat("Found unmatched case in GetNext."));
  }
}

inline bool OthelloState::OnBoard(int row, int col) const {
  return (0 <= row) && (row < kNumRows) && (0 <= col) && (col < kNumCols);
}

int OthelloState::CountSteps(Player player, int row, int col, Direction dir) const {
  std::tuple<int, int> rowcol = GetNext(row, col, dir);
  row = std::get<0>(rowcol);
  col = std::get<1>(rowcol);

  int count = 0;
  CellState cell = PlayerToState(player);
  while (OnBoard(row, col)) {
    if (BoardAt(row, col) == cell) {
      return count;
    } else if (BoardAt(row, col) == CellState::kEmpty) {
      return 0;
    }
      
    count++;
    rowcol = GetNext(row, col, dir);
    row = std::get<0>(rowcol);
    col = std::get<1>(rowcol);
  }

  return 0;
}

bool OthelloState::CanCapture(Player player, int move) const {
  if (board_[move] != CellState::kEmpty) return false;

  std::tuple<int, int> row_col = XYFromCode(move);
  int row = std::get<0>(row_col);
  int col = std::get<1>(row_col);

  // bool none_adjacent = true;
  // for (int i = -1; i < 2; i++) {
  //   for (int j = -1; j < 2; j++) {
  //     if (OnBoard(row + i, col + j) && (BoardAt(row + i, col + j) != CellState::kEmpty)) none_adjacent = false;
  //   }
  // }

  // if (none_adjacent) return false;

  bool valid = false;
  for (int direction = Direction::kUp; direction < Direction::kLast; direction++) {
    if (CountSteps(player, row, col, static_cast<Direction>(direction)) != 0) {
      return true;
    }
  }

  return false;
}

void OthelloState::Capture(Player player, int row, int col, Direction dir, int steps) {
  std::tuple<int, int> rowcol = GetNext(row, col, dir);
  row = std::get<0>(rowcol);
  col = std::get<1>(rowcol);
  CellState cell = PlayerToState(player);

  for (int step = 0; step < steps; step++) {
    if (BoardAt(row, col) == CellState::kEmpty || BoardAt(row, col) == cell) {
      SpielFatalError(absl::StrCat("Cannot capture cell (", row, ", ", col, ")"));
    }

    board_[row * kNumCols + col] = cell;
    
    rowcol = GetNext(row, col, dir);
    row = std::get<0>(rowcol);
    col = std::get<1>(rowcol);
  }
}

int OthelloState::DiskCount(Player player) const {
  int count = 0;
  CellState cell = PlayerToState(player);
  for (int i = 0; i < kNumCells; i++) {
    if (board_[i] == cell) count++;
  }

  return count;
}

bool OthelloState::NoValidActions() const {
  return (LegalRegularActions(Player(0)).empty() & LegalRegularActions(Player(1)).empty());
}

std::tuple<int, int> OthelloState::XYFromCode(int move) const {
  if (move >= kNumCells || move < 0) {
    SpielFatalError(absl::StrCat("Move too large: ", move));
  }

  return std::make_tuple(move / kNumCols, move % kNumRows);
}

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kBlack;
    case 1:
      return CellState::kWhite;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kWhite:
      return "o";
    case CellState::kBlack:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

bool OthelloState::ValidAction(Player player, int move) const {
  return (board_[move] == CellState::kEmpty && CanCapture(player, move));
}

void OthelloState::DoApplyAction(Action move) {
  if (move == passMove) {  // pass
    current_player_ = 1 - current_player_;
    num_moves_ += 1;
    return;
  }

  if (!ValidAction(current_player_, move)) {
    SpielFatalError(absl::StrCat("Invalid move ", move));
  }

  std::tuple<int, int> rowcol = XYFromCode(move);
  CellState cell = PlayerToState(CurrentPlayer());

  int row = std::get<0>(rowcol);
  int col = std::get<1>(rowcol);
  board_[row * kNumCols + col] = cell;

  for (int direction = Direction::kUp; direction < Direction::kLast; direction++) {
    int steps = 0;
    if ((steps = CountSteps(CurrentPlayer(), row, col, static_cast<Direction>(direction))) > 0) {
      Capture(CurrentPlayer(), row, col, static_cast<Direction>(direction), steps);
    }
  }

  if (NoValidActions()) {
    int count_zero = DiskCount(Player(0));
    int count_one = DiskCount(Player(1));

    if (count_zero > count_one) {
      outcome_ = Player(0);
    } else if (count_zero < count_one) {
      outcome_ = Player(1);
    }
  }

  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> OthelloState::LegalRegularActions(Player p) const {  // list 
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (ValidAction(p, cell)) {
      moves.push_back(cell);
    }
  }

  return moves;
}

std::vector<Action> OthelloState::LegalActions() const {
  if (IsTerminal()) return {};
  
  // can move in an empty cell that captures
  std::vector<Action> moves = LegalRegularActions(CurrentPlayer());
  if (moves.empty()) {
    moves.push_back(passMove);  // pass
  }

  return moves;
}

std::string OthelloState::ActionToString(Player player,
                                           Action action_id) const {

  if (action_id == passMove) {
    return absl::StrCat(StateToString(PlayerToState(player)), "(pass)");
  } else {
    return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
  }
}

bool OthelloState::IsFull() const { 
  for (int i = 0; i < kNumCells; i++) {
    if (board_[i] == CellState::kEmpty) return false;
  }

  return true;
}

OthelloState::OthelloState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
  board_[27] = CellState::kWhite;
  board_[28] = CellState::kBlack;
  board_[35] = CellState::kBlack;
  board_[36] = CellState::kWhite;
}

std::string OthelloState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool OthelloState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || NoValidActions();
}

std::vector<double> OthelloState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string OthelloState::InformationStateString(Player player) const {
  return HistoryString();
}

std::string OthelloState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void OthelloState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void OthelloState::UndoAction(Player player, Action move) {
  SpielFatalError("Undo not implemented for this game.");
}

std::unique_ptr<State> OthelloState::Clone() const {
  return std::unique_ptr<State>(new OthelloState(*this));
}

OthelloGame::OthelloGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace othello
}  // namespace open_spiel
