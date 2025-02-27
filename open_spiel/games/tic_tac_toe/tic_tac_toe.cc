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

#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe",
    /*long_name=*/"Tic Tac Toe",
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
    {{"rows", GameParameter(kDefaultRows)},
     {"columns", GameParameter(kDefaultCols)}}
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TicTacToeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

Component PlayerToComponent(Player player) {
  Component component;
  switch (player) {
    case 0:
      component.state_ = CellState::kCross;
      break;
    case 1:
      component.state_ = CellState::kNought;
      break;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      component.state_ = CellState::kEmpty;
      break;
  }

  return component;
}

int TileToState(const GridBoard::Tile &tile) {
  return static_cast<int>(tile.component_.state_);
}

Component::Component() : state_(CellState::kEmpty) {}

std::string Component::ToString() const {
  switch (state_) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

bool operator==(const Component& lhs, const Component& rhs) {
  return lhs.state_ == rhs.state_;
}

std::string GridBoard::Tile::ToString() const {
  return component_.ToString();
}

void GridBoard::Tile::Clear() {
  component_.state_ = CellState::kEmpty;
}

bool GridBoard::Tile::IsEmpty() const {
  return component_.state_ == CellState::kEmpty;
}

GridBoard::GridBoard(size_t num_rows, size_t num_cols)
    // All cells on the board start empty
    : board_(num_rows * num_cols), num_rows_(num_rows),
      num_cols_(num_cols) {}

const GridBoard::Tile& GridBoard::At(size_t index) const {
  return board_.at(index);
}

GridBoard::Tile& GridBoard::At(size_t index) {
  return const_cast<Tile &>(std::as_const(*this).At(index));
}

const GridBoard::Tile& GridBoard::At(size_t row, size_t col) const {
  return board_.at(row * Cols() + col);
}

GridBoard::Tile& GridBoard::At(size_t row, size_t col) {
  return const_cast<Tile &>(std::as_const(*this).At(row, col));
}

size_t GridBoard::Rows() const {
  return num_rows_;
}

size_t GridBoard::Cols() const {
  return num_cols_;
}

size_t GridBoard::Size() const {
  return board_.size();
}

std::string GridBoard::ToString() const {
  std::string str;
  for (int r = 0; r < Rows(); ++r) {
    for (int c = 0; c < Cols(); ++c) {
      absl::StrAppend(&str, board_.at(r * Cols() + c).ToString());
    }
    if (r < (Rows() - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool BoardHasLine(const GridBoard& board, const Player player) {
  const auto c = PlayerToComponent(player);

  // We assume that we have lines everywhere, and we will check if that stands
  // after checking the contents of the cells
  std::vector<bool> is_row_line(board.Rows());
  std::vector<bool> is_col_line(board.Cols());
  std::fill(is_row_line.begin(), is_row_line.end(), true);
  std::fill(is_col_line.begin(), is_col_line.end(), true);

  // Check if any row or column has a line
  for (size_t row = 0; row < is_row_line.size(); ++row) {
    for (size_t col = 0; col < is_col_line.size(); ++col) {
      // If the cell does not match the player marker, then the line is
      // no longer possible in this row/col
      is_row_line[row] =
        is_row_line[row] && (board.At(row, col).component_ == c);
      is_col_line[col] =
        is_col_line[col] && (board.At(row, col).component_ == c);
    }

    // By now we have processed all columns in this row, so end the search
    // prematurely if we have a line in this row
    if (is_row_line[row]) {
      return true;
    }
  }

  // If any of the columns has a line, we are done
  if (std::find(is_col_line.cbegin(), is_col_line.cend(), true) !=
      is_col_line.end()) {
    return true;
  }

  // If no rows nor columns contain a line, we have to check the two
  // possible diagonals
  std::array<bool, 2> is_diag_line;
  std::fill(is_diag_line.begin(), is_diag_line.end(), true);
  for (size_t row = 0; row < is_row_line.size(); ++row) {
    is_diag_line[0] &= board.At(row, row).component_ == c;
    is_diag_line[1] &=
      board.At(is_row_line.size() - row - 1, row).component_ == c;
  }

  return is_diag_line[0] || is_diag_line[1];
}

void TicTacToeState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_.At(move).IsEmpty(), true);
  board_.At(move).component_ = PlayerToComponent(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeState::LegalActions(const GridBoard &board) const {
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < board.Size(); ++cell) {
    if (board.At(cell).IsEmpty()) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::vector<Action> TicTacToeState::LegalActions() const {
  if (IsTerminal()) return {};
  return LegalActions(board_);
}

std::string TicTacToeState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}

bool TicTacToeState::HasLine(Player player) const {
  return BoardHasLine(board_, player);
}

bool TicTacToeState::IsFull() const { return num_moves_ == board_.Size(); }

TicTacToeState::TicTacToeState(std::shared_ptr<const Game> game, size_t rows,
                               size_t cols)
    : State(game), board_(rows, cols) {}

std::string TicTacToeState::ToString() const {
  return board_.ToString();
}

bool TicTacToeState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeState::Returns() const {
  if (HasLine(Player{0})) {
    return {1.0, -1.0};
  } else if (HasLine(Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string TicTacToeState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, board_.Size()}, true);
  for (int cell = 0; cell < board_.Size(); ++cell) {
    view[{TileToState(board_.At(cell)), cell}] = 1.0;
  }
}

void TicTacToeState::UndoAction(Player player, Action move) {
  board_.At(move).Clear();
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToeState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeState(*this));
}

std::string TicTacToeGame::ActionToString(Player player,
                                          Action action_id) const {
  return absl::StrCat(PlayerToComponent(player).ToString(), "(",
                      action_id / cols_, ",", action_id % cols_, ")");
}

TicTacToeGame::TicTacToeGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      cols_(ParameterValue<int>("columns")) {}

size_t TicTacToeGame::Rows() const {
  return rows_;
}

size_t TicTacToeGame::Cols() const {
  return cols_;
}


}  // namespace tic_tac_toe
}  // namespace open_spiel
