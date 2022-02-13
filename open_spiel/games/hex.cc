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

#include "open_spiel/games/hex.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace hex {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"hex",
                         /*long_name=*/"Hex",
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
                         {
                             {"board_size", GameParameter(kDefaultBoardSize)},
                             {"num_cols", GameParameter(kDefaultBoardSize)},
                             {"num_rows", GameParameter(kDefaultBoardSize)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HexGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

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

CellState HexState::PlayerAndActionToState(Player player, Action move) const {
  // This function returns the CellState resulting from the given move.
  // The cell state tells us:
  // - The colour of the stone.
  // - If the stone results in a winning connection.
  // - If the stone connects to one of the edges needed for that colour's
  //   winning connection.
  //
  // We know the colour from the argument player
  // For connectedness to the edges, we check if the move is in first/last
  // row/column, or if any of the neighbours are the same colour and connected.
  if (player == 0) {
    bool north_connected = false;
    bool south_connected = false;
    if (move < num_cols_) {  // First row
      north_connected = true;
    } else if (move >= (board_.size() - num_cols_)) {  // Last row
      south_connected = true;
    }
    for (int neighbour : AdjacentCells(move)) {
      if (board_[neighbour] == CellState::kBlackNorth) {
        north_connected = true;
      } else if (board_[neighbour] == CellState::kBlackSouth) {
        south_connected = true;
      }
    }
    if (north_connected && south_connected) {
      return CellState::kBlackWin;
    } else if (north_connected) {
      return CellState::kBlackNorth;
    } else if (south_connected) {
      return CellState::kBlackSouth;
    } else {
      return CellState::kBlack;
    }
  } else if (player == 1) {
    bool west_connected = false;
    bool east_connected = false;
    if (move % num_cols_ == 0) {  // First column
      west_connected = true;
    } else if (move % num_cols_ == num_cols_ - 1) {  // Last column
      east_connected = true;
    }
    for (int neighbour : AdjacentCells(move)) {
      if (board_[neighbour] == CellState::kWhiteWest) {
        west_connected = true;
      } else if (board_[neighbour] == CellState::kWhiteEast) {
        east_connected = true;
      }
    }
    if (west_connected && east_connected) {
      return CellState::kWhiteWin;
    } else if (west_connected) {
      return CellState::kWhiteWest;
    } else if (east_connected) {
      return CellState::kWhiteEast;
    } else {
      return CellState::kWhite;
    }
  } else {
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
    case CellState::kWhiteWin:
      return "O";
    case CellState::kWhiteWest:
      return "p";
    case CellState::kWhiteEast:
      return "q";
    case CellState::kBlack:
      return "x";
    case CellState::kBlackWin:
      return "X";
    case CellState::kBlackNorth:
      return "y";
    case CellState::kBlackSouth:
      return "z";
    default:
      SpielFatalError("Unknown state.");
      return "This will never return.";
  }
}

void HexState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  CellState move_cell_state = PlayerAndActionToState(CurrentPlayer(), move);
  board_[move] = move_cell_state;
  if (move_cell_state == CellState::kBlackWin) {
    result_black_perspective_ = 1;
  } else if (move_cell_state == CellState::kWhiteWin) {
    result_black_perspective_ = -1;
  } else if (move_cell_state != CellState::kBlack &&
             move_cell_state != CellState::kWhite) {
    // Move is connected to an edge but not winning.
    // Update edge-connected groups with a flood-fill, to maintain that all edge
    // connected nodes are known about.
    // We don't do flood fill if a player has won, so it's impossible for a cell
    // connected to an edge to be changed by the flood-fill.
    CellState cell_state_to_change =
        (current_player_ == 0 ? CellState::kBlack : CellState::kWhite);
    // We assume that move can safely be cast to int
    std::vector<int> flood_stack = {static_cast<int>(move)};
    int latest_cell;
    while (!flood_stack.empty()) {
      latest_cell = flood_stack.back();
      flood_stack.pop_back();
      for (int neighbour : AdjacentCells(latest_cell)) {
        if (board_[neighbour] == cell_state_to_change) {
          // We make the change before putting the cell on the queue to avoid
          // putting the same cell on the queue multiple times
          board_[neighbour] = move_cell_state;
          flood_stack.push_back(neighbour);
        }
      }
    }
  }
  current_player_ = 1 - current_player_;
}

std::vector<Action> HexState::LegalActions() const {
  // Can move in any empty cell.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  for (int cell = 0; cell < board_.size(); ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string HexState::ActionToString(Player player, Action action_id) const {
  // This does not comply with the Hex Text Protocol
  // TODO(author8): Make compliant with HTP
  return absl::StrCat(StateToString(PlayerAndActionToState(player, action_id)),
                      "(", action_id % num_cols_, ",", action_id / num_cols_,
                      ")");
}

std::vector<int> HexState::AdjacentCells(int cell) const {
  std::vector<int> neighbours = {};
  bool north_edge = (cell < num_cols_);
  bool south_edge = (cell >= (board_.size() - num_cols_));
  bool west_edge = (cell % num_cols_ == 0);
  bool east_edge = (cell % num_cols_ == num_cols_ - 1);
  if (!north_edge) { neighbours.push_back(cell - num_cols_); }
  if (!north_edge && !east_edge) { neighbours.push_back(cell - num_cols_ + 1); }
  if (!east_edge) { neighbours.push_back(cell + 1); }
  if (!south_edge) { neighbours.push_back(cell + num_cols_); }
  if (!south_edge && !west_edge) { neighbours.push_back(cell + num_cols_ - 1); }
  if (!west_edge) { neighbours.push_back(cell - 1); }
  return neighbours;
}

HexState::HexState(std::shared_ptr<const Game> game, int num_cols, int num_rows)
    : State(game), num_cols_(num_cols), num_rows_(num_rows) {
  // for all num_colss & num_rowss -> num_colss_ >= num_rowss_
  board_.resize(num_cols * num_rows, CellState::kEmpty);
}

std::string HexState::ToString() const {
  std::string str;
  // Each cell has the cell plus a space
  // nth line has n spaces, and 1 "\n", except last line has no "\n"
  str.reserve(num_cols_ * num_rows_ * 2 + num_rows_ * (num_rows_ + 1) / 2 - 1);
  int line_num = 0;
  for (int cell = 0; cell < board_.size(); ++cell) {
    // if it's the first cell in a new row
    if (cell && cell % num_cols_ == 0) {
      absl::StrAppend(&str, "\n");
      line_num++;
      absl::StrAppend(&str, std::string(line_num, ' '));
    }
    absl::StrAppend(&str, StateToString(board_[cell]));
    absl::StrAppend(&str, " ");
  }
  return str;
}

bool HexState::IsTerminal() const { return result_black_perspective_ != 0; }

std::vector<double> HexState::Returns() const {
  return {result_black_perspective_, -result_black_perspective_};
}

std::string HexState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HexState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void HexState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  // TODO(author8): Make an option to not expose connection info
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (int cell = 0; cell < board_.size(); ++cell) {
    view[{static_cast<int>(board_[cell]) - kMinValueCellState, cell}] = 1.0;
  }
}

std::unique_ptr<State> HexState::Clone() const {
  return std::unique_ptr<State>(new HexState(*this));
}

HexGame::HexGame(const GameParameters& params)
    : Game(kGameType, params),
      // Use board_size as the default value of num_cols and num_rows
      num_cols_(
          ParameterValue<int>("num_cols", ParameterValue<int>("board_size"))),
      num_rows_(
          ParameterValue<int>("num_rows", ParameterValue<int>("board_size"))) {}

}  // namespace hex
}  // namespace open_spiel
