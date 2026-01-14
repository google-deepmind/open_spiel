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
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TicTacToeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "x";
    case 1:
      return "o";
    default:
      return DefaultPlayerString(player);
  }
}

CellState StringToCellState(const std::string& s) {
  if (s == "x") return CellState::kCross;
  if (s == "o") return CellState::kNought;
  if (s == ".") return CellState::kEmpty;
  SpielFatalError(absl::StrCat("Invalid cell string: ", s));
}

std::string StateToString(CellState state) {
  switch (state) {
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

bool BoardHasLine(const std::array<CellState, kNumCells>& board,
                  const Player player) {
  CellState c = PlayerToState(player);
  return (board[0] == c && board[1] == c && board[2] == c) ||
         (board[3] == c && board[4] == c && board[5] == c) ||
         (board[6] == c && board[7] == c && board[8] == c) ||
         (board[0] == c && board[3] == c && board[6] == c) ||
         (board[1] == c && board[4] == c && board[7] == c) ||
         (board[2] == c && board[5] == c && board[8] == c) ||
         (board[0] == c && board[4] == c && board[8] == c) ||
         (board[2] == c && board[4] == c && board[6] == c);
}

std::vector<CellState> TicTacToeState::Board() const {
  std::vector<CellState> board(board_.begin(), board_.end());
  return board;
}


void TicTacToeState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string TicTacToeState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}

bool TicTacToeState::HasLine(Player player) const {
  return BoardHasLine(board_, player);
}

bool TicTacToeState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeState::TicTacToeState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeState::ToString() const {
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

std::unique_ptr<StateStruct> TicTacToeState::ToStruct() const {
  auto rv = std::make_unique<TicTacToeStateStruct>();
  std::vector<std::string> board;
  board.reserve(board_.size());
  for (const CellState& cell : board_) {
    board.push_back(StateToString(cell));
  }
  rv->current_player = PlayerToString(CurrentPlayer());
  rv->board = board;
  return rv;
}

std::unique_ptr<ObservationStruct> TicTacToeState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return std::make_unique<TicTacToeObservationStruct>(this->ToJson());
}

std::unique_ptr<ActionStruct> TicTacToeState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<TicTacToeActionStruct>();
  action_struct->row = action_id / kNumCols;
  action_struct->col = action_id % kNumCols;
  return action_struct;
}

Action TicTacToeState::StructToAction(
    const ActionStruct& action_struct) const {
  const auto* ttt_action_struct =
      dynamic_cast<const TicTacToeActionStruct*>(&action_struct);
  SPIEL_CHECK_TRUE(ttt_action_struct != nullptr);
  SPIEL_CHECK_GE(ttt_action_struct->row, 0);
  SPIEL_CHECK_LT(ttt_action_struct->row, kNumRows);
  SPIEL_CHECK_GE(ttt_action_struct->col, 0);
  SPIEL_CHECK_LT(ttt_action_struct->col, kNumCols);
  return ttt_action_struct->row * kNumCols + ttt_action_struct->col;
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
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void TicTacToeState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
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
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

TicTacToeState::TicTacToeState(const std::shared_ptr<const Game> game,
                               const nlohmann::json& json) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);

  TicTacToeStateStruct state_struct(json);
  if (state_struct.board.size() != kNumCells) {
    SpielFatalError(absl::StrFormat("Invalid board size: expected %d, got %d",
                                    kNumCells, state_struct.board.size()));
  }
  num_moves_ = 0;
  int num_x = 0;
  int num_o = 0;
  for (Action action = 0; action < state_struct.board.size(); ++action) {
    CellState cell_state = StringToCellState(state_struct.board[action]);
    if (cell_state != CellState::kEmpty) {
      board_[action] = cell_state;
      num_moves_++;
      if (cell_state == CellState::kCross) {
        num_x++;
      } else {
        num_o++;
      }
    }
  }
  if (num_x < num_o || num_x > num_o + 1) {
    SpielFatalError(absl::StrFormat(
        "Invalid board state: invalid number of pieces, got x = %d, o = %d",
        num_x, num_o));
  }
  current_player_ = (num_x == num_o ? 0 : 1);

  bool x_wins = HasLine(0);
  bool o_wins = HasLine(1);

  if (x_wins && o_wins) {
    SpielFatalError("Invalid board state: both players have a line.");
  }

  if (x_wins) {
    if (num_x != num_o + 1) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: x has a line, but number of pieces is "
          "inconsistent, got x = %d, o = %d",
          num_x, num_o));
    }
    outcome_ = 0;
  } else if (o_wins) {
    if (num_x != num_o) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: o has a line, but number of pieces is "
          "inconsistent, got x = %d, o = %d",
          num_x, num_o));
    }
    outcome_ = 1;
  } else {
    outcome_ = kInvalidPlayer;
  }

  if (state_struct.current_player != PlayerToString(CurrentPlayer())) {
    SpielFatalError(absl::StrCat("Invalid current player: expected ",
                                 PlayerToString(CurrentPlayer()),
                                 ", got ", state_struct.current_player));
  }

  starting_state_str_ = this->ToJson();
}

TicTacToeGame::TicTacToeGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe
}  // namespace open_spiel
