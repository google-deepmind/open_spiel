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

#include "open_spiel/games/tic_tac_toe.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

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
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new TicTacToeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

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
      return "This will never return.";
  }
}

void TicTacToeState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  current_player_ = 1 - current_player_;
}

std::vector<Action> TicTacToeState::LegalActions() const {
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
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id % kNumCols, ",", action_id / kNumCols, ")");
}

bool TicTacToeState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  return (board_[0] == c && board_[1] == c && board_[2] == c) ||
         (board_[3] == c && board_[4] == c && board_[5] == c) ||
         (board_[6] == c && board_[7] == c && board_[8] == c) ||
         (board_[0] == c && board_[3] == c && board_[6] == c) ||
         (board_[1] == c && board_[4] == c && board_[7] == c) ||
         (board_[2] == c && board_[5] == c && board_[8] == c) ||
         (board_[0] == c && board_[4] == c && board_[8] == c) ||
         (board_[2] == c && board_[4] == c && board_[6] == c);
}

bool TicTacToeState::IsFull() const {
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) return false;
  }
  return true;
}

TicTacToeState::TicTacToeState(int num_distinct_actions)
    : State(num_distinct_actions, kNumPlayers) {
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

bool TicTacToeState::IsTerminal() const {
  return HasLine(0) || HasLine(1) || IsFull();
}

std::vector<double> TicTacToeState::Returns() const {
  if (HasLine(0)) {
    return {1.0, -1.0};
  } else if (HasLine(1)) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string TicTacToeState::InformationState(Player player) const {
  return HistoryString();
}

std::string TicTacToeState::Observation(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeState::ObservationAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(kNumCells * kCellStates);
  std::fill(values->begin(), values->end(), 0.);
  for (int cell = 0; cell < kNumCells; ++cell) {
    (*values)[kNumCells * static_cast<int>(board_[cell]) + cell] = 1.0;
  }
}

void TicTacToeState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  history_.pop_back();
}

std::unique_ptr<State> TicTacToeState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeState(*this));
}

TicTacToeGame::TicTacToeGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe
}  // namespace open_spiel
