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

#include "open_spiel/games/mnk/mnk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace mnk {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"mnk",
                         /*long_name=*/"m,n,k-game",
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
                         {{"m", GameParameter(kDefaultNumCols)},
                          {"n", GameParameter(kDefaultNumRows)},
                          {"k", GameParameter(kDefaultNumInARow)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MNKGame(params));
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

bool BoardHasLine(const std::vector<std::vector<CellState>>& board,
                  const Player player, int k, int r, int c, int dr, int dc) {
  CellState state = PlayerToState(player);
  int count = 0;

  for (int i = 0;
       i < k && 0 <= r && r < board.size() && 0 <= c && c < board[r].size();
       ++i, r += dr, c += dc)
    count += board[r][c] == state;

  return count == k;
}

bool BoardHasLine(const std::vector<std::vector<CellState>>& board,
                  const Player player, int k) {
  for (int r = 0; r < board.size(); ++r)
    for (int c = 0; c < board[r].size(); ++c)
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc)
          if (dr || dc)
            if (BoardHasLine(board, player, k, r, c, dr, dc)) return true;

  return false;
}

void MNKState::DoApplyAction(Action move) {
  auto [row, column] = ActionToCoordinates(move);
  SPIEL_CHECK_EQ(board_[row][column], CellState::kEmpty);
  board_[row][column] = PlayerToState(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::pair<int, int> MNKState::ActionToCoordinates(Action move) const {
  return {move / NumCols(), move % NumCols()};
}

int MNKState::CoordinatesToAction(int row, int column) const {
  return row * NumCols() + column;
}

int MNKState::NumRows() const {
  return std::static_pointer_cast<const MNKGame>(game_)->NumRows();
}

int MNKState::NumCols() const {
  return std::static_pointer_cast<const MNKGame>(game_)->NumCols();
}

int MNKState::NumCells() const {
  return std::static_pointer_cast<const MNKGame>(game_)->NumCells();
}

int MNKState::NumInARow() const {
  return std::static_pointer_cast<const MNKGame>(game_)->NumInARow();
}

std::vector<Action> MNKState::LegalActions() const {
  if (IsTerminal()) return {};

  // Can move in any empty cell.
  std::vector<Action> moves;

  for (int r = 0; r < board_.size(); ++r)
    for (int c = 0; c < board_[r].size(); ++c)
      if (board_[r][c] == CellState::kEmpty)
        moves.push_back(CoordinatesToAction(r, c));

  return moves;
}

std::string MNKState::ActionToString(Player player, Action action_id) const {
  return game_->ActionToString(player, action_id);
}

bool MNKState::HasLine(Player player) const {
  return BoardHasLine(board_, player, NumInARow());
}

bool MNKState::IsFull() const { return num_moves_ == NumCells(); }

MNKState::MNKState(std::shared_ptr<const Game> game) : State(game) {
  board_.resize(NumRows());

  for (int r = 0; r < board_.size(); ++r)
    board_[r].resize(NumCols(), CellState::kEmpty);
}

std::string MNKState::ToString() const {
  std::string str;
  for (int r = 0; r < NumRows(); ++r) {
    for (int c = 0; c < NumCols(); ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (NumRows() - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool MNKState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> MNKState::Returns() const {
  if (HasLine(Player{0})) {
    return {1.0, -1.0};
  } else if (HasLine(Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string MNKState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string MNKState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void MNKState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  for (int r = 0; r < NumRows(); ++r) {
    for (int c = 0; c < NumCols(); ++c) {
      int i = static_cast<int>(board_[r][c]);
      int j = CoordinatesToAction(r, c);
      values[i * NumCells() + j] = 1.0;
    }
  }
}

void MNKState::UndoAction(Player player, Action move) {
  auto [r, c] = ActionToCoordinates(move);
  board_[r][c] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> MNKState::Clone() const {
  return std::unique_ptr<State>(new MNKState(*this));
}

std::string MNKGame::ActionToString(Player player, Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / NumCols(), ",", action_id % NumCols(), ")");
}

MNKGame::MNKGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace mnk
}  // namespace open_spiel
