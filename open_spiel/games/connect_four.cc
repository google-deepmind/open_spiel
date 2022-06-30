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

#include "open_spiel/games/connect_four.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace connect_four {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"connect_four",
    /*long_name=*/"Connect Four",
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
  return std::shared_ptr<const Game>(new ConnectFourGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
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
}  // namespace

CellState& ConnectFourState::CellAt(int row, int col) {
  return board_[row * kCols + col];
}

CellState ConnectFourState::CellAt(int row, int col) const {
  return board_[row * kCols + col];
}

int ConnectFourState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void ConnectFourState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(CellAt(kRows - 1, move), CellState::kEmpty);
  int row = 0;
  while (CellAt(row, move) != CellState::kEmpty) ++row;
  CellAt(row, move) = PlayerToState(CurrentPlayer());

  if (HasLine(current_player_)) {
    outcome_ = static_cast<Outcome>(current_player_);
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }

  current_player_ = 1 - current_player_;
}

std::vector<Action> ConnectFourState::LegalActions() const {
  // Can move in any non-full column.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  for (int col = 0; col < kCols; ++col) {
    if (CellAt(kRows - 1, col) == CellState::kEmpty) moves.push_back(col);
  }
  return moves;
}

std::string ConnectFourState::ActionToString(Player player,
                                             Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), action_id);
}

bool ConnectFourState::HasLineFrom(Player player, int row, int col) const {
  return HasLineFromInDirection(player, row, col, 0, 1) ||
         HasLineFromInDirection(player, row, col, -1, -1) ||
         HasLineFromInDirection(player, row, col, -1, 0) ||
         HasLineFromInDirection(player, row, col, -1, 1);
}

bool ConnectFourState::HasLineFromInDirection(Player player, int row, int col,
                                              int drow, int dcol) const {
  if (row + 3 * drow >= kRows || col + 3 * dcol >= kCols ||
      row + 3 * drow < 0 || col + 3 * dcol < 0)
    return false;
  CellState c = PlayerToState(player);
  for (int i = 0; i < 4; ++i) {
    if (CellAt(row, col) != c) return false;
    row += drow;
    col += dcol;
  }
  return true;
}

bool ConnectFourState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  for (int col = 0; col < kCols; ++col) {
    for (int row = 0; row < kRows; ++row) {
      if (CellAt(row, col) == c && HasLineFrom(player, row, col)) return true;
    }
  }
  return false;
}

bool ConnectFourState::IsFull() const {
  for (int col = 0; col < kCols; ++col) {
    if (CellAt(kRows - 1, col) == CellState::kEmpty) return false;
  }
  return true;
}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string ConnectFourState::ToString() const {
  std::string str;
  for (int row = kRows - 1; row >= 0; --row) {
    for (int col = 0; col < kCols; ++col) {
      str.append(StateToString(CellAt(row, col)));
    }
    str.append("\n");
  }
  return str;
}
bool ConnectFourState::IsTerminal() const {
  return outcome_ != Outcome::kUnknown;
}

std::vector<double> ConnectFourState::Returns() const {
  if (outcome_ == Outcome::kPlayer1) return {1.0, -1.0};
  if (outcome_ == Outcome::kPlayer2) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string ConnectFourState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ConnectFourState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(CellState state, Player current) {
  switch (state) {
    case CellState::kNought:
      return current == 0 ? 0 : 1;
    case CellState::kCross:
      return current == 1 ? 0 : 1;
    case CellState::kEmpty:
      return 2;
    default:
      SpielFatalError("Unknown player type.");
  }
}

void ConnectFourState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, kNumCells}, true);

  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{PlayerRelative(board_[cell], player), cell}] = 1.0;
  }
}

std::unique_ptr<State> ConnectFourState::Clone() const {
  return std::unique_ptr<State>(new ConnectFourState(*this));
}

ConnectFourGame::ConnectFourGame(const GameParameters& params)
    : Game(kGameType, params) {}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game,
                                   const std::string& str)
    : State(game) {
  int xs = 0;
  int os = 0;
  int r = 5;
  int c = 0;
  for (const char ch : str) {
    switch (ch) {
      case '.':
        CellAt(r, c) = CellState::kEmpty;
        break;
      case 'x':
        ++xs;
        CellAt(r, c) = CellState::kCross;
        break;
      case 'o':
        ++os;
        CellAt(r, c) = CellState::kNought;
        break;
    }
    if (ch == '.' || ch == 'x' || ch == 'o') {
      ++c;
      if (c >= kCols) {
        r--;
        c = 0;
      }
    }
  }
  SPIEL_CHECK_TRUE(xs == os || xs == (os + 1));
  SPIEL_CHECK_TRUE(r == -1 && ("Problem parsing state (incorrect rows)."));
  SPIEL_CHECK_TRUE(c == 0 &&
                   ("Problem parsing state (column value should be 0)"));
  current_player_ = (xs == os) ? 0 : 1;

  if (HasLine(0)) {
    outcome_ = Outcome::kPlayer1;
  } else if (HasLine(1)) {
    outcome_ = Outcome::kPlayer2;
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }
}

}  // namespace connect_four
}  // namespace open_spiel
