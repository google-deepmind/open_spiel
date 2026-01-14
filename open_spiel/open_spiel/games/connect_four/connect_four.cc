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

#include "open_spiel/games/connect_four/connect_four.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
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
    /*parameter_specification=*/
    {{"egocentric_obs_tensor", GameParameter(kDefaultEgocentricObsTensor)},
     {"rows", GameParameter(kDefaultNumRows)},
     {"columns", GameParameter(kDefaultNumCols)},
     {"x_in_row", GameParameter(kDefaultXInRow)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ConnectFourGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

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

Player StateToPlayer(const CellState& state) {
  switch (state) {
    case CellState::kCross:
      return 0;
    case CellState::kNought:
      return 1;
    case CellState::kEmpty:
      return 2;
    default:
      SpielFatalError("Invalid cell state in StateToPlayer");
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
  return board_[row * static_cast<const ConnectFourGame&>(*game_).cols() + col];
}

CellState ConnectFourState::CellAt(int row, int col) const {
  return board_[row * static_cast<const ConnectFourGame&>(*game_).cols() + col];
}

int ConnectFourState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void ConnectFourState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(
      CellAt(static_cast<const ConnectFourGame&>(*game_).rows() - 1, move),
      CellState::kEmpty);
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
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    if (CellAt(game.rows() - 1, col) == CellState::kEmpty) moves.push_back(col);
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
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  const int x_in_row = game.x_in_row();
  if (row + (x_in_row - 1) * drow >= game.rows() ||
      col + (x_in_row - 1) * dcol >= game.cols() ||
      row + (x_in_row - 1) * drow < 0 || col + (x_in_row - 1) * dcol < 0)
    return false;
  CellState c = PlayerToState(player);
  for (int i = 0; i < x_in_row; ++i) {
    if (CellAt(row, col) != c) return false;
    row += drow;
    col += dcol;
  }
  return true;
}

bool ConnectFourState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    for (int row = 0; row < game.rows(); ++row) {
      if (CellAt(row, col) == c && HasLineFrom(player, row, col)) return true;
    }
  }
  return false;
}

bool ConnectFourState::IsFull() const {
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    if (CellAt(game.rows() - 1, col) == CellState::kEmpty) return false;
  }
  return true;
}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game)
    : State(game) {
  const auto& parent_game = static_cast<const ConnectFourGame&>(*game);
  board_.assign(parent_game.rows() * parent_game.cols(), CellState::kEmpty);
}

std::string ConnectFourState::ToString() const {
  std::string str;
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int row = game.rows() - 1; row >= 0; --row) {
    for (int col = 0; col < game.cols(); ++col) {
      str.append(StateToString(CellAt(row, col)));
    }
    str.append("\n");
  }
  return str;
}

std::unique_ptr<StateStruct> ConnectFourState::ToStruct() const {
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  std::vector<std::vector<std::string>> board(
      game.rows(), std::vector<std::string>(game.cols()));
  for (int r = 0; r < game.rows(); ++r) {
    for (int c = 0; c < game.cols(); ++c) {
      board[r][c] = StateToString(CellAt(r, c));
    }
  }
  ConnectFourStateStruct rv;
  rv.board = board;
  rv.current_player = PlayerToString(CurrentPlayer());
  rv.is_terminal = IsTerminal();
  rv.winner = "";
  if (rv.is_terminal) {
    switch (outcome_) {
      case Outcome::kPlayer1:
        rv.winner = "x";
        break;
      case Outcome::kPlayer2:
        rv.winner = "o";
        break;
      case Outcome::kDraw:
        rv.winner = "draw";
        break;
      default:
        SpielFatalError("Game is terminal but outcome is unknown.");
    }
  }
  return std::make_unique<ConnectFourStateStruct>(rv);
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

  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  TensorView<3> view(values, {kCellStates, game.rows(), game.cols()}, true);
  for (int r = 0; r < game.rows(); ++r) {
    for (int c = 0; c < game.cols(); ++c) {
      if (game.egocentric_obs_tensor()) {
        view[{PlayerRelative(CellAt(r, c), player), r, c}] = 1.0;
      } else {
        view[{StateToPlayer(CellAt(r, c)), r, c}] = 1.0;
      }
    }
  }
}

std::unique_ptr<State> ConnectFourState::Clone() const {
  return std::unique_ptr<State>(new ConnectFourState(*this));
}

ConnectFourGame::ConnectFourGame(const GameParameters& params)
    : Game(kGameType, params),
      egocentric_obs_tensor_(
          ParameterValue<bool>("egocentric_obs_tensor")),
      rows_(ParameterValue<int>("rows")),
      cols_(ParameterValue<int>("columns")),
      x_in_row_(ParameterValue<int>("x_in_row")) {}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game,
                                   const std::string& str)
    : State(game) {
  const auto& parent_game = static_cast<const ConnectFourGame&>(*game);
  board_.resize(parent_game.rows() * parent_game.cols());
  int xs = 0;
  int os = 0;
  int r = parent_game.rows() - 1;
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
      if (c >= parent_game.cols()) {
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
