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

#include "open_spiel/games/catch.h"

#include <algorithm>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace catch_ {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"catch",
                         /*long_name=*/"Catch",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"rows", GameParameter(kDefaultRows)},
                          {"columns", GameParameter(kDefaultColumns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CatchGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kPaddle:
      return "x";
    case CellState::kBall:
      return "o";
    default:
      SpielFatalError("Unknown state.");
      return "This will never return.";
  }
}

}  // namespace

CatchState::CatchState(std::shared_ptr<const Game> game) : State(game) {
  const CatchGame& parent_game = static_cast<const CatchGame&>(*game);
  num_rows_ = parent_game.NumRows();
  num_columns_ = parent_game.NumColumns();
}

int CatchState::CurrentPlayer() const {
  if (!initialized_) return kChancePlayerId;
  if (IsTerminal()) return kTerminalPlayerId;
  return 0;
}

std::vector<Action> CatchState::LegalActions() const {
  if (IsTerminal()) return {};
  if (initialized_) {
    return {0, 1, 2};  // Left, stay, right.
  }
  std::vector<Action> moves;
  moves.reserve(num_columns_);
  for (int i = 0; i < num_columns_; i++) moves.push_back(i);
  return moves;
}

ActionsAndProbs CatchState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(!initialized_);
  ActionsAndProbs action_and_probs;
  action_and_probs.reserve(num_columns_);
  for (int c = 0; c < num_columns_; c++) {
    action_and_probs.emplace_back(c, 1. / num_columns_);
  }
  return action_and_probs;
}

CellState CatchState::BoardAt(int row, int column) const {
  if (row == num_rows_ - 1 && column == paddle_col_)
    return CellState::kPaddle;
  else if (row == ball_row_ && column == ball_col_)
    return CellState::kBall;
  return CellState::kEmpty;
}

std::string CatchState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Initialized ball to ", action_id);
  SPIEL_CHECK_EQ(player, 0);
  switch (action_id) {
    case 0:
      return "LEFT";
    case 1:
      return "STAY";
    case 2:
      return "RIGHT";
    default:
      SpielFatalError("Out of range action");
  }
}

std::string CatchState::ToString() const {
  std::string str;
  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_columns_; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

bool CatchState::IsTerminal() const {
  return initialized_ && ball_row_ >= num_rows_ - 1;
}

std::vector<double> CatchState::Returns() const {
  if (!IsTerminal()) {
    return {0.0};
  } else if (ball_col_ == paddle_col_) {
    return {1.0};
  } else {
    return {-1.0};
  }
}

std::string CatchState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CatchState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {num_rows_, num_columns_}, true);
  if (initialized_) {
    view[{ball_row_, ball_col_}] = 1.0;
    view[{num_rows_ - 1, paddle_col_}] = 1.0;
  }
}

std::unique_ptr<State> CatchState::Clone() const {
  return std::unique_ptr<State>(new CatchState(*this));
}

void CatchState::DoApplyAction(Action move) {
  if (!initialized_) {
    initialized_ = true;
    ball_col_ = move;
    ball_row_ = 0;
    paddle_col_ = num_columns_ / 2;
  } else {
    ball_row_++;
    int direction = move - 1;
    paddle_col_ =
        std::min(std::max(paddle_col_ + direction, 0), num_columns_ - 1);
  }
}

CatchGame::CatchGame(const GameParameters& params)
    : Game(kGameType, params),
      num_rows_(ParameterValue<int>("rows")),
      num_columns_(ParameterValue<int>("columns")) {}

}  // namespace catch_
}  // namespace open_spiel
