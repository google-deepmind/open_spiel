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

#include "open_spiel/games/catch.h"

#include <algorithm>
#include <utility>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace catch_ {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"catch",
    /*long_name=*/"Catch",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {{"rows", {GameParameter::Type::kInt, false}},
     {"columns", {GameParameter::Type::kInt, false}}}};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new CatchGame(params));
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

CatchState::CatchState(const CatchGame& game)
    : State(/*num_distinct_actions=*/kNumActions,
            /*num_players=*/kNumPlayers),
      game_(game) {}

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
  moves.reserve(game_.NumColumns());
  for (int i = 0; i < game_.NumColumns(); i++) moves.push_back(i);
  return moves;
}

ActionsAndProbs CatchState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(!initialized_);
  ActionsAndProbs action_and_probs;
  action_and_probs.reserve(game_.NumColumns());
  for (int c = 0; c < game_.NumColumns(); c++) {
    action_and_probs.emplace_back(c, 1. / game_.NumColumns());
  }
  return action_and_probs;
}

CellState CatchState::BoardAt(int row, int column) const {
  if (row == game_.NumRows() - 1 && column == paddle_col_)
    return CellState::kPaddle;
  else if (row == ball_row_ && column == ball_col_)
    return CellState::kBall;
  return CellState::kEmpty;
}

std::string CatchState::ActionToString(int player, Action action_id) const {
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
  for (int r = 0; r < game_.NumRows(); ++r) {
    for (int c = 0; c < game_.NumColumns(); ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

bool CatchState::IsTerminal() const {
  return initialized_ && ball_row_ >= game_.NumRows() - 1;
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

std::string CatchState::InformationState(int player) const {
  SPIEL_CHECK_EQ(player, 0);
  return HistoryString();
}

std::string CatchState::Observation(int player) const {
  SPIEL_CHECK_EQ(player, 0);
  return ToString();
}

void CatchState::ObservationAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_EQ(player, 0);

  values->resize(game_.NumRows() * game_.NumColumns());
  std::fill(values->begin(), values->end(), 0.);
  if (initialized_) {
    (*values)[ball_row_ * game_.NumColumns() + ball_col_] = 1.0;
    (*values)[(game_.NumRows() - 1) * game_.NumColumns() + paddle_col_] = 1.0;
  }
}

void CatchState::InformationStateAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_EQ(player, 0);

  values->resize(game_.NumColumns() + kNumActions * game_.NumRows());
  std::fill(values->begin(), values->end(), 0.);
  if (initialized_) {
    (*values)[ball_col_] = 1;
    int offset = history_.size() - ball_row_ - 1;
    for (int i = 0; i < ball_row_; i++) {
      (*values)[game_.NumColumns() + i * kNumActions + history_[offset + i]] =
          1;
    }
  }
}

void CatchState::UndoAction(int player, Action move) {
  if (player == kChancePlayerId) {
    initialized_ = false;
    return;
  }
  int direction = move - 1;
  paddle_col_ =
      std::min(std::max(paddle_col_ - direction, 0), game_.NumColumns() - 1);
  --ball_row_;
  history_.pop_back();
}

std::unique_ptr<State> CatchState::Clone() const {
  return std::unique_ptr<State>(new CatchState(*this));
}

void CatchState::DoApplyAction(Action move) {
  if (!initialized_) {
    initialized_ = true;
    ball_col_ = move;
    ball_row_ = 0;
    paddle_col_ = game_.NumColumns() / 2;
  } else {
    ball_row_++;
    int direction = move - 1;
    paddle_col_ =
        std::min(std::max(paddle_col_ + direction, 0), game_.NumColumns() - 1);
  }
}

CatchGame::CatchGame(const GameParameters& params)
    : Game(kGameType, params),
      num_rows_(ParameterValue<int>("rows", kDefaultRows)),
      num_columns_(ParameterValue<int>("columns", kDefaultColumns)) {}

}  // namespace catch_
}  // namespace open_spiel
