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

#include "open_spiel/games/cliff_walking.h"

#include <algorithm>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cliff_walking {
namespace {

enum CliffWalkingAction { RIGHT = 0, UP = 1, LEFT = 2, DOWN = 3 };

// Facts about the game.
const GameType kGameType{/*short_name=*/"cliff_walking",
                         /*long_name=*/"CliffWalking",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"height", GameParameter(kDefaultHeight)},
                          {"width", GameParameter(kDefaultWidth)},
                          {"horizon", GameParameter(kDefaultHorizon)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<Game>(new CliffWalkingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CliffWalkingState::CliffWalkingState(std::shared_ptr<const Game> game)
    : State(game) {
  const CliffWalkingGame& parent_game =
      static_cast<const CliffWalkingGame&>(*game);
  height_ = parent_game.Height();
  width_ = parent_game.Width();
  horizon_ = parent_game.MaxGameLength();
  player_row_ = parent_game.Height() - 1;
}

int CliffWalkingState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  return 0;
}

std::vector<Action> CliffWalkingState::LegalActions() const {
  if (IsTerminal()) return {};
  return {RIGHT, UP, LEFT, DOWN};
}

std::string CliffWalkingState::ActionToString(int player,
                                              Action action_id) const {
  SPIEL_CHECK_EQ(player, 0);
  switch (action_id) {
    case RIGHT:
      return "RIGHT";
    case UP:
      return "UP";
    case LEFT:
      return "LEFT";
    case DOWN:
      return "DOWN";
    default:
      SpielFatalError("Out of range action");
  }
}

std::string CliffWalkingState::ToString() const {
  std::string str;
  str.reserve(height_ * (width_ + 1));
  for (int r = 0; r < height_; ++r) {
    for (int c = 0; c < width_; ++c) {
      if (r == player_row_ && c == player_col_)
        str += 'P';
      else if (IsCliff(r, c))
        str += 'X';
      else if (IsGoal(r, c))
        str += 'G';
      else
        str += '.';
    }
    str += '\n';
  }
  return str;
}

bool CliffWalkingState::IsTerminal() const {
  return time_counter_ >= horizon_ || IsCliff(player_row_, player_col_) ||
         IsGoal(player_row_, player_col_);
}

std::vector<double> CliffWalkingState::Rewards() const {
  if (IsCliff(player_row_, player_col_)) return {-100.0};
  if (time_counter_ == 0) return {0.0};
  return {-1.0};
}

std::vector<double> CliffWalkingState::Returns() const {
  if (IsCliff(player_row_, player_col_)) return {-100.0 - time_counter_ + 1};
  return {time_counter_ * -1.0};
}

std::string CliffWalkingState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CliffWalkingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CliffWalkingState::ObservationTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), height_ * width_);
  std::fill(values.begin(), values.end(), 0.);
  values[player_row_ * width_ + player_col_] = 1.0;
}

void CliffWalkingState::InformationStateTensor(Player player,
                                               absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kNumActions * horizon_);
  for (int i = 0; i < history_.size(); i++) {
    values[i * kNumActions + history_[i].action] = 1;
  }
}

std::unique_ptr<State> CliffWalkingState::Clone() const {
  return std::unique_ptr<State>(new CliffWalkingState(*this));
}

void CliffWalkingState::DoApplyAction(Action move) {
  switch (move) {
    case RIGHT:
      ++player_col_;
      break;
    case UP:
      --player_row_;
      break;
    case LEFT:
      --player_col_;
      break;
    case DOWN:
      ++player_row_;
      break;
    default:
      SpielFatalError("Unexpected action");
  }
  player_row_ = std::min(std::max(player_row_, 0), height_ - 1);
  player_col_ = std::min(std::max(player_col_, 0), width_ - 1);
  ++time_counter_;
}

bool CliffWalkingState::IsCliff(int row, int col) const {
  return col > 0 && col < width_ - 1 && row == height_ - 1;
}

bool CliffWalkingState::IsGoal(int row, int col) const {
  return row == height_ - 1 && col == width_ - 1;
}

CliffWalkingGame::CliffWalkingGame(const GameParameters& params)
    : Game(kGameType, params),
      height_(ParameterValue<int>("height")),
      width_(ParameterValue<int>("width")),
      horizon_(ParameterValue<int>("horizon")) {
  SPIEL_CHECK_GE(height_, 2);
  SPIEL_CHECK_GE(width_, 3);
}

}  // namespace cliff_walking
}  // namespace open_spiel
