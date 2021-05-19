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

#include "open_spiel/games/mfg/crowd_modelling.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crowd_modelling {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"mfg_crowd_modelling",
                         /*long_name=*/"Mean Field Crowd Modelling",
                         GameType::Dynamics::kMeanField,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/kNumPlayers,
                         /*min_num_players=*/kNumPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"size", GameParameter(kDefaultSize)},
                          {"horizon", GameParameter(kDefaultHorizon)}},
                         /*default_loadable*/true,
                         /*provides_factored_observation_string*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrowdModellingGame(params));
}

std::string StateToString(int x, int t, Player player_id, bool is_chance_init) {
  if (is_chance_init) {
    return "None";
  }
  if (player_id == 0) {
    return absl::Substitute("($0, $1)", x, t);
  }
  if (player_id == kMeanFieldPlayerId) {
    return absl::Substitute("($0, $1)_a", x, t);
  }
  if (player_id == kChancePlayerId) {
    return absl::Substitute("($0, $1)_a_mu", x, t);
  }
  SpielFatalError(absl::Substitute(
      "Unexpected state (player_id: $0, is_chance_init: $1)",
      player_id, is_chance_init));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CrowdModellingState::CrowdModellingState(std::shared_ptr<const Game> game,
                                         int size, int horizon)
    : State(game),
      size_(size),
      horizon_(horizon),
      distribution_(size_, 1. / size_) {}

std::vector<Action> CrowdModellingState::LegalActions() const {
  if (IsTerminal()) return {};
  if (current_player_ != 0) {
    if (current_player_ == kChancePlayerId) {
      if (is_chance_init_) {
        std::vector<Action> outcomes;
        outcomes.reserve(size_);
        for (int i = 0; i < size_; ++i) {
          outcomes.push_back(i);
        }
        return outcomes;
      }
      return {0, 1, 2};
    }
    return {};
  }
  return {0, 1, 2};
}

ActionsAndProbs CrowdModellingState::ChanceOutcomes() const {
  if (is_chance_init_) {
    ActionsAndProbs outcomes;
    for (int i = 0; i < size_; ++i) {
      outcomes.push_back({i, 1. / size_});
    }
    return outcomes;
  }
  return {{0, 1. / 3}, {1, 1. / 3}, {2, 1. / 3}};
}

void CrowdModellingState::DoApplyAction(Action action) {
  SPIEL_CHECK_NE(current_player_, kMeanFieldPlayerId);
  return_value_ += Rewards()[0];
  if (is_chance_init_) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, size_);
    SPIEL_CHECK_EQ(current_player_, kChancePlayerId);
    x_ = action;
    is_chance_init_ = false;
    current_player_ = 0;
  } else if (current_player_ == kChancePlayerId) {
    x_ = (x_ + kActionToMove.at(action) + size_) % size_;
    ++t_;
    current_player_ = 0;
  } else {
    SPIEL_CHECK_EQ(current_player_, 0);
    x_ = (x_ + kActionToMove.at(action) + size_) % size_;
    last_action_ = action;
    current_player_ = kMeanFieldPlayerId;
  }
}

std::string CrowdModellingState::ActionToString(Player player,
                                                Action action) const {
  return std::to_string(kActionToMove.at(action));
}

std::vector<std::string> CrowdModellingState::DistributionSupport() {
  std::vector<std::string> support;
  support.reserve(size_);
  for (int x = 0; x < size_; ++x) {
    support.push_back(StateToString(x, t_, 0, false));
  }
  return support;
}

void CrowdModellingState::UpdateDistribution(
    const std::vector<double>& distribution) {
  SPIEL_CHECK_EQ(current_player_, kMeanFieldPlayerId);
  SPIEL_CHECK_EQ(distribution.size(), size_);
  distribution_ = distribution;
  current_player_ = kChancePlayerId;
}

bool CrowdModellingState::IsTerminal() const { return t_ >= horizon_; }

std::vector<double> CrowdModellingState::Rewards() const {
  if (current_player_ != 0) {
    return {0.};
  }
  double r_x = 1 - 1.0 * std::abs(x_ - size_ / 2) / (size_ / 2);
  double r_a = -1.0 * std::abs(kActionToMove.at(last_action_)) / size_;
  double r_mu = -std::log(distribution_[x_]);
  return {r_x + r_a + r_mu};
}

std::vector<double> CrowdModellingState::Returns() const {
  return {return_value_ + Rewards()[0]};
}

std::string CrowdModellingState::ToString() const {
  return StateToString(x_, t_, current_player_, is_chance_init_);
}

std::string CrowdModellingState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CrowdModellingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CrowdModellingState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), size_ + horizon_);
  SPIEL_CHECK_GE(x_, 0);
  SPIEL_CHECK_LT(x_, size_);
  SPIEL_CHECK_GE(t_, 0);
  SPIEL_CHECK_LT(t_, horizon_);
  std::fill(values.begin(), values.end(), 0.);
  values[x_] = 1.;
  values[size_ + t_] = 1.;
}

std::unique_ptr<State> CrowdModellingState::Clone() const {
  return std::unique_ptr<State>(new CrowdModellingState(*this));
}

CrowdModellingGame::CrowdModellingGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::vector<int> CrowdModellingGame::ObservationTensorShape() const {
  return {ParameterValue<int>("size") + ParameterValue<int>("horizon")};
}

}  // namespace crowd_modelling
}  // namespace open_spiel
