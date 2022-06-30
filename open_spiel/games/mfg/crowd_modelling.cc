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

#include "open_spiel/games/mfg/crowd_modelling.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crowd_modelling {
namespace {
inline constexpr float kEpsilon = 1e-25;

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
    return "initial";
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

CrowdModellingState::CrowdModellingState(
    std::shared_ptr<const Game> game, int size, int horizon,
    Player current_player, bool is_chance_init, int x, int t, int last_action,
    double return_value, const std::vector<double>& distribution)
    : State(game),
      size_(size),
      horizon_(horizon),
      current_player_(current_player),
      is_chance_init_(is_chance_init),
      x_(x),
      t_(t),
      last_action_(last_action),
      return_value_(return_value),
      distribution_(distribution) {}

std::vector<Action> CrowdModellingState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsMeanFieldNode()) return {};
  SPIEL_CHECK_TRUE(IsPlayerNode());
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
    current_player_ = kMeanFieldPlayerId;
  } else {
    SPIEL_CHECK_EQ(current_player_, 0);
    x_ = (x_ + kActionToMove.at(action) + size_) % size_;
    last_action_ = action;
    current_player_ = kChancePlayerId;
  }
}

std::string CrowdModellingState::ActionToString(Player player,
                                                Action action) const {
  if (IsChanceNode() && is_chance_init_) {
    return absl::Substitute("init_state=$0", action);
  }
  return std::to_string(kActionToMove.at(action));
}

std::vector<std::string> CrowdModellingState::DistributionSupport() {
  std::vector<std::string> support;
  support.reserve(size_);
  for (int x = 0; x < size_; ++x) {
    support.push_back(StateToString(x, t_, kMeanFieldPlayerId, false));
  }
  return support;
}

void CrowdModellingState::UpdateDistribution(
    const std::vector<double>& distribution) {
  SPIEL_CHECK_EQ(current_player_, kMeanFieldPlayerId);
  SPIEL_CHECK_EQ(distribution.size(), size_);
  distribution_ = distribution;
  current_player_ = kDefaultPlayerId;
}

bool CrowdModellingState::IsTerminal() const { return t_ >= horizon_; }

std::vector<double> CrowdModellingState::Rewards() const {
  if (current_player_ != 0) {
    return {0.};
  }
  double r_x = 1 - 1.0 * std::abs(x_ - size_ / 2) / (size_ / 2);
  double r_a = -1.0 * std::abs(kActionToMove.at(last_action_)) / size_;
  double r_mu = -std::log(distribution_[x_]+kEpsilon);
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
  SPIEL_CHECK_EQ(values.size(), size_ + horizon_ + 1);
  SPIEL_CHECK_LT(x_, size_);
  SPIEL_CHECK_GE(t_, 0);
  // Allow t_ == horizon_.
  SPIEL_CHECK_LE(t_, horizon_);
  std::fill(values.begin(), values.end(), 0.);
  if (x_ >= 0) {
    values[x_] = 1.;
  }
  // x_ equals -1 for the initial (blank) state, don't set any
  // position bit in that case.
  values[size_ + t_] = 1.;
}

std::unique_ptr<State> CrowdModellingState::Clone() const {
  return std::unique_ptr<State>(new CrowdModellingState(*this));
}

std::string CrowdModellingState::Serialize() const {
  std::string out =
      absl::StrCat(current_player_, ",", is_chance_init_, ",", x_, ",", t_, ",",
                   last_action_, ",", return_value_, "\n");
  absl::StrAppend(&out, absl::StrJoin(distribution_, ","));
  return out;
}

CrowdModellingGame::CrowdModellingGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size", kDefaultSize)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)) {}

std::vector<int> CrowdModellingGame::ObservationTensorShape() const {
  // +1 to allow for t_ == horizon.
  return {size_ + horizon_ + 1};
}

std::unique_ptr<State> CrowdModellingGame::DeserializeState(
    const std::string& str) const {
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  if (lines.size() != 2) {
    SpielFatalError(absl::StrCat("Expected 2 lines in serialized state, got: ",
                                 lines.size()));
  }
  Player current_player;
  int is_chance_init;
  int x;
  int t;
  int last_action;
  double return_value;
  std::vector<std::string> properties = absl::StrSplit(lines[0], ',');
  if (properties.size() != 6) {
    SpielFatalError(
        absl::StrCat("Expected 6 properties for serialized state, got: ",
                     properties.size()));
  }
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[0], &current_player));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[1], &is_chance_init));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[2], &x));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[3], &t));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[4], &last_action));
  SPIEL_CHECK_TRUE(absl::SimpleAtod(properties[5], &return_value));
  std::vector<std::string> serialized_distrib = absl::StrSplit(lines[1], ',');
  std::vector<double> distribution;
  distribution.reserve(serialized_distrib.size());
  for (std::string& v : serialized_distrib) {
    double parsed_weight;
    SPIEL_CHECK_TRUE(absl::SimpleAtod(v, &parsed_weight));
    distribution.push_back(parsed_weight);
  }
  return absl::make_unique<CrowdModellingState>(
      shared_from_this(), size_, horizon_, current_player, is_chance_init, x, t,
      last_action, return_value, distribution);
}

}  // namespace crowd_modelling
}  // namespace open_spiel
