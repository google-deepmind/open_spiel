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

#include "open_spiel/games/mfg/crowd_modelling_2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/strip.h"
#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crowd_modelling_2d {

std::vector<absl::string_view> ProcessStringParam(
    const std::string& string_param_str, int max_size) {
  // ProcessStringParam takes a parameter string and split it is a sequence of
  // substring. Example:
  // "" -> {}
  // "[0|0;0|1]" -> {"0|0", "0|1"}
  // "[0.5;0.5]" -> {"0.5", "0.5"}
  absl::string_view string_param = absl::StripAsciiWhitespace(string_param_str);
  SPIEL_CHECK_TRUE(absl::ConsumePrefix(&string_param, "["));
  SPIEL_CHECK_TRUE(absl::ConsumeSuffix(&string_param, "]"));

  std::vector<absl::string_view> split_string_list;
  if (!string_param.empty()) {
    split_string_list = absl::StrSplit(string_param, ';');
  }
  SPIEL_CHECK_GE(split_string_list.size(), 0);
  SPIEL_CHECK_LE(split_string_list.size(), max_size * max_size);
  return split_string_list;
}

namespace {
inline constexpr float kEpsilon = 1e-25;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"mfg_crowd_modelling_2d",
    /*long_name=*/"Mean Field Crowd Modelling 2D",
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
    {
        {"size", GameParameter(kDefaultSize)},
        {"horizon", GameParameter(kDefaultHorizon)},
        {"only_distribution_reward",
         GameParameter(kDefaultOnlyDistributionReward)},
        {"forbidden_states", GameParameter(kDefaultForbiddenStates)},
        {"initial_distribution", GameParameter(kDefaultInitialDistribution)},
        {"initial_distribution_value",
         GameParameter(kDefaultInitialDistributionValue)},
        {"positional_reward", GameParameter(kDefaultPositionalReward)},
        {"positional_reward_value",
         GameParameter(kDefaultPositionalRewardValue)},
        {"with_congestion", GameParameter(kDefaultWithCongestion)},
        {"noise_intensity", GameParameter(kDefaultNoiseIntensity)},
        {"crowd_aversion_coef", GameParameter(kDefaultCrowdAversionCoef)},
    },
    /*default_loadable*/ true,
    /*provides_factored_observation_string*/ false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrowdModelling2dGame(params));
}

std::string StateToString(int x, int y, int t, Player player_id,
                          bool is_chance_init) {
  if (is_chance_init) {
    return "initial";
  }
  if (player_id == 0) {
    return absl::Substitute("($0, $1, $2)", x, y, t);
  }
  if (player_id == kMeanFieldPlayerId) {
    return absl::Substitute("($0, $1, $2)_a", x, y, t);
  }
  if (player_id == kChancePlayerId) {
    return absl::Substitute("($0, $1, $2)_a_mu", x, y, t);
  }
  SpielFatalError(
      absl::Substitute("Unexpected state (player_id: $0, is_chance_init: $1)",
                       player_id, is_chance_init));
}

std::vector<std::pair<int, int>> StringListToPairs(
    std::vector<absl::string_view> strings) {
  // Transforms a list of strings and returns a list of pairs
  // {} -> {}
  // {"0|0", "0|1"} -> {(0, 0), (0, 1)}
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < strings.size(); ++i) {
    std::vector<absl::string_view> xy = absl::StrSplit(strings[i], '|');
    int xx;
    int yy;
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(xy[0], &xx));
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(xy[1], &yy));
    pairs.push_back({xx, yy});
  }
  return pairs;
}
std::vector<float> StringListToFloats(std::vector<absl::string_view> strings) {
  // Transforms a list of strings and returns a list of float
  // {} -> {}
  // {"0.5","0.5"} -> {0.5, 0.5}
  std::vector<float> floats;
  floats.reserve(strings.size());
  for (int i = 0; i < strings.size(); ++i) {
    float ff;
    SPIEL_CHECK_TRUE(absl::SimpleAtof(strings[i], &ff));
    floats.push_back(ff);
  }
  return floats;
}

int GetX(int i, int size) { return i % size; }

int GetY(int i, int size) { return i / size; }

int MergeXY(int xx, int yy, int size) {
  SPIEL_CHECK_GE(xx, 0);
  SPIEL_CHECK_LE(xx, size - 1);
  SPIEL_CHECK_GE(yy, 0);
  SPIEL_CHECK_LE(yy, size - 1);
  return yy + xx * size;
}

bool ComparisonPair(const std::pair<int, int>& a,
                    const std::pair<int, int>& b) {
  return a.first < b.first;
}

std::vector<int> StringListToInts(std::vector<absl::string_view> strings,
                                  int size) {
  // Transforms a list of strings and returns a list of pairs
  // {} -> {}
  // {"0|0", "0|1"} -> {(0, 0), (0, 1)}
  std::vector<int> ints;
  for (int i = 0; i < strings.size(); ++i) {
    std::vector<absl::string_view> xy = absl::StrSplit(strings[i], '|');
    int xx;
    int yy;
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(xy[0], &xx));
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(xy[1], &yy));
    ints.push_back(MergeXY(xx, yy, size));
  }
  return ints;
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CrowdModelling2dState::CrowdModelling2dState(
    std::shared_ptr<const Game> game, int size, int horizon,
    bool only_distribution_reward, const std::string& forbidden_states,
    const std::string& initial_distribution,
    const std::string& initial_distribution_value,
    const std::string& positional_reward,
    const std::string& positional_reward_value, bool with_congestion,
    double noise_intensity, double crowd_aversion_coef)
    : State(game),
      size_(size),
      horizon_(horizon),
      only_distribution_reward_(only_distribution_reward),
      with_congestion_(with_congestion),
      noise_intensity_(noise_intensity),
      crowd_aversion_coef_(crowd_aversion_coef),
      distribution_(size_ * size_, 1. / (size_ * size_)) {
  std::vector<absl::string_view> initial_distribution_list =
      ProcessStringParam(initial_distribution, size_);
  std::vector<absl::string_view> initial_distribution_value_list =
      ProcessStringParam(initial_distribution_value, size_);
  SPIEL_CHECK_EQ(initial_distribution_list.size(),
                 initial_distribution_value_list.size());

  auto initial_distribution_pair = StringListToPairs(initial_distribution_list);
  auto initial_distribution_value_f =
      StringListToFloats(initial_distribution_value_list);

  int initial_distribution_action_prob_size = initial_distribution_list.size();

  if (initial_distribution_action_prob_size == 0) {
    for (int i = 0; i < size_ * size_; ++i) {
      initial_distribution_action_prob_.push_back({i, 1. / (size_ * size_)});
    }
  } else {
    for (int i = 0; i < initial_distribution_action_prob_size; ++i) {
      int kk = MergeXY(initial_distribution_pair[i].first,
                       initial_distribution_pair[i].second, size_);
      initial_distribution_action_prob_.push_back(
          {kk, initial_distribution_value_f[i]});
    }
  }

  std::sort(initial_distribution_action_prob_.begin(),
            initial_distribution_action_prob_.end(), ComparisonPair);

  std::vector<absl::string_view> forbidden_states_list =
      ProcessStringParam(forbidden_states, size_);
  forbidden_states_xy_ = StringListToPairs(forbidden_states_list);
  for (const auto& forbidden_state_xy : forbidden_states_xy_) {
    SPIEL_CHECK_GE(forbidden_state_xy.first, 0);
    SPIEL_CHECK_LE(forbidden_state_xy.first, size_ - 1);
    SPIEL_CHECK_GE(forbidden_state_xy.second, 0);
    SPIEL_CHECK_LE(forbidden_state_xy.second, size_ - 1);
  }

  std::vector<absl::string_view> positional_reward_list =
      ProcessStringParam(positional_reward, size_);
  std::vector<absl::string_view> positional_reward_value_list =
      ProcessStringParam(positional_reward_value, size_);
  positional_reward_xy_ = StringListToPairs(positional_reward_list);
  positional_reward_value_ = StringListToFloats(positional_reward_value_list);
  // There should be a reward for each positional reward XY pair.
  SPIEL_CHECK_EQ(positional_reward_xy_.size(), positional_reward_value_.size());
  for (const auto& positional_reward_xy : positional_reward_xy_) {
    SPIEL_CHECK_GE(positional_reward_xy.first, 0);
    SPIEL_CHECK_LE(positional_reward_xy.first, size_ - 1);
    SPIEL_CHECK_GE(positional_reward_xy.second, 0);
    SPIEL_CHECK_LE(positional_reward_xy.second, size_ - 1);
  }

  if (positional_reward_xy_.empty()) {
    // Use the center point as the reward position.
    positional_reward_xy_.push_back({size_ / 2, size_ / 2});
    positional_reward_value_.push_back(1.0);
  }

  // Forbid to the initial distribution and the forbidden states to overlap.
  auto forbidden_states_int = StringListToInts(forbidden_states_list, size_);
  auto initial_distribution_int =
      StringListToInts(initial_distribution_list, size_);
  std::vector<int> intersection;
  std::sort(forbidden_states_int.begin(), forbidden_states_int.end());
  std::sort(initial_distribution_int.begin(), initial_distribution_int.end());

  std::set_intersection(
      forbidden_states_int.begin(), forbidden_states_int.end(),
      initial_distribution_int.begin(), initial_distribution_int.end(),
      back_inserter(intersection));
  SPIEL_CHECK_EQ(intersection.size(), 0);
}

CrowdModelling2dState::CrowdModelling2dState(
    std::shared_ptr<const Game> game, int size, int horizon,
    bool only_distribution_reward, const std::string& forbidden_states,
    const std::string& initial_distribution,
    const std::string& initial_distribution_value,
    const std::string& positional_reward,
    const std::string& positional_reward_value, Player current_player,
    bool is_chance_init, int x, int y, int t, int last_action,
    double return_value, const std::vector<double>& distribution,
    bool with_congestion, double noise_intensity, double crowd_aversion_coef)
    : CrowdModelling2dState(game, size, horizon, only_distribution_reward,
                            forbidden_states, initial_distribution,
                            initial_distribution_value, positional_reward,
                            positional_reward_value, with_congestion,
                            noise_intensity, crowd_aversion_coef) {
  current_player_ = current_player;
  is_chance_init_ = is_chance_init;
  x_ = x;
  y_ = y;
  t_ = t;
  last_action_ = last_action;
  return_value_ = return_value;
}

std::vector<Action> CrowdModelling2dState::LegalPlayerActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumActions);
  for (Action action = 0; action < kNumActions; ++action) {
    if (!IsForbidden(action)) {
      legal_actions.push_back(action);
    }
  }
  return legal_actions;
}

std::vector<Action> CrowdModelling2dState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsMeanFieldNode()) return {};
  SPIEL_CHECK_TRUE(IsPlayerNode());
  return LegalPlayerActions();
}

ActionsAndProbs CrowdModelling2dState::ChanceOutcomes() const {
  if (is_chance_init_) {
    return initial_distribution_action_prob_;
  }
  const std::vector<Action> legal_actions = LegalPlayerActions();
  ActionsAndProbs outcomes;
  if (legal_actions.empty()) {
    return outcomes;
  }
  // Neutral action will always be present in the legal actions.
  const double prob = noise_intensity_ / legal_actions.size();
  outcomes.reserve(legal_actions.size());
  for (const Action action : legal_actions) {
    if (action == kNeutralAction) {
      outcomes.emplace_back(action, 1.0 - noise_intensity_ + prob);
    } else {
      outcomes.emplace_back(action, prob);
    }
  }
  return outcomes;
}

bool CrowdModelling2dState::IsForbidden(Action action) const {
  int xx = (x_ + kActionToMoveX.at(action) + size_) % size_;
  int yy = (y_ + kActionToMoveY.at(action) + size_) % size_;
  return IsForbiddenPosition(xx, yy);
}

bool CrowdModelling2dState::IsForbiddenPosition(int x, int y) const {
  for (const auto& forbidden_xy : forbidden_states_xy_) {
    if (x == forbidden_xy.first && y == forbidden_xy.second) {
      return true;
    }
  }
  return false;
}

void CrowdModelling2dState::DoApplyAction(Action action) {
  SPIEL_CHECK_NE(current_player_, kMeanFieldPlayerId);
  return_value_ += Rewards()[0];
  int xx;
  int yy;
  // Compute the next state
  if (is_chance_init_) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, (size_ * size_));
    SPIEL_CHECK_EQ(current_player_, kChancePlayerId);
    xx = GetX(action, size_);
    yy = GetY(action, size_);
    is_chance_init_ = false;
    current_player_ = 0;
  } else if (current_player_ == kChancePlayerId) {
    xx = (x_ + kActionToMoveX.at(action) + size_) % size_;
    yy = (y_ + kActionToMoveY.at(action) + size_) % size_;
    ++t_;
    current_player_ = kMeanFieldPlayerId;
  } else {
    SPIEL_CHECK_EQ(current_player_, 0);
    xx = (x_ + kActionToMoveX.at(action) + size_) % size_;
    yy = (y_ + kActionToMoveY.at(action) + size_) % size_;
    last_action_ = action;
    current_player_ = kChancePlayerId;
  }
  // Assign the new (x,y) position if it isn't forbidden.
  if (!IsForbiddenPosition(xx, yy) || is_chance_init_) {
    x_ = xx;
    y_ = yy;
  }
}

std::string CrowdModelling2dState::ActionToString(Player player,
                                                  Action action) const {
  if (IsChanceNode() && is_chance_init_) {
    return absl::Substitute("init_state=$0", action);
  }
  return absl::Substitute("($0,$1)", kActionToMoveX.at(action),
                          kActionToMoveY.at(action));
}

std::vector<std::string> CrowdModelling2dState::DistributionSupport() {
  std::vector<std::string> support;
  support.reserve(size_ * size_);
  for (int x = 0; x < size_; ++x) {
    for (int y = 0; y < size_; ++y) {
      support.push_back(StateToString(x, y, t_, kMeanFieldPlayerId, false));
    }
  }
  return support;
}

void CrowdModelling2dState::UpdateDistribution(
    const std::vector<double>& distribution) {
  SPIEL_CHECK_EQ(current_player_, kMeanFieldPlayerId);
  SPIEL_CHECK_EQ(distribution.size(), size_ * size_);
  distribution_ = distribution;
  current_player_ = kDefaultPlayerId;
}

bool CrowdModelling2dState::IsTerminal() const { return t_ >= horizon_; }

std::vector<double> CrowdModelling2dState::Rewards() const {
  if (current_player_ != 0) {
    return {0.};
  }
  // Distribution-based reward
  double r_mu = -crowd_aversion_coef_ *
                std::log(distribution_[MergeXY(x_, y_, size_)] + kEpsilon);
  if (only_distribution_reward_) {
    return {r_mu};
  }
  // Positional reward
  double r_x = 1;
  double r_y = 1;
  for (int i = 0; i < positional_reward_xy_.size(); ++i) {
    double val_r = 2.0 * positional_reward_value_[i] / size_;
    r_x -= val_r * std::abs(x_ - positional_reward_xy_[i].first);
    r_y -= val_r * std::abs(y_ - positional_reward_xy_[i].second);
  }
  double r_a = -1.0 *
               (std::abs(kActionToMoveX.at(last_action_)) +
                std::abs(kActionToMoveY.at(last_action_))) /
               size_;
  if (with_congestion_) {
    // Congestion effect: higher penalty when moving in a high-density area
    r_a *= distribution_[MergeXY(x_, y_, size_)];
  }
  return {r_x + r_y + r_a + r_mu};
}

std::vector<double> CrowdModelling2dState::Returns() const {
  return {return_value_ + Rewards()[0]};
}

std::string CrowdModelling2dState::ToString() const {
  return StateToString(x_, y_, t_, current_player_, is_chance_init_);
}

std::string CrowdModelling2dState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CrowdModelling2dState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CrowdModelling2dState::ObservationTensor(Player player,
                                              absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), 2 * size_ + horizon_ + 1);
  SPIEL_CHECK_LT(x_, size_);
  SPIEL_CHECK_LT(y_, size_);
  SPIEL_CHECK_GE(t_, 0);
  // Allow t_ == horizon_.
  SPIEL_CHECK_LE(t_, horizon_);
  std::fill(values.begin(), values.end(), 0.);
  if (x_ >= 0 && y_ >= 0) {
    values[x_] = 1.;
    values[y_ + size_] = 1.;
  } else {
    // x_ and y_ equal -1 for the initial (blank) state, don't set any position
    // bit in that case.
    SPIEL_CHECK_EQ(x_, -1);
    SPIEL_CHECK_EQ(y_, -1);
  }
  values[2 * size_ + t_] = 1.;
}

std::unique_ptr<State> CrowdModelling2dState::Clone() const {
  return std::unique_ptr<State>(new CrowdModelling2dState(*this));
}

std::string CrowdModelling2dState::Serialize() const {
  std::string out =
      absl::StrCat(current_player_, ",", is_chance_init_, ",", x_, ",", y_, ",",
                   t_, ",", last_action_, ",", return_value_, "\n");
  absl::StrAppend(&out, absl::StrJoin(distribution_, ","));
  return out;
}

CrowdModelling2dGame::CrowdModelling2dGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size", kDefaultSize)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)),
      only_distribution_reward_(ParameterValue<bool>(
          "only_distribution_reward", kDefaultOnlyDistributionReward)),
      forbidden_states_(ParameterValue<std::string>("forbidden_states",
                                                    kDefaultForbiddenStates)),
      initial_distribution_(ParameterValue<std::string>(
          "initial_distribution", kDefaultInitialDistribution)),
      initial_distribution_value_(ParameterValue<std::string>(
          "initial_distribution_value", kDefaultInitialDistributionValue)),
      positional_reward_(ParameterValue<std::string>("positional_reward",
                                                     kDefaultPositionalReward)),
      positional_reward_value_(ParameterValue<std::string>(
          "positional_reward_value", kDefaultPositionalRewardValue)),
      with_congestion_(
          ParameterValue<bool>("with_congestion", kDefaultWithCongestion)),
      noise_intensity_(
          ParameterValue<double>("noise_intensity", kDefaultNoiseIntensity)),
      crowd_aversion_coef_(ParameterValue<double>("crowd_aversion_coef",
                                                  kDefaultCrowdAversionCoef)) {}

std::vector<int> CrowdModelling2dGame::ObservationTensorShape() const {
  // +1 to allow for t_ == horizon.
  return {2 * ParameterValue<int>("size") + ParameterValue<int>("horizon") + 1};
}

std::unique_ptr<State> CrowdModelling2dGame::DeserializeState(
    const std::string& str) const {
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  if (lines.size() != 2) {
    SpielFatalError(absl::StrCat("Expected 2 lines in serialized state, got: ",
                                 lines.size()));
  }
  Player current_player;
  int is_chance_init, x, y, t, last_action;
  double return_value;
  std::vector<double> distribution;

  std::vector<std::string> properties = absl::StrSplit(lines[0], ',');
  if (properties.size() != 7) {
    SpielFatalError(
        absl::StrCat("Expected 7 properties for serialized state, got: ",
                     properties.size()));
  }
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[0], &current_player));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[1], &is_chance_init));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[2], &x));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[3], &y));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[4], &t));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[5], &last_action));
  SPIEL_CHECK_TRUE(absl::SimpleAtod(properties[6], &return_value));
  std::vector<std::string> serialized_distrib = absl::StrSplit(lines[1], ',');
  distribution.reserve(serialized_distrib.size());
  for (std::string& v : serialized_distrib) {
    double parsed_weight;
    SPIEL_CHECK_TRUE(absl::SimpleAtod(v, &parsed_weight));
    distribution.push_back(parsed_weight);
  }
  return absl::make_unique<CrowdModelling2dState>(
      shared_from_this(), size_, horizon_, only_distribution_reward_,
      forbidden_states_, initial_distribution_, initial_distribution_value_,
      positional_reward_, positional_reward_value_, current_player,
      is_chance_init, x, y, t, last_action, return_value, distribution,
      with_congestion_, noise_intensity_, crowd_aversion_coef_);
}

}  // namespace crowd_modelling_2d
}  // namespace open_spiel
