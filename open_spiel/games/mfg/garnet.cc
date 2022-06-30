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

#include "open_spiel/games/mfg/garnet.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace garnet {
namespace {
inline constexpr float kEpsilon = 1e-25;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"mfg_garnet",
    /*long_name=*/"Mean Field Garnet",
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
     {"horizon", GameParameter(kDefaultHorizon)},
     {"seed", GameParameter(kDefaultSeed)},
     {"num_action", GameParameter(kDefaultNumActions)},
     {"num_chance_action", GameParameter(kDefaultNumChanceActions)},
     {"sparsity_factor", GameParameter(kDefaultSparsityFactor)},
     {"eta", GameParameter(kDefaultEta)}},
    /*default_loadable*/ true,
    /*provides_factored_observation_string*/ false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GarnetGame(params));
}

std::string StateToString(int x, int t, Action last_action, Player player_id,
                          bool is_chance_init) {
  if (is_chance_init) {
    return "initial";
  } else if (player_id == 0) {
    return absl::Substitute("($0, $1)", x, t);
  } else if (player_id == kMeanFieldPlayerId) {
    return absl::Substitute("($0, $1)_a", x, t);
  } else if (player_id == kChancePlayerId) {
    return absl::Substitute("($0, $1, $2)_a_mu", x, t, last_action);
  } else {
    SpielFatalError(
        absl::Substitute("Unexpected state (player_id: $0, is_chance_init: $1)",
                         player_id, is_chance_init));
  }
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

GarnetState::GarnetState(std::shared_ptr<const Game> game, int size,
                         int horizon, int seed, int num_action,
                         int num_chance_action, double sparsity_factor,
                         double eta)
    : State(game),
      size_(size),
      horizon_(horizon),
      seed_(seed),
      num_action_(num_action),
      num_chance_action_(num_chance_action),
      sparsity_factor_(sparsity_factor),
      eta_(eta),
      distribution_(size_, 1. / size_) {
  std::mt19937 rng(seed_);
  double normalization;
  double proba;
  double cdf_proba;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < num_action; ++j) {
      double r_sparse = absl::Uniform<double>(rng, 0.0, 1.0);
      if (r_sparse < sparsity_factor_) {
        garnet_reward_.push_back(absl::Uniform<double>(rng, 0.0, 1.0));
      } else {
        garnet_reward_.push_back(0.0);
      }

      normalization = 0;
      std::vector<double> cdf;
      cdf.push_back(0.0);
      cdf.push_back(1.0);
      for (int kk = 0; kk < num_chance_action - 1; ++kk) {
        cdf_proba = absl::Uniform<double>(rng, 0.0, 1.0);
        cdf.push_back(cdf_proba);
      }
      std::sort(cdf.begin(), cdf.end());
      for (int k = 0; k < num_chance_action; ++k) {
        proba = cdf[k+1]-cdf[k];
        normalization += proba;
        garnet_transition_proba_unnormalized_.push_back(proba);
        garnet_transition_.push_back(absl::Uniform<int>(rng, 0, size));
      }
      garnet_transition_proba_normalization_.push_back(normalization);
    }
  }
}

GarnetState::GarnetState(std::shared_ptr<const Game> game, int size,
                         int horizon, int seed, int num_action,
                         int num_chance_action, double sparsity_factor,
                         double eta, Player current_player, bool is_chance_init,
                         int x, int t, int last_action, double return_value,
                         const std::vector<double>& distribution)
    : State(game),
      size_(size),
      horizon_(horizon),
      seed_(seed),
      num_action_(num_action),
      num_chance_action_(num_chance_action),
      sparsity_factor_(sparsity_factor),
      eta_(eta),
      current_player_(current_player),
      is_chance_init_(is_chance_init),
      x_(x),
      t_(t),
      last_action_(last_action),
      return_value_(return_value),
      distribution_(distribution) {
  std::mt19937 rng(seed_);
  double normalization;
  double proba;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < num_action; ++j) {
      double r_sparse = absl::Uniform<double>(rng, 0.0, 1.0);
      if (r_sparse < sparsity_factor_) {
        garnet_reward_.push_back(absl::Uniform<double>(rng, 0.0, 1.0));
      } else {
        garnet_reward_.push_back(0.0);
      }
      normalization = 0;
      for (int k = 0; k < num_chance_action; ++k) {
        proba = absl::Uniform<double>(rng, 0.0, 1.0);
        normalization += proba;
        garnet_transition_proba_unnormalized_.push_back(proba);
        garnet_transition_.push_back(absl::Uniform<int>(rng, 0, size));
      }
      garnet_transition_proba_normalization_.push_back(normalization);
    }
  }
}

double GarnetState::GetTransitionProba(int x, int action,
                                       int chance_action) const {
  return (garnet_transition_proba_unnormalized_[num_chance_action_ *
                                                    (x + size_ * action) +
                                                chance_action] /
          garnet_transition_proba_normalization_[x + size_ * action]);
}

int GarnetState::GetTransition(int x, int action, int chance_action) const {
  return garnet_transition_[num_chance_action_ * (x + size_ * action) +
                            chance_action];
}

double GarnetState::GetReward(int x, int action) const {
  return garnet_reward_[x + size_ * action];
}

std::vector<Action> GarnetState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsMeanFieldNode()) return {};
  SPIEL_CHECK_TRUE(IsPlayerNode());
  std::vector<Action> outcomes;
  outcomes.reserve(num_action_);
  for (int i = 0; i < num_action_; ++i) {
    outcomes.push_back(i);
  }
  return outcomes;
}

ActionsAndProbs GarnetState::ChanceOutcomes() const {
  if (is_chance_init_) {
    ActionsAndProbs outcomes;
    for (int i = 0; i < size_; ++i) {
      outcomes.push_back({i, 1. / size_});
    }
    return outcomes;
  }
  ActionsAndProbs outcomes;
  double proba;
  for (int i = 0; i < num_chance_action_; ++i) {
    proba = GetTransitionProba(x_, last_action_, i);
    outcomes.push_back({i, proba});
  }
  return outcomes;
}

void GarnetState::DoApplyAction(Action action) {
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
    x_ = GetTransition(x_, last_action_, action);
    ++t_;
    current_player_ = kMeanFieldPlayerId;
  } else {
    SPIEL_CHECK_EQ(current_player_, 0);
    last_action_ = action;
    current_player_ = kChancePlayerId;
  }
}

std::string GarnetState::ActionToString(Player player, Action action) const {
  if (IsChanceNode() && is_chance_init_) {
    return absl::Substitute("init_state=$0", action);
  }
  return std::to_string(action);
}

std::vector<std::string> GarnetState::DistributionSupport() {
  std::vector<std::string> support;
  support.reserve(size_);
  for (int x = 0; x < size_; ++x) {
    support.push_back(
        StateToString(x, t_, last_action_, kMeanFieldPlayerId, false));
  }
  return support;
}

void GarnetState::UpdateDistribution(const std::vector<double>& distribution) {
  SPIEL_CHECK_EQ(current_player_, kMeanFieldPlayerId);
  SPIEL_CHECK_EQ(distribution.size(), size_);
  distribution_ = distribution;
  current_player_ = kDefaultPlayerId;
}

bool GarnetState::IsTerminal() const { return t_ >= horizon_; }

std::vector<double> GarnetState::Rewards() const {
  if (current_player_ != 0) {
    return {0.};
  }
  double r_x = GetReward(x_, last_action_);
  double r_mu = -std::log(distribution_[x_] + kEpsilon);
  return {r_x + eta_ * r_mu};
}

std::vector<double> GarnetState::Returns() const {
  return {return_value_ + Rewards()[0]};
}

std::string GarnetState::ToString() const {
  return StateToString(x_, t_, last_action_, current_player_, is_chance_init_);
}

std::string GarnetState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return StateToString(x_, t_, last_action_, current_player_, is_chance_init_);
}

std::string GarnetState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void GarnetState::ObservationTensor(Player player,
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

std::unique_ptr<State> GarnetState::Clone() const {
  return std::unique_ptr<State>(new GarnetState(*this));
}

std::string GarnetState::Serialize() const {
  std::string out =
      absl::StrCat(current_player_, ",", is_chance_init_, ",", x_, ",", t_, ",",
                   last_action_, ",", return_value_, "\n");
  absl::StrAppend(&out, absl::StrJoin(distribution_, ","));
  return out;
}

GarnetGame::GarnetGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size", kDefaultSize)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)),
      seed_(ParameterValue<int>("seed", kDefaultSeed)),
      num_action_(ParameterValue<int>("num_action", kDefaultNumActions)),
      num_chance_action_(
          ParameterValue<int>("num_chance_action", kDefaultNumChanceActions)),
      sparsity_factor_(
          ParameterValue<double>("sparsity_factor", kDefaultSparsityFactor)),
      eta_(ParameterValue<double>("eta", kDefaultEta)) {}

std::vector<int> GarnetGame::ObservationTensorShape() const {
  // +1 to allow for t_ == horizon.
  return {size_ + horizon_ + 1};
}

std::unique_ptr<State> GarnetGame::DeserializeState(
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
  return absl::make_unique<GarnetState>(
      shared_from_this(), size_, horizon_, seed_, num_action_,
      num_chance_action_, sparsity_factor_, eta_, current_player,
      is_chance_init, x, t, last_action, return_value, distribution);
}

}  // namespace garnet
}  // namespace open_spiel
