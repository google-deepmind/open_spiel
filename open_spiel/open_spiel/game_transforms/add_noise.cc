// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/game_transforms/add_noise.h"

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace add_noise {
namespace {

// These parameters are the most-general case. The actual game may be simpler.
const GameType kGameType{
    /*short_name=*/"add_noise",
    /*long_name=*/"Add noise to terminal utilities.",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {{"game", GameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)},
     {"epsilon", GameParameter(1.0, /*is_mandatory=*/true)},
     {"seed", GameParameter(1, /*is_mandatory=*/true)}},
    /*default_loadable=*/false,
    /*provides_factored_observation_string=*/true,
    /*is_concrete=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = game->GetType();
  // Only terminal reward models are supported.
  SPIEL_CHECK_EQ(game_type.reward_model, GameType::RewardModel::kTerminal);

  game_type.short_name = kGameType.short_name;
  game_type.long_name =
      absl::StrCat("Add noise to", " game=", game_type.long_name,
                   " epsilon=", params.at("epsilon").double_value(),
                   " seed=", params.at("seed").int_value());
  return std::make_shared<AddNoiseGame>(game, game_type, params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

AddNoiseGame::AddNoiseGame(std::shared_ptr<const Game> game, GameType game_type,
                           GameParameters game_parameters)
    : WrappedGame(game, game_type, game_parameters),
      epsilon_(ParameterValue<double>("epsilon")),
      rng_(ParameterValue<int>("seed")) {}

std::unique_ptr<State> AddNoiseGame::NewInitialState() const {
  return std::make_unique<AddNoiseState>(shared_from_this(),
                                         game_->NewInitialState());
}

double AddNoiseGame::GetNoise(const AddNoiseState& state) {
  std::string state_str = state.HistoryString();
  auto it = noise_table_.find(state_str);
  if (it != noise_table_.end()) {
    return it->second;
  }

  std::uniform_real_distribution<double> dist(-epsilon_, epsilon_);
  double noise = dist(rng_);
  noise_table_[state_str] = noise;
  return noise;
}

double AddNoiseGame::MaxUtility() const {
  return WrappedGame::MaxUtility() + epsilon_;
}

double AddNoiseGame::MinUtility() const {
  return WrappedGame::MinUtility() - epsilon_;
}

AddNoiseState::AddNoiseState(std::shared_ptr<const Game> transformed_game,
                             std::unique_ptr<State> state)
    : WrappedState(transformed_game, std::move(state)) {}

std::vector<double> AddNoiseState::Returns() const {
  std::vector<double> returns = state_->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);

  if (state_->IsTerminal()) {
    auto const_noise_game = down_cast<const AddNoiseGame*>(game_.get());
    AddNoiseGame* noise_game = const_cast<AddNoiseGame*>(const_noise_game);
    double noise = noise_game->GetNoise(*this);
    returns[0] += noise;
    returns[1] -= noise;
  }

  return returns;
}

std::vector<double> AddNoiseState::Rewards() const {
  if (IsTerminal()) {
    return Returns();
  } else {
    SPIEL_CHECK_FALSE(IsChanceNode());
    return std::vector<double>(num_players_, 0.0);
  }
}

}  // namespace add_noise
}  // namespace open_spiel
