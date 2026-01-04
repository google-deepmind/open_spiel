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

#include "open_spiel/game_transforms/zerosum.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

// These parameters are the most-general case, except for utility which is
// zero-sum. The actual game may be simpler.
const GameType kGameType{/*short_name=*/"zerosum",
                         /*long_name=*/"ZeroSum Version of a Regular Game",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kSampledStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/100,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         {{"game", GameParameter(GameParameter::Type::kGame,
                                                 /*is_mandatory=*/true)}},
                         /*default_loadable=*/false,
                         /*provides_factored_observation_string=*/true,
                         /*is_concrete=*/false};

GameType ZeroSumGameType(GameType game_type) {
  game_type.short_name = kGameType.short_name;
  game_type.long_name = absl::StrCat("ZeroSum ", game_type.long_name);
  game_type.utility = GameType::Utility::kZeroSum;
  return game_type;
}

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = ZeroSumGameType(game->GetType());
  return std::shared_ptr<const Game>(new ZeroSumGame(game, game_type, params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

ZeroSumGame::ZeroSumGame(std::shared_ptr<const Game> game, GameType game_type,
                       GameParameters game_parameters)
    : WrappedGame(game, game_type, game_parameters) {}

}  // namespace open_spiel
