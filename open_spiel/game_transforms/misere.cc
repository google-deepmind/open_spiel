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

#include "open_spiel/game_transforms/misere.h"

namespace open_spiel {
namespace {

// These parameters are the most-general case. The actual game may be simpler.
const GameType kGameType{
    /*short_name=*/"misere",
    /*long_name=*/"Misere Version of a Regular Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    {{"game",
      GameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)}}};

GameType MisereGameType(GameType game_type) {
  game_type.short_name = kGameType.short_name;
  game_type.long_name = absl::StrCat("Misere ", game_type.long_name);
  return game_type;
}

std::unique_ptr<Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = MisereGameType(game->GetType());
  return std::unique_ptr<Game>(
      new MisereGame(std::move(game), game_type, params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

MisereGame::MisereGame(std::unique_ptr<Game> game, GameType game_type,
                       GameParameters game_parameters)
    : WrappedGame(std::move(game), game_type, game_parameters) {}

}  // namespace open_spiel
