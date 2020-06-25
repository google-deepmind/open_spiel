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


#include "open_spiel/public_states/public_states.h"

#include <memory>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace public_states {

PrivateInformation::PrivateInformation(std::shared_ptr<const Game> game)
    : game_(std::move(game)) {
  SPIEL_CHECK_TRUE(game_->GetType().provides_factored_observation_string);
}

PublicState::PublicState(std::shared_ptr<const Game> game)
    : game_(std::move(game)) {
  SPIEL_CHECK_TRUE(game_->GetType().provides_factored_observation_string);
}

GameWithPublicStatesRegisterer::GameWithPublicStatesRegisterer(
    const GameWithPublicStatesType& game_type, CreateFunc creator) {
  RegisterGame(game_type, std::move(creator));
}

std::shared_ptr<const GameWithPublicStates>
GameWithPublicStatesRegisterer::CreateByName(const std::string& short_name,
                                             const GameParameters& params) {
  auto iter = factories().find(short_name);
  if (iter == factories().end()) {
    SpielFatalError(
        absl::StrCat("Unknown game '", short_name,
                     "'. Available games with public state API are:\n",
                     absl::StrJoin(RegisteredNames(), "\n")));
  } else {
    std::shared_ptr<const Game> base_game =
        GameRegisterer::CreateByName(short_name, params);
    return (iter->second.second)(base_game);
  }
}

std::shared_ptr<const GameWithPublicStates>
GameWithPublicStatesRegisterer::CreateByGame(
    std::shared_ptr<const Game> base_game) {
  const auto& short_name = base_game->GetType().short_name;
  auto iter = factories().find(short_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("The game '", short_name,
                                 "' does not have a public state API. "
                                 " Games which provide public state API are:\n",
                                 absl::StrJoin(RegisteredNames(), "\n")));
  } else {
    return (iter->second.second)(base_game);
  }
}

std::vector<std::string> GameWithPublicStatesRegisterer::RegisteredNames() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) {
    names.push_back(key_val.first);
  }
  return names;
}

std::vector<GameWithPublicStatesType>
GameWithPublicStatesRegisterer::RegisteredGames() {
  std::vector<GameWithPublicStatesType> games;
  for (const auto& key_val : factories()) {
    games.push_back(key_val.second.first);
  }
  return games;
}

bool GameWithPublicStatesRegisterer::IsValidName(
    const std::string& short_name) {
  return factories().find(short_name) != factories().end();
}

void GameWithPublicStatesRegisterer::RegisterGame(
    const GameWithPublicStatesType& game_type,
    GameWithPublicStatesRegisterer::CreateFunc creator) {
  factories()[game_type.short_name] = {game_type, creator};
}

bool IsGameRegisteredWithPublicStates(const std::string& short_name) {
  return GameWithPublicStatesRegisterer::IsValidName(short_name);
}

std::vector<std::string> RegisteredGamesWithPublicStates() {
  return GameWithPublicStatesRegisterer::RegisteredNames();
}

std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string& game_string) {
  return LoadGameWithPublicStates(GameParametersFromString(game_string));
}

std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string& short_name, const GameParameters& params) {
  std::shared_ptr<const GameWithPublicStates> result =
      GameWithPublicStatesRegisterer::CreateByName(short_name, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat(
        "Unable to create game with public state API: ", short_name));
  }
  return result;
}

std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    GameParameters params) {
  auto it = params.find("name");
  if (it == params.end()) {
    SpielFatalError(absl::StrCat("No 'name' parameter in params: ",
                                 GameParametersToString(params)));
  }
  std::string name = it->second.string_value();
  params.erase(it);
  std::shared_ptr<const GameWithPublicStates> result =
      GameWithPublicStatesRegisterer::CreateByName(name, params);
  if (result == nullptr) {
    SpielFatalError(
        absl::StrCat("Unable to create game with public state API: ", name));
  }
  return result;
}

std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    std::shared_ptr<const Game> base_game) {
  std::shared_ptr<const GameWithPublicStates> result =
      GameWithPublicStatesRegisterer::CreateByGame(base_game);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat(
        "Unable to create game with public state API from base game: ",
        base_game->GetType().short_name));
  }
  return result;
}

std::string SerializeGameWithPublicState(const GameWithPublicStates& game,
                                         const PublicState& state) {
  // TODO(author13): implement
  SpielFatalError("SerializeGameWithPublicState() is not implemented yet.");
}

std::pair<std::shared_ptr<const GameWithPublicStates>,
          std::unique_ptr<PublicState>>
DeserializeGameWithPublicState(const std::string& serialized_state) {
  // TODO(author13): implement
  SpielFatalError("DeserializeGameWithPublicState() is not implemented yet.");
}

}  // namespace public_states
}  // namespace open_spiel
