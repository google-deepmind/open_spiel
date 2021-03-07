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
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace public_states {

PrivateInformation::PrivateInformation(std::shared_ptr<const Game> base_game)
    : base_game_(std::move(base_game)) {
  SPIEL_CHECK_TRUE(base_game_->GetType().provides_factored_observation_string);
}

PublicState::PublicState(
    std::shared_ptr<const GameWithPublicStates> public_game)
    : public_game_(std::move(public_game)),
      base_game_(public_game_->GetBaseGame()),
      observer_(base_game_->MakeObserver(
          IIGObservationType{.public_info = true,
                             .perfect_recall = true,
                             .private_info = PrivateInfoType::kNone},
          {})) {
  SPIEL_CHECK_TRUE(base_game_->GetType().provides_factored_observation_string);
}

PublicState::PublicState(
    std::shared_ptr<const GameWithPublicStates> public_game,
    std::vector<PublicTransition> pub_obs_history)
    : public_game_(std::move(public_game)),
      base_game_(public_game_->GetBaseGame()) {
  SPIEL_CHECK_TRUE(base_game_->GetType().provides_factored_observation_string);
  SPIEL_CHECK_EQ(pub_obs_history[0], kStartOfGamePublicObservation);
  for (int i = 1; i < pub_obs_history.size(); ++i) {
    ApplyPublicTransition(pub_obs_history[i]);
  }
}

std::vector<double> PublicState::ReachProbsTensor(
    const std::vector<ReachProbs>& reach_probs) const {
  std::vector<int> sizes = public_game_->MaxDistinctPrivateInformationsCount();
  const int reach_probs_size = absl::c_accumulate(sizes, 0);

  std::vector<double> tensor(reach_probs_size, kTensorUnusedSlotValue);
  // Place reach probs of each player.
  int player_offset = 0;
  for (int i = 0; i < base_game_->NumPlayers(); ++i) {
    const std::vector<PrivateInformation> player_privates =
        GetPrivateInformations(i);
    SPIEL_CHECK_EQ(player_privates.size(), reach_probs[i].probs.size());
    for (int j = 0; j < player_privates.size(); ++j) {
      SPIEL_CHECK_EQ(player_privates[j].ReachProbsIndex(), j);
      // We use NetworkIndex because there can be "holes" in the input
      // These "holes" have value 0 by default.
      const int placement = player_offset + player_privates[j].NetworkIndex();
      tensor[placement] = reach_probs[i].probs[j];
    }
    player_offset += sizes[i];
  }
  SPIEL_CHECK_EQ(player_offset, reach_probs_size);
  return tensor;
}

std::vector<double> PublicState::ToTensor(
    const std::vector<ReachProbs>& reach_probs) const {
  std::vector<double> tensor = ReachProbsTensor(reach_probs);
  SPIEL_CHECK_EQ(tensor.size(),
                 public_game_->SumMaxDistinctPrivateInformations());

  std::vector<double> public_features = PublicFeaturesTensor();
  SPIEL_CHECK_EQ(public_features.size(), public_game_->NumPublicFeatures());

  const int reach_probs_size = tensor.size();
  const int features_size = public_features.size();
  tensor.resize(reach_probs_size + features_size);
  tensor.insert(tensor.end(), public_features.begin(), public_features.end());
  return tensor;
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

std::vector<GameWithPublicStatesType> RegisteredGameTypesWithPublicStates() {
  return GameWithPublicStatesRegisterer::RegisteredGames();
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

std::ostream& operator<<(std::ostream& stream, const CfPrivValues& values) {
  return stream << "CfPrivValues{player=" << values.player
        << ", cfvs=" << values.cfvs << "}";
}

std::ostream& operator<<(std::ostream& stream, const CfActionValues& values) {
  return stream << "CfActionValues{player=" << values.player
        << ", cfavs=" << values.cfavs << "}";
}


}  // namespace public_states
}  // namespace open_spiel
