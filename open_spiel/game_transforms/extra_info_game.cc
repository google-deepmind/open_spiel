// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/game_transforms/extra_info_game.h"

#include <algorithm>
#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

// These parameters are the most-general case. The actual game may be simpler.
const GameType kGameType{/*short_name=*/"extra_info",
                         /*long_name=*/"Extra Info Version of a Regular Game",
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
                         {{"game", GameParameter(GameParameter::Type::kGame,
                                                 /*is_mandatory=*/true)},
                         {"extra_observation_tensor_size",GameParameter(0)},
                         {"extra_information_tensor_size",GameParameter(0)}},
                         /*default_loadable=*/false,
                         /*provides_factored_observation_string=*/true,
                         /*is_concrete=*/false};

GameType ExtraInfoGameType(GameType game_type) {
  game_type.short_name = kGameType.short_name;
  game_type.long_name = absl::StrCat("Extra Info ", game_type.long_name);
  return game_type;
}

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = ExtraInfoGameType(game->GetType());
  return std::shared_ptr<const Game>(new ExtraInfoGame(game, game_type, params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

ExtraInfoState::ExtraInfoState(
    std::shared_ptr<const Game> game,
    std::unique_ptr<State> state,
    int extra_observation_tensor_size,
    int extra_information_tensor_size)
    : WrappedState(game, std::move(state)),
      extra_information_tensor_(
          game->NumPlayers(),
          std::vector<float>(extra_information_tensor_size, 0.0f)),
      extra_observation_tensor_(
          game->NumPlayers(),
          std::vector<float>(extra_observation_tensor_size, 0.0f)),
      extra_observation_string_(game->NumPlayers()),
      extra_information_state_string_(game->NumPlayers()) {}


void ExtraInfoState::SetExtraObservationTensor(
    Player player, const std::vector<float>& values) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  SPIEL_CHECK_LT(player, extra_observation_tensor_.size());
  SPIEL_CHECK_EQ(values.size(), extra_observation_tensor_[player].size());

  extra_observation_tensor_[player] = values;
}

void ExtraInfoState::SetExtraObservationTensor(
    const std::vector<float>& values) {
  for (Player player = 0; player < game_->NumPlayers(); ++player) {
    SetExtraObservationTensor(player, values);
  }
}

void ExtraInfoState::SetExtraInformationStateTensor(
    Player player, const std::vector<float>& values) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  SPIEL_CHECK_LT(player, extra_information_tensor_.size());
  SPIEL_CHECK_EQ(values.size(), extra_information_tensor_[player].size());

  extra_information_tensor_[player] = values;
}

void ExtraInfoState::SetExtraInformationStateTensor(
    const std::vector<float>& values) {
  for (Player player = 0; player < game_->NumPlayers(); ++player) {
    SetExtraInformationStateTensor(player, values);
  }
}

void ExtraInfoState::SetExtraObservationString(
    Player player, const std::string& value) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  extra_observation_string_[player] = value;
}

void ExtraInfoState::SetExtraObservationString(
    const std::string& value) {
  for (Player player = 0; player < game_->NumPlayers(); ++player) {
    SetExtraObservationString(player, value);
  }
}

void ExtraInfoState::SetExtraInformationStateString(
    Player player, const std::string& value) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  extra_information_state_string_[player] = value;
}

void ExtraInfoState::SetExtraInformationStateString(
    const std::string& value) {
  for (Player player = 0; player < game_->NumPlayers(); ++player) {
    SetExtraInformationStateString(player, value);
  }
}

void ExtraInfoState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  const int base_size = state_->GetGame()->ObservationTensorSize();

  state_->ObservationTensor(player, values.subspan(0, base_size));

  std::copy(extra_observation_tensor_[player].begin(),
            extra_observation_tensor_[player].end(),
            values.begin() + base_size);
}

void ExtraInfoState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  const int base_size = state_->GetGame()->InformationStateTensorSize();

  state_->InformationStateTensor(player, values.subspan(0, base_size));

  std::copy(extra_information_tensor_[player].begin(),
            extra_information_tensor_[player].end(),
            values.begin() + base_size);
}

std::string ExtraInfoState::ObservationString(Player player) const {
  return state_->ObservationString(player) +
         extra_observation_string_[player];
}

std::string ExtraInfoState::InformationStateString(Player player) const {
  return state_->InformationStateString(player) +
         extra_information_state_string_[player];
}


std::vector<int> ExtraInfoGame::ObservationTensorShape() const {
  if (extra_observation_tensor_size_ == 0) {
    return game_->ObservationTensorShape();
  }
  return {game_->ObservationTensorSize() + extra_observation_tensor_size_};
}


std::vector<int> ExtraInfoGame::InformationStateTensorShape() const {
  if (extra_information_tensor_size_ == 0) {
    return game_->InformationStateTensorShape();
  }
  return {game_->InformationStateTensorSize() + extra_information_tensor_size_};
}

ExtraInfoGame::ExtraInfoGame(std::shared_ptr<const Game> game, GameType game_type,
                       GameParameters game_parameters)
    : WrappedGame(game, game_type, game_parameters) {
       extra_observation_tensor_size_ = ParameterValue<int>("extra_observation_tensor_size", 0);
       extra_information_tensor_size_ = ParameterValue<int>("extra_information_tensor_size", 0);
}


std::shared_ptr<const Game> CreateExtraInfoGame(
    const Game& wrapped_game, const GameParameters& params) {
  GameType game_type = ExtraInfoGameType(wrapped_game.GetType());

  return std::make_shared<const ExtraInfoGame>(
      wrapped_game.shared_from_this(), game_type, params);
}

std::shared_ptr<const Game> CreateExtraInfoGame(
    const std::string& wrapped_game_name, const GameParameters& params) {
  auto game = LoadGame(wrapped_game_name);
  return CreateExtraInfoGame(*game, params);
}




}  // namespace open_spiel
