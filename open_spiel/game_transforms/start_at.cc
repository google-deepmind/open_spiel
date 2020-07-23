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

#include "open_spiel/game_transforms/start_at.h"

namespace open_spiel {
namespace {

constexpr char kHistorySeparator = ';';

// These parameters are the most-general case. The actual game may be simpler.
const GameType kGameType{
    /*short_name=*/"start_at",
    /*long_name=*/"Start at specified subgame of a regular game.",
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
    {
        {"game", GameParameter(
            GameParameter::Type::kGame, /*is_mandatory=*/true)},
        {"history", GameParameter(
            GameParameter::Type::kString, /*is_mandatory=*/true)}
    },
    /*default_loadable=*/false,
    /*provides_factored_observation_string=*/true,
};


std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = game->GetType();
  game_type.short_name = kGameType.short_name;
  game_type.long_name = absl::StrCat(
      "StartAt history=", params.at("history").string_value(),
      "game=", game_type.long_name);

  return std::shared_ptr<const Game>(
      new StartAtTransformationGame(game, game_type, params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// History can be a string of numbers separated by kHistorySeparators (commas).
std::vector<Action> HistoryFromString(const std::string& str) {
  std::vector<Action> history;
  if (str.empty()) return history;  // Identity transformation.

  // No beginning comma.
  SPIEL_CHECK_TRUE(str[0] != kHistorySeparator);
  // No trailing comma.
  SPIEL_CHECK_TRUE(str[str.size()-1] != kHistorySeparator);

  int action = 0;
  for (int i = 0; i < str.size(); ++i) {
    // Only allowed chars.
    SPIEL_CHECK_TRUE(str[i] == kHistorySeparator
      || (str[i] >= '0' && str[i] <= '9'));
    // No consecutive commas.
    SPIEL_CHECK_TRUE(
      i == 0 || str[i] != kHistorySeparator || str[i-1] != kHistorySeparator);

    action = (str[i] == kHistorySeparator)
        ? 0
        : action*10 + (str[i] - '0');
  }
  history.push_back(action);
  return history;
}

StartAtTransformationGame::StartAtTransformationGame(
    std::shared_ptr<const Game> game, GameType game_type,
    GameParameters game_parameters)
    : WrappedGame(game, game_type, game_parameters),
      start_at_(HistoryFromString(
          game_parameters.at("history").string_value())) {}

std::unique_ptr<State> StartAtTransformationGame::NewInitialState() const {
  std::unique_ptr<State> state = game_->NewInitialState();
  for (const Action& a : start_at_) state->ApplyAction(a);
  return std::make_unique<StartAtTransformationState>(
      shared_from_this(), std::move(state));
}

StartAtTransformationState::StartAtTransformationState(
    std::shared_ptr<const Game> game, std::unique_ptr<State> state)
    : WrappedState(game, std::move(state)) {
  // Check that the state is indeed limited to the specified subgame.
  const StartAtTransformationGame* start_game =
      subclass_cast<const StartAtTransformationGame*>(game_.get());
  const std::vector<Action>& start_at = start_game->StartAt();
  const std::vector<PlayerAction>& state_history = state_->FullHistory();
  SPIEL_CHECK_GE(state_history.size(), start_at.size());
  for (int i = 0; i < start_at.size(); ++i) {
    SPIEL_CHECK_EQ(state_history[i].action, start_at[i]);
  }
}

}  // namespace open_spiel
