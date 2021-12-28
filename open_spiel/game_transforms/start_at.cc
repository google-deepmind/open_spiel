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

#include "open_spiel/game_transforms/start_at.h"

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr char kActionSeparator = ';';

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
    {{"game", GameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)},
     {"history",
      GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)}},
    /*default_loadable=*/false,
    /*provides_factored_observation_string=*/true,
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto game = LoadGame(params.at("game").game_value());
  GameType game_type = game->GetType();
  game_type.short_name = kGameType.short_name;
  game_type.long_name =
      absl::StrCat("StartAt history=", params.at("history").string_value(),
                   " game=", game_type.long_name);
  return std::make_shared<StartAtTransformationGame>(game, game_type, params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// History is represented by a string of numbers separated by kActionSeparator.
std::vector<Action> HistoryFromString(const std::string& str) {
  std::vector<Action> history;
  if (str.empty()) return history;  // Identity transformation.

  std::vector<absl::string_view> str_actions =
      absl::StrSplit(str, kActionSeparator);
  for (const auto& str_action : str_actions) {
    Action a;
    if (!absl::SimpleAtoi(str_action, &a)) {
      SpielFatalError(
          absl::StrCat("Error when parsing the action: ", str_action));
    }
    history.push_back(a);
  }
  return history;
}

std::unique_ptr<State> StateFromHistory(std::shared_ptr<const Game> game,
                                        const std::vector<Action>& history) {
  std::unique_ptr<State> state = game->NewInitialState();
  for (const Action& a : history) state->ApplyAction(a);
  return state;
}

StartAtTransformationGame::StartAtTransformationGame(
    std::shared_ptr<const Game> game, GameType game_type,
    GameParameters game_parameters)
    : WrappedGame(game, game_type, game_parameters),
      start_state_(StateFromHistory(
          game,
          HistoryFromString(game_parameters.at("history").string_value()))) {}

std::unique_ptr<State> StartAtTransformationGame::NewInitialState() const {
  return std::make_unique<StartAtTransformationState>(shared_from_this(),
                                                      start_state_->Clone());
}

StartAtTransformationState::StartAtTransformationState(
    std::shared_ptr<const Game> transformed_game, std::unique_ptr<State> state)
    : WrappedState(transformed_game, std::move(state)) {
  const auto* start_at_game = open_spiel::down_cast<
      const StartAtTransformationGame*>(game_.get());
  const std::vector<State::PlayerAction> start_history =
      start_at_game->StartAtHistory();
  const std::vector<State::PlayerAction> wrap_history = state_->FullHistory();
  SPIEL_DCHECK_TRUE(std::equal(start_history.begin(), start_history.end(),
                               wrap_history.begin()));
}

}  // namespace open_spiel
