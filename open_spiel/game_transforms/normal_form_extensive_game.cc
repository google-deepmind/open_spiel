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

#include "open_spiel/game_transforms/normal_form_extensive_game.h"

#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

using open_spiel::tensor_game::TensorGame;

// These parameters reflect the most-general game, with the maximum
// API coverage. The actual game may be simpler and might not provide
// all the interfaces.
// This is used as a placeholder for game registration. The actual instantiated
// game will have more accurate information.
const GameType kGameType{
    /*short_name=*/"normal_form_extensive_game",
    /*long_name=*/"Normal-Form Version of an Extensive Game",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {{"game",
      GameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)}},
    /*default_loadable=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return ExtensiveToTensorGame(*LoadGame(params.at("game").game_value()));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::shared_ptr<const TensorGame> ExtensiveToTensorGame(const Game& game) {
  std::vector<std::vector<std::string>> action_names(game.NumPlayers());

  GameType type = game.GetType();

  std::vector<algorithms::DeterministicTabularPolicy> policies;
  for (Player player = 0; player < game.NumPlayers(); ++player) {
    algorithms::DeterministicTabularPolicy policy(game, player);
    do {
      action_names[player].push_back(policy.ToString(/*delimiter=*/" --- "));
    } while (policy.NextPolicy());
    policy.ResetDefaultPolicy();
    policies.push_back(policy);
  }
  std::vector<const Policy*> policy_ptrs(policies.size());
    for (Player player = 0; player < game.NumPlayers(); ++player) {
      policy_ptrs[player] = &policies[player];
    }
  const std::unique_ptr<State> initial_state = game.NewInitialState();
  std::vector<std::vector<double>> utils(game.NumPlayers());
  bool last_entry;
  do {
    std::vector<double> returns = algorithms::ExpectedReturns(
        *initial_state, policy_ptrs, /*depth_limit=*/-1);
    for (Player player = 0; player < game.NumPlayers(); ++player) {
      utils[player].push_back(returns[player]);
    }
    last_entry = true;
    for (auto policy = policies.rbegin(); policy != policies.rend(); ++policy) {
      if (policy->NextPolicy()) {
        last_entry = false;
        break;
      } else {
        policy->ResetDefaultPolicy();
      }
    }
  } while (!last_entry);

  return tensor_game::CreateTensorGame(kGameType.short_name,
                                       "Normal-form " + type.long_name,
                                       action_names, utils);
}

}  // namespace open_spiel
