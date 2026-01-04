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

#include "open_spiel/algorithms/tensor_game_utils.h"

#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

using open_spiel::tensor_game::TensorGame;

std::shared_ptr<const TensorGame> LoadTensorGame(const std::string& name) {
  std::shared_ptr<const Game> game = LoadGame(name);
  // Make sure it is indeed a tensor game.
  const TensorGame* tensor_game = dynamic_cast<const TensorGame*>(game.get());
  if (tensor_game == nullptr) {
    // If it is not already a tensor game, check if it is an NFG.
    // If so, convert it.
    const NormalFormGame* nfg = dynamic_cast<const NormalFormGame*>(game.get());
    if (nfg != nullptr) {
      return AsTensorGame(nfg);
    } else {
      SpielFatalError(absl::StrCat("Cannot load ", name, " as a tensor game."));
    }
  }
  return std::static_pointer_cast<const TensorGame>(game);
}

std::shared_ptr<const TensorGame> AsTensorGame(const Game* game) {
  const NormalFormGame* nfg = dynamic_cast<const NormalFormGame*>(game);
  SPIEL_CHECK_TRUE(nfg);
  return AsTensorGame(nfg);
}

std::shared_ptr<const TensorGame> AsTensorGame(const NormalFormGame* game) {
  const int num_players = game->NumPlayers();
  std::unique_ptr<State> initial_state = game->NewInitialState();
  std::vector<std::vector<Action>> legal_actions(num_players);
  std::vector<std::vector<std::string>> action_names(num_players);
  for (Player player = 0; player < num_players; ++player) {
    legal_actions[player] = initial_state->LegalActions(player);
    for (const Action& action : legal_actions[player]) {
      action_names[player].push_back(
          initial_state->ActionToString(player, action));
    }
  }
  std::vector<std::vector<double>> utils(num_players);

  GameType type = game->GetType();
  type.min_num_players = num_players;
  type.max_num_players = num_players;

  std::vector<Action> actions(num_players);
  bool last_entry;
  do {
    std::unique_ptr<State> clone = initial_state->Clone();
    clone->ApplyActions(actions);
    SPIEL_CHECK_TRUE(clone->IsTerminal());
    std::vector<double> returns = clone->Returns();
    SPIEL_CHECK_EQ(returns.size(), num_players);
    for (Player player = 0; player < num_players; ++player) {
      utils[player].push_back(returns[player]);
    }
    last_entry = true;
    for (Player player = num_players - 1; player >= 0; --player) {
      if (++actions[player] < legal_actions[player].size()) {
        last_entry = false;
        break;
      } else {
        actions[player] = 0;
      }
    }
  } while (!last_entry);

  return std::shared_ptr<TensorGame>(
      new TensorGame(type, {}, action_names, utils));
}

}  // namespace algorithms
}  // namespace open_spiel
