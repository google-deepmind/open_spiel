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

#include "open_spiel/algorithms/maxn.h"

#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

std::vector<double> _maxn(
    const State* state, int depth,
    std::function<double(const State&, Player player)> value_function,
    Action* best_action) {
  const int num_players = state->NumPlayers();

  if (state->IsTerminal()) {
    return state->Returns();
  }

  if (depth == 0 && !value_function) {
    SpielFatalError(
        "We assume we can walk the full depth of the tree. "
        "Try increasing depth or provide a value_function.");
  }

  if (depth == 0) {
    std::vector<double> values(num_players);
    for (Player p = 0; p < num_players; ++p) {
      values[p] = value_function(*state, p);
    }
    return values;
  }

  Player player = state->CurrentPlayer();
  if (state->IsChanceNode()) {
    std::vector<double> values(num_players, 0.0);
    for (const auto& actionprob : state->ChanceOutcomes()) {
      std::unique_ptr<State> child_state = state->Child(actionprob.first);
      std::vector<double> child_values =
          _maxn(child_state.get(), depth, value_function,
                /*best_action=*/nullptr);
      for (Player p = 0; p < num_players; ++p) {
        values[p] += actionprob.second * child_values[p];
      }
    }
    return values;
  } else {
    double value = -std::numeric_limits<double>::infinity();
    std::vector<double> values(num_players, 0);

    for (Action action : state->LegalActions()) {
      std::unique_ptr<State> child_state = state->Child(action);
      std::vector<double> child_values =
          _maxn(child_state.get(),
                /*depth=*/depth - 1, value_function,
                /*best_action=*/nullptr);

      if (child_values[player] > value) {
        value = child_values[player];
        values = child_values;
        if (best_action != nullptr) {
          *best_action = action;
        }
      }
    }
    return values;
  }
}
}  // namespace

std::pair<std::vector<double>, Action> MaxNSearch(
    const Game& game, const State* state,
    std::function<double(const State&, Player player)> value_function,
    int depth_limit) {
  GameType game_info = game.GetType();
  SPIEL_CHECK_TRUE(
      game_info.chance_mode == GameType::ChanceMode::kDeterministic ||
      game_info.chance_mode == GameType::ChanceMode::kExplicitStochastic);
  // Do not check perfect information. Used by PIMC.
  SPIEL_CHECK_EQ(game_info.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_info.reward_model, GameType::RewardModel::kTerminal);

  std::unique_ptr<State> search_root;
  if (state == nullptr) {
    search_root = game.NewInitialState();
  } else {
    search_root = state->Clone();
  }

  SPIEL_CHECK_FALSE(search_root->IsChanceNode());

  Action best_action = kInvalidAction;
  std::vector<double> values = _maxn(search_root.get(), /*depth=*/depth_limit,
                                     value_function, &best_action);

  return {values, best_action};
}

}  // namespace algorithms
}  // namespace open_spiel
