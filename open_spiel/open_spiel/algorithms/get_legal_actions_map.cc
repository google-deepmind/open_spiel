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

#include "open_spiel/algorithms/get_legal_actions_map.h"

namespace open_spiel {
namespace algorithms {
namespace {

// Do the tree traversal to fill the map. This function does a depth-first
// search of all the subtrees to fill the map for all the information states.
void FillMap(const State& state,
             std::unordered_map<std::string, std::vector<Action>>* map,
             int depth_limit, int depth, Player player) {
  if (state.IsTerminal()) {
    return;
  }

  if (depth_limit >= 0 && depth > depth_limit) {
    return;
  }

  if (state.IsChanceNode()) {
    // Do nothing at chance nodes (no information states).
  } else if (state.IsSimultaneousNode()) {
    // Many players can play at this node.
    for (auto p = Player{0}; p < state.NumPlayers(); ++p) {
      if (player == kInvalidPlayer || p == player) {
        std::string info_state = state.InformationStateString(p);
        if (map->find(info_state) == map->end()) {
          // Only add it if we don't already have it.
          std::vector<Action> legal_actions = state.LegalActions(p);
          (*map)[info_state] = legal_actions;
        }
      }
    }
  } else {
    // Regular decision node.
    if (player == kInvalidPlayer || state.CurrentPlayer() == player) {
      std::string info_state = state.InformationStateString();
      if (map->find(info_state) == map->end()) {
        // Only add it if we don't already have it.
        std::vector<Action> legal_actions = state.LegalActions();
        (*map)[info_state] = legal_actions;
      }
    }
  }

  // Recursively fill the map for each subtree below.
  for (auto action : state.LegalActions()) {
    std::unique_ptr<State> next_state = state.Child(action);
    FillMap(*next_state, map, depth_limit, depth + 1, player);
  }
}

}  // namespace

std::unordered_map<std::string, std::vector<Action>> GetLegalActionsMap(
    const Game& game, int depth_limit, Player player) {
  std::unordered_map<std::string, std::vector<Action>> map;
  std::unique_ptr<State> initial_state = game.NewInitialState();
  FillMap(*initial_state, &map, depth_limit, 0, player);
  return map;
}

}  // namespace algorithms
}  // namespace open_spiel
