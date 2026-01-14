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

#ifndef OPEN_SPIEL_ALGORITHMS_GET_LEGAL_ACTIONS_MAP_H_
#define OPEN_SPIEL_ALGORITHMS_GET_LEGAL_ACTIONS_MAP_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Gets a map of information state (string) to vector of legal actions, by doing
// (depth-limited) tree traversal through the game, for a specific player. To
// do a tree traversal over the entire game, use a negative depth limit. To
// bundle all the legal actions for all players in the same map, use
// kInvalidPlayer.
std::unordered_map<std::string, std::vector<Action>> GetLegalActionsMap(
    const Game& game, int depth_limit, Player player);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_GET_LEGAL_ACTIONS_MAP_H_
