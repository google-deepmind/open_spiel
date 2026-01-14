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

#ifndef OPEN_SPIEL_ALGORITHMS_GET_ALL_HISTORIES_H_
#define OPEN_SPIEL_ALGORITHMS_GET_ALL_HISTORIES_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Returns a vector of states corresponding to unique histories in the game.
//
// For small games only!
//
// Use this implementation with caution as it does a recursive tree
// walk of the game and could easily fill up memory for larger games or games
// with long horizons.
//
// Currently only works for sequential games.
//
// Note: negative depth limit means no limit, 0 means only root, etc..
// The default arguments will return all decision nodes in the game.
std::vector<std::unique_ptr<State>> GetAllHistories(
    const Game& game, int depth_limit = -1, bool include_terminals = false,
    bool include_chance_states = false);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_GET_ALL_HISTORIES_H_
