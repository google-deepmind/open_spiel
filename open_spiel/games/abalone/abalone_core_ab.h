// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_AB_H_
#define OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_AB_H_

#include <utility>
#include <vector>

#include "open_spiel/games/abalone/abalone_core.h"

// Unlike other OpenSpiel games, Abalone bundles its own alpha-beta solver
// here instead of relying on the generic open_spiel/algorithms/minimax.h.
// Reason: this search operates directly on the lightweight `core_state`
// struct (plain arrays, no virtual dispatch), which lets it explore many
// more nodes per second than the generic algorithms::AlphaBetaSearch, which
// walks the polymorphic State/Game API (LegalActions/ApplyAction/UndoAction
// virtual calls) on every visited node. Abalone's branching factor is large
// enough that this overhead is significant at useful search depths.

namespace abalone_core {

// @return score relative to _player; range in (-1000, 1000)
int Heuristic(const core_state& _state, CellState _player);

// @return best move with associated value (i.e., the current state's
// value)
std::pair<core_Action, float> AlphaBeta(
    const core_state& _state, int _depth,
    float _alpha = -1.f, float _beta = 1.f,
    std::vector<std::pair<core_Action, float>>* _all_moves = nullptr);

}  // namespace abalone_core

#endif  // OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_AB_H_
