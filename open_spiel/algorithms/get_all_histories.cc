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

#include "open_spiel/algorithms/get_all_histories.h"

namespace open_spiel {
namespace algorithms {
namespace {

// Walk a subgame and return all histories contained in the subgames. This does
// a recursive tree walk, therefore all valid sequences must have finite number
// of actions.
// Requires State::Clone() to be implemented.
// Use with extreme caution!
// Currently not implemented for simultaneous games.
void GetSubgameHistories(State* state,
                         std::vector<std::unique_ptr<State>>* all_histories,
                         int depth_limit, int depth, bool include_terminals,
                         bool include_chance_states) {
  if (state->IsTerminal()) {
    if (include_terminals) {
      // Include, then terminate recursion.
      all_histories->push_back(state->Clone());
    }
    return;
  }

  if (depth_limit >= 0 && depth > depth_limit) {
    return;
  }

  if (!state->IsChanceNode() || include_chance_states) {
    all_histories->push_back(state->Clone());
  }

  for (auto action : state->LegalActions()) {
    auto next_state = state->Clone();
    next_state->ApplyAction(action);
    GetSubgameHistories(next_state.get(), all_histories, depth_limit, depth + 1,
                        include_terminals, include_chance_states);
  }
}

}  // namespace

std::vector<std::unique_ptr<State>> GetAllHistories(
    const Game& game, int depth_limit, bool include_terminals,
    bool include_chance_states) {
  // Get the root state.
  std::unique_ptr<State> state = game.NewInitialState();
  std::vector<std::unique_ptr<State>> all_histories;

  // Then, do a recursive tree walk to fill up the vector.
  GetSubgameHistories(state.get(), &all_histories, depth_limit, 0,
                      include_terminals, include_chance_states);

  if (all_histories.empty()) {
    SpielFatalError("GetSubgameHistories returned 0 histories!");
  }

  return all_histories;
}

}  // namespace algorithms
}  // namespace open_spiel
