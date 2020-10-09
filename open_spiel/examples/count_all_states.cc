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

#include <string>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"


// Counts all states and prints them to the console.
int main(int argc, char** argv) {
  for (std::string_view game_name :
       {std::string("kuhn_poker"), std::string("leduc_poker"),
        std::string("liars_dice"), open_spiel::TurnBasedGoofspielGameString(4),
        open_spiel::TurnBasedGoofspielGameString(5),
        open_spiel::TurnBasedGoofspielGameString(6)}) {
    std::shared_ptr<const open_spiel::Game> game =
        open_spiel::LoadGame(std::string(game_name));
    std::map<std::string, std::unique_ptr<open_spiel::State>> all_states =
        open_spiel::algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                             /*include_terminals=*/true,
                                             /*include_chance_states=*/true);
    absl::flat_hash_set<std::string> all_infostates;
    const int num_histories = all_states.size();
    int num_terminal_states = 0;
    int num_chance_nodes = 0;
    // TODO: Fix counting of information states for some games after having a
    // GetAllHistories. Right now the counting of information states will not
    // be correct for perfect information games. See this issue for details:
    // https://github.com/deepmind/open_spiel/issues/401
    for (const auto& [_, state] : all_states) {
      if (state->CurrentPlayer() >= 0) {
        all_infostates.insert(state->InformationStateString());
      }
      if (state->IsTerminal()) ++num_terminal_states;
      if (state->IsChanceNode()) ++num_chance_nodes;
    }
    const int num_infostates = all_infostates.size();
    std::cout << absl::StreamFormat(
                     "Game: %s, num_histories: %i, num_terminals: %i, "
                     "num_chance: %i, num_infostates: %i",
                     game_name, num_histories, num_terminal_states,
                     num_chance_nodes, num_infostates)
              << std::endl;
  }
}
