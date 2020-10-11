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
#include "open_spiel/algorithms/get_all_histories.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"


// Counts all states and prints them to the console.
int main(int argc, char** argv) {
  for (std::string_view game_name :
       {std::string("tic_tac_toe"), std::string("kuhn_poker"),
        std::string("leduc_poker"), std::string("liars_dice"),
        open_spiel::TurnBasedGoofspielGameString(4),
        open_spiel::TurnBasedGoofspielGameString(5),
        open_spiel::TurnBasedGoofspielGameString(6)}) {
    std::shared_ptr<const open_spiel::Game> game =
        open_spiel::LoadGame(std::string(game_name));
    std::vector<std::unique_ptr<open_spiel::State>> all_histories =
        open_spiel::algorithms::GetAllHistories(*game, /*depth_limit=*/-1,
                                                /*include_terminals=*/true,
                                                /*include_chance_states=*/true);
    absl::flat_hash_set<std::string> nonterminal_states;
    absl::flat_hash_set<std::string> terminal_states;
    const int num_histories = all_histories.size();
    int num_terminal_histories = 0;
    int num_chance_nodes = 0;
    for (const auto& state : all_histories) {
      if (state->CurrentPlayer() >= 0) {
        if (game->GetType().information ==
            open_spiel::GameType::Information::kPerfectInformation) {
          nonterminal_states.insert(state->ToString());
        } else {
          nonterminal_states.insert(state->InformationStateString());
        }
      }
      if (state->IsTerminal()) {
         ++num_terminal_histories;
         terminal_states.insert(state->ToString());
      }
      if (state->IsChanceNode()) ++num_chance_nodes;
    }
    const int num_nonterminal_states = nonterminal_states.size();
    const int num_terminal_states = terminal_states.size();
    std::cout << absl::StreamFormat(
                     "Game: %s, num_histories: %i, num_terminal_histories: %i, "
                     "num_chance_nodes: %i, num_nonterminal_states: %i, "
                     "num_terminal_states: %i",
                     game_name, num_histories, num_terminal_histories,
                     num_chance_nodes, num_nonterminal_states,
                     num_terminal_states)
              << std::endl;
  }
}
