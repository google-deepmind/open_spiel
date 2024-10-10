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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/algorithms/get_all_histories.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game_string, "kuhn_poker", "Game to count states for.");


using open_spiel::GameType;
using open_spiel::StateType;
using open_spiel::algorithms::GetAllHistories;

// Counts the number of states in the game according to various measures.
//   - histories is a sequence of moves (for all players) and chance outcomes
//   - states is for imperfect information games, information states (i.e.
//     sets of histories which are indistinguishable to the acting player);
//     for example in poker, the acting player's private cards plus the sequence
//     of bets and public cards, for perfect information games, Markov states
//     (i.e. sets of histories which yield the same result with the same actions
//     applied), e.g. in tic-tac-toe the current state of the board, regardless
//     of the order in which the moves were played.
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string game_name = absl::GetFlag(FLAGS_game_string);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);
  std::vector<std::unique_ptr<open_spiel::State>> all_histories =
      GetAllHistories(*game, /*depth_limit=*/-1, /*include_terminals=*/true,
                      /*include_chance_states=*/true);
  absl::flat_hash_set<std::string> nonterminal_states;
  absl::flat_hash_set<std::string> terminal_states;
  const int num_histories = all_histories.size();
  int num_terminal_histories = 0;
  int num_chance_nodes = 0;
  for (const auto& state : all_histories) {
    switch (state->GetType()) {
      case StateType::kDecision:
        if (game->GetType().information ==
            GameType::Information::kPerfectInformation) {
          nonterminal_states.insert(state->ToString());
        } else {
          nonterminal_states.insert(state->InformationStateString());
        }
        break;
      case StateType::kTerminal:
        ++num_terminal_histories;
        terminal_states.insert(state->ToString());
        break;
      case StateType::kChance:
        ++num_chance_nodes;
        break;
      case StateType::kMeanField:
        open_spiel::SpielFatalError("kMeanField not handeled.");
    }
  }
  const int num_nonterminal_states = nonterminal_states.size();
  const int num_terminal_states = terminal_states.size();
  std::cout << "Game: " << game_name
            << ", num_histories: " << num_histories
            << ", num_terminal_histories: " << num_terminal_histories
            << ", num_chance_nodes: " << num_chance_nodes
            << ", num_nonterminal_states: " << num_nonterminal_states
            << ", num_terminal_states: " << num_terminal_states
            << std::endl;
}
