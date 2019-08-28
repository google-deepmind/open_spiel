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

#include <unistd.h>

#include <memory>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

const char* kUsageStr =
    "example --game=<shortname> [--players=<num>] "
    "[--show_infostate] [--seed=<num>] [--show_legals=<true/false>]";

void PrintLegalActions(const open_spiel::State& state, int player,
                       const std::vector<open_spiel::Action>& movelist) {
  std::cerr << "Legal moves for player " << player << ":" << std::endl;
  for (open_spiel::Action action : movelist) {
    std::cerr << "  " << state.ActionToString(player, action) << std::endl;
  }
}

int main(int argc, char** argv) {
  std::string game_name =
      open_spiel::ParseCmdLineArgDefault(argc, argv, "game", "");
  int players =
      std::stoi(open_spiel::ParseCmdLineArgDefault(argc, argv, "players", "0"));
  bool show_infostate = open_spiel::ParseCmdLineArgDefault(
                            argc, argv, "show_infostate", "false") == "true";
  std::pair<bool, std::string> seed =
      open_spiel::ParseCmdLineArg(argc, argv, "seed");
  bool show_legals = open_spiel::ParseCmdLineArgDefault(
                         argc, argv, "show_legals", "false") == "true";

  // Print out registered games.
  std::cerr << "Registered games:" << std::endl;
  std::vector<std::string> names = open_spiel::RegisteredGames();
  for (const std::string& name : names) {
    std::cerr << name << std::endl;
  }

  if (game_name.empty()) {
    std::cerr << kUsageStr << std::endl;
    return -1;
  }

  // Random number generator.
  std::mt19937 rng(seed.first ? std::stol(seed.second) : time(0));

  // Create the game.
  std::cerr << "Creating game..\n" << std::endl;

  // Add any specified parameters to override the defaults.
  open_spiel::GameParameters params;
  if (players > 0) {
    params["players"] = open_spiel::GameParameter(players);
  }
  std::unique_ptr<open_spiel::Game> game =
      open_spiel::LoadGame(game_name, params);

  if (!game) {
    std::cerr << "problem with loading game, exiting..." << std::endl;
    return -1;
  }

  std::cerr << "Starting new game..." << std::endl;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  std::cerr << "Initial state:" << std::endl;
  std::cerr << "State:" << std::endl << state->ToString() << std::endl;

  while (!state->IsTerminal()) {
    std::cerr << "player " << state->CurrentPlayer() << std::endl;

    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      std::vector<std::pair<open_spiel::Action, double>> outcomes =
          state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleChanceOutcome(
          outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(rng));
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else if (state->IsSimultaneousNode()) {
      // Players choose simultaneously?
      std::vector<open_spiel::Action> joint_action;
      std::vector<double> infostate;

      // Sample a action for each player
      for (int player = 0; player < game->NumPlayers(); ++player) {
        if (show_infostate) {
          if (game->GetType().provides_information_state_as_normalized_vector) {
            state->InformationStateAsNormalizedVector(player, &infostate);
            std::cerr << "player " << player << ": "
                      << absl::StrJoin(infostate, " ") << std::endl;
          }
          if (game->GetType().provides_information_state) {
            std::cerr << "player " << player << ": "
                      << state->InformationState(player) << std::endl;
          }
        }

        std::vector<open_spiel::Action> actions = state->LegalActions(player);
        if (show_legals) {
          PrintLegalActions(*state, player, actions);
        }

        std::uniform_int_distribution<> dis(0, actions.size() - 1);
        open_spiel::Action action = actions[dis(rng)];
        joint_action.push_back(action);
        std::cerr << "player " << player << " chose "
                  << state->ActionToString(player, action) << std::endl;
      }

      state->ApplyActions(joint_action);
    } else {
      // Decision node, sample one uniformly.
      int player = state->CurrentPlayer();
      if (show_infostate) {
        if (game->GetType().provides_information_state_as_normalized_vector) {
          std::vector<double> infostate;
          state->InformationStateAsNormalizedVector(player, &infostate);
          std::cerr << "player " << player << ": "
                    << absl::StrJoin(infostate, " ") << std::endl;
        }
        if (game->GetType().provides_information_state) {
          std::cerr << "player " << player << ": "
                    << state->InformationState(player) << std::endl;
        }
      }

      std::vector<open_spiel::Action> actions = state->LegalActions();
      if (show_legals) {
        PrintLegalActions(*state, player, actions);
      }

      std::uniform_int_distribution<> dis(0, actions.size() - 1);
      auto action = actions[dis(rng)];
      std::cerr << "chose action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    }

    std::cerr << "State: " << std::endl << state->ToString() << std::endl;
  }

  auto returns = state->Returns();
  for (int p = 0; p < game->NumPlayers(); p++) {
    std::cerr << "Final return to player " << p << " is " << returns[p]
              << std::endl;
  }
}
