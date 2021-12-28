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

#include <memory>
#include <random>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(int, players, 0, "How many players in this game, 0 for default.");
ABSL_FLAG(bool, show_infostate, false, "Show the information state.");
ABSL_FLAG(int, seed, 0, "Seed for the random number generator. 0 for auto.");
ABSL_FLAG(bool, show_legals, false, "Show the legal moves.");

void PrintLegalActions(const open_spiel::State& state,
                       open_spiel::Player player,
                       const std::vector<open_spiel::Action>& movelist) {
  std::cerr << "Legal moves for player " << player << ":" << std::endl;
  for (open_spiel::Action action : movelist) {
    std::cerr << "  " << state.ActionToString(player, action) << std::endl;
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string game_name = absl::GetFlag(FLAGS_game);
  auto players = absl::GetFlag(FLAGS_players);
  bool show_infostate = absl::GetFlag(FLAGS_show_infostate);
  int seed = absl::GetFlag(FLAGS_seed);
  bool show_legals = absl::GetFlag(FLAGS_show_legals);

  // Print out registered games.
  std::cerr << "Registered games:" << std::endl;
  std::vector<std::string> names = open_spiel::RegisteredGames();
  for (const std::string& name : names) {
    std::cerr << name << std::endl;
  }

  // Random number generator.
  std::mt19937 rng(seed ? seed : time(0));

  // Create the game.
  std::cerr << "Creating game..\n" << std::endl;

  // Add any specified parameters to override the defaults.
  open_spiel::GameParameters params;
  if (players > 0) {
    params["players"] = open_spiel::GameParameter(players);
  }
  std::shared_ptr<const open_spiel::Game> game =
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
      open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else if (state->IsSimultaneousNode()) {
      // open_spiel::Players choose simultaneously?
      std::vector<open_spiel::Action> joint_action;
      std::vector<float> infostate(game->InformationStateTensorSize());

      // Sample a action for each player
      for (auto player = open_spiel::Player{0}; player < game->NumPlayers();
           ++player) {
        if (show_infostate) {
          if (game->GetType().provides_information_state_tensor) {
            state->InformationStateTensor(player, absl::MakeSpan(infostate));
            std::cerr << "player " << player << ": "
                      << absl::StrJoin(infostate, " ") << std::endl;
          }
          if (game->GetType().provides_information_state_string) {
            std::cerr << "player " << player << ": "
                      << state->InformationStateString(player) << std::endl;
          }
        }

        std::vector<open_spiel::Action> actions = state->LegalActions(player);
        if (show_legals) {
          PrintLegalActions(*state, player, actions);
        }

        open_spiel::Action action = 0;
        if (!actions.empty()){
          absl::uniform_int_distribution<> dis(0, actions.size() - 1);
          action = actions[dis(rng)];
        }
        joint_action.push_back(action);
        std::cerr << "player " << player << " chose "
                  << state->ActionToString(player, action) << std::endl;
      }

      state->ApplyActions(joint_action);
    } else {
      // Decision node, sample one uniformly.
      auto player = state->CurrentPlayer();
      if (show_infostate) {
        if (game->GetType().provides_information_state_tensor) {
          std::vector<float> infostate;
          state->InformationStateTensor(player, absl::MakeSpan(infostate));
          std::cerr << "player " << player << ": "
                    << absl::StrJoin(infostate, " ") << std::endl;
        }
        if (game->GetType().provides_information_state_string) {
          std::cerr << "player " << player << ": "
                    << state->InformationStateString(player) << std::endl;
        }
      }

      std::vector<open_spiel::Action> actions = state->LegalActions();
      if (show_legals) {
        PrintLegalActions(*state, player, actions);
      }

      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      auto action = actions[dis(rng)];
      std::cerr << "chose action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    }

    std::cerr << "State: " << std::endl << state->ToString() << std::endl;
  }

  auto returns = state->Returns();
  for (auto p = open_spiel::Player{0}; p < game->NumPlayers(); p++) {
    std::cerr << "Final return to player " << p << " is " << returns[p]
              << std::endl;
  }
}
