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

// This is a simple example that is used to demonstrate how to use OpenSpiel
// when it is built as a shared library. To use OpenSpiel as a library,
// see: https://github.com/deepmind/open_spiel/blob/master/docs/library.md

#include <memory>
#include <random>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

void PrintLegalActions(const open_spiel::State& state,
                       open_spiel::Player player,
                       const std::vector<open_spiel::Action>& movelist) {
  std::cerr << "Legal moves for player " << player << ":" << std::endl;
  for (open_spiel::Action action : movelist) {
    std::cerr << "  " << state.ActionToString(player, action) << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage shared_library_example <game string>" << std::endl;
    return -1;
  }

  // Print out registered games.
  std::cerr << "Registered games:" << std::endl;
  std::vector<std::string> names = open_spiel::RegisteredGames();
  for (const std::string& name : names) {
    std::cerr << name << std::endl;
  }

  // Random number generator.
  std::mt19937 rng(time(0));

  // Load the game.
  std::cerr << "Loading game..\n" << std::endl;
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(argv[1]);

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
        std::vector<open_spiel::Action> actions = state->LegalActions(player);
        PrintLegalActions(*state, player, actions);

        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        open_spiel::Action action = actions[dis(rng)];
        joint_action.push_back(action);
        std::cerr << "player " << player << " chose "
                  << state->ActionToString(player, action) << std::endl;
      }

      state->ApplyActions(joint_action);
    } else {
      // Decision node, sample one uniformly.
      auto player = state->CurrentPlayer();
      std::vector<open_spiel::Action> actions = state->LegalActions();
      PrintLegalActions(*state, player, actions);

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
