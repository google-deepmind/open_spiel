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
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(int, sims, 1000, "How many simulations to run.");
ABSL_FLAG(int, attempts, 5, "How many sets of simulations to run.");
ABSL_FLAG(bool, verbose, false,
          "Boolean flag indicating whether to print all simulation info.");

namespace open_spiel {

int RandomSimulation(std::mt19937* rng, const Game& game, bool verbose) {
  std::unique_ptr<State> state = game.NewInitialState();

  if (verbose) {
    std::cout << "Initial state:" << std::endl
              << "State:" << std::endl
              << state->ToString() << std::endl;
  }

  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;
  std::vector<float> obs;
  if (provides_info_state_tensor) {
    obs = std::vector<float>(game.InformationStateTensorSize());
  } else if (provides_observations_tensor) {
    obs = std::vector<float>(game.ObservationTensorSize());
  }

  int game_length = 0;
  while (!state->IsTerminal()) {
    if (provides_info_state_tensor && state->CurrentPlayer() >= 0) {
      state->InformationStateTensor(state->CurrentPlayer(),
                                    absl::MakeSpan(obs));
    } else if (provides_observations_tensor && state->CurrentPlayer() >= 0) {
      state->ObservationTensor(state->CurrentPlayer(), absl::MakeSpan(obs));
    }
    ++game_length;
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action;
      if (game.GetType().chance_mode ==
          GameType::ChanceMode::kSampledStochastic) {
        action = outcomes.front().first;
      } else {
        // Explicit chance node; sample one according to underlying
        // distribution.
        action = SampleAction(outcomes, *rng).first;
      }
      if (verbose) {
        std::cout << "Sampled outcome: "
                  << state->ActionToString(kChancePlayerId, action)
                  << std::endl;
      }
      state->ApplyAction(action);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      // Sample an action for each player
      std::vector<Action> joint_action;
      for (int p = 0; p < game.NumPlayers(); p++) {
        std::vector<Action> actions;
        actions = state->LegalActions(p);
        Action action = 0;
        if (!actions.empty()) {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
        joint_action.push_back(action);
        if (verbose) {
          std::cout << "Player " << p
                    << " chose action:" << state->ActionToString(p, action)
                    << std::endl;
        }
      }
      state->ApplyActions(joint_action);
    } else {
      // Sample an action uniformly.
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];
      if (verbose) {
        int p = state->CurrentPlayer();
        std::cout << "Player " << p
                  << " chose action: " << state->ActionToString(p, action)
                  << std::endl;
      }
      state->ApplyAction(action);
    }
    if (verbose) {
      std::cout << "State: " << std::endl << state->ToString() << std::endl;
      std::cout << "Observation: " << obs << std::endl;
    }
  }
  return game_length;
}

// Perform num_sims random simulations of the specified game, and output the
// time taken.
void RandomSimBenchmark(const std::string& game_def, int num_sims,
                        bool verbose) {
  std::mt19937 rng;
  std::cout << absl::StrFormat("Benchmark: game: %s, num_sims: %d. ", game_def,
                               num_sims);

  auto game = LoadGame(game_def);

  absl::Time start = absl::Now();
  int num_moves = 0;
  for (int sim = 0; sim < num_sims; ++sim) {
    num_moves += RandomSimulation(&rng, *game, verbose);
  }
  absl::Time end = absl::Now();
  double seconds = absl::ToDoubleSeconds(end - start);

  std::cout << absl::StrFormat(
                   "Finished %d moves in %.1f ms: %.1f sim/s, %.1f moves/s",
                   num_moves, seconds * 1000, num_sims / seconds,
                   num_moves / seconds)
            << std::endl;
}

}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  for (int i = 0; i < absl::GetFlag(FLAGS_attempts); ++i) {
    open_spiel::RandomSimBenchmark(absl::GetFlag(FLAGS_game),
                                   absl::GetFlag(FLAGS_sims),
                                   absl::GetFlag(FLAGS_verbose));
  }
}
