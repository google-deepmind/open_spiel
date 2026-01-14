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

#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/utils/file.h"

ABSL_FLAG(std::string, game, "kuhn_poker(players=3)", "Game to run CFR on.");
ABSL_FLAG(std::string, file_prefix, "/tmp", "Path prefix for file writing.");
ABSL_FLAG(int, seed, 39827891, "Seed to use for randomization.");
ABSL_FLAG(int, repeats, 3, "How many iters to run for.");
ABSL_FLAG(int, num_cfr_iters, 1000, "How many iters of CFR to run for.");
ABSL_FLAG(int, num_cfros_iters, 100000,
          "How many iters of Outcome Sample MCCFR to run for.");

// This example check to see how different the approximate equilibria CFR find
// are, based on random initial regrets and Monte Carlo sampling.

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));
  absl::uniform_int_distribution<int> dist;
  std::string file_prefix = absl::GetFlag(FLAGS_file_prefix);

  // Random initial regrets
  for (int i = 0; i < absl::GetFlag(FLAGS_repeats); ++i) {
    std::string filename = absl::StrCat(file_prefix, "/cfr_rir_", i, ".txt");
    std::cout << "Random initial regrets, repeat number " << i
              << ", generating " << filename << std::endl;
    int seed = dist(rng);
    open_spiel::algorithms::CFRSolverBase solver(*game,
      /*alternating_updates*/true, /*linear_averaging*/false,
      /*regret_matching_plus*/false, /*random_initial_regrets*/true,
      /*seed*/seed);

    for (int i = 0; i < absl::GetFlag(FLAGS_num_cfr_iters); ++i) {
      solver.EvaluateAndUpdatePolicy();
    }
    open_spiel::TabularPolicy avg_policy = solver.TabularAveragePolicy();
    open_spiel::file::File outfile(filename, "w");
    outfile.Write(avg_policy.ToStringSorted());
  }

  // Outcome Sampling MCCFR
  for (int i = 0; i < absl::GetFlag(FLAGS_repeats); ++i) {
    std::string filename = absl::StrCat(file_prefix, "/cfr_cfros_", i, ".txt");
    std::cout << "Outcome Sampling MCCFR, repeat number " << i
              << ", generating " << filename << std::endl;
    int seed = dist(rng);
    open_spiel::algorithms::OutcomeSamplingMCCFRSolver solver(*game,
        /*epsilon*/0.6, seed);

    for (int i = 0; i < absl::GetFlag(FLAGS_num_cfros_iters); ++i) {
      solver.RunIteration();
    }
    open_spiel::TabularPolicy avg_policy = solver.TabularAveragePolicy();
    open_spiel::file::File outfile(filename, "w");
    outfile.Write(avg_policy.ToStringSorted());
  }
}
