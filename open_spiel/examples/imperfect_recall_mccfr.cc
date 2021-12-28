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

#include <ostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/tabular_best_response_mdp.h"
#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/games/phantom_ttt.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// E.g. another choice: dark_hex_ir(board_size=2)
ABSL_FLAG(std::string, game, "liars_dice_ir", "Game string");
ABSL_FLAG(int, num_iters, 1000000, "How many iters to run for.");
ABSL_FLAG(int, report_every, 1000, "How often to report.");

namespace open_spiel {
namespace {

using algorithms::TabularBestResponseMDP;
using algorithms::TabularBestResponseMDPInfo;

void ImperfectRecallMCCFR() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  // algorithms::ExternalSamplingMCCFRSolver solver(*game);
  algorithms::OutcomeSamplingMCCFRSolver solver(*game);

  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    solver.RunIteration();

    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      // Must use tabular best response MDP as it supports imperfect recall
      // games.
      std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
      TabularBestResponseMDP tbr(*game, *average_policy);
      TabularBestResponseMDPInfo br_info = tbr.NashConv();
      std::cout << i << " " << br_info.nash_conv << std::endl;
    }
  }
}
}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::ImperfectRecallMCCFR();
}
