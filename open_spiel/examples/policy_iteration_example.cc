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
#include <string>

#include "open_spiel/algorithms/policy_iteration.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "abseil-cpp/absl/flags/flag.h"
#include "abseil-cpp/absl/flags/parse.h"
// Example code for using policy iteration algorithm to solve tic-tac-toe.
ABSL_FLAG(std::string, game, "mpg", "The name of the game to play.");
ABSL_FLAG(int, depth_limit, -1,
"Depth limit until which to compute value iteration.");
ABSL_FLAG(double, threshold, 0.01,
"Threshold accuracy at which to stop value iteration.");
int main(int argc, char** argv)
{
    absl::ParseCommandLine(argc, argv);

    std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));

  absl::flat_hash_map<std::string, double> solution =
      open_spiel::algorithms::PolicyIteration(*game, absl::GetFlag(FLAGS_depth_limit), absl::GetFlag(FLAGS_threshold));
  for (const auto& kv : solution) {
    std::cerr << "State: " << std::endl
              << kv.first << std::endl
              << "Value: " << kv.second << std::endl;
  }


}
