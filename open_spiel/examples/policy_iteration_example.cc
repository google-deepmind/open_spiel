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
// Example code for using policy iteration algorithm to solve tic-tac-toe.
int main(int argc, char** argv) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("mpg(max_moves=5,max_size=20,generator=gnp,generator_params=20 0.5 -1 1,compact_string=True)");

  absl::flat_hash_map<std::string, double> solution =
      open_spiel::algorithms::PolicyIteration(*game, -1, 0.01);
  for (const auto& kv : solution) {
    std::cerr << "State: " << std::endl
              << kv.first << std::endl
              << "Value: " << kv.second << std::endl;
  }


}
