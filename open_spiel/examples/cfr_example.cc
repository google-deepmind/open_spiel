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

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  std::unique_ptr<open_spiel::Game> game = open_spiel::LoadGame("kuhn_poker");
  open_spiel::algorithms::CFRPlusSolver solver(*game);
  std::cerr << "Starting CFR and CFR+ on kuhn_poker..." << std::endl;

  constexpr int iterations = 1000;
  for (int i = 0; i < iterations; i++) {
    if (i % 100 == 0) {
      double exploitability = open_spiel::algorithms::Exploitability(
          *game, *solver.AveragePolicy());

      std::cerr << "Iteration " << i << " exploitability=" << exploitability
                << std::endl;
    }
    solver.EvaluateAndUpdatePolicy();
  }
  double exploitability =
      open_spiel::algorithms::Exploitability(*game, *solver.AveragePolicy());

  std::cerr << "Exploitability of " << exploitability << " reached after "
            << iterations << " iterations." << std::endl;
}
