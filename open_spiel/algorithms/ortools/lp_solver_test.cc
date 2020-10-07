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

#include "open_spiel/algorithms/ortools/lp_solver.h"

#include <memory>

#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

void TestSolveMatrixGame() {
  std::shared_ptr<const matrix_game::MatrixGame> rps =
      LoadMatrixGame("matrix_rps");
  ZeroSumGameSolution solution = SolveZeroSumMatrixGame(*rps);
  SPIEL_CHECK_FLOAT_NEAR(solution.values[0], 0, 1e-10);
  SPIEL_CHECK_FLOAT_NEAR(solution.values[1], 0, 1e-10);
  for (Player p : {0, 1}) {
    for (Action a : {0, 1, 2}) {
      SPIEL_CHECK_FLOAT_NEAR(solution.strategies[p][a], 1.0 / 3.0, 1e-10);
    }
  }
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) { algorithms::ortools::TestSolveMatrixGame(); }
