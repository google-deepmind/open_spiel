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

// This file tests whether we can build a shared library that contains all
// the optional dependencies.

#include <iostream>

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
#include "open_spiel/algorithms/ortools/lp_solver.h"
#include "open_spiel/algorithms/matrix_game_utils.h"
#endif  // OPEN_SPIEL_BUILD_WITH_ORTOOLS

namespace {

void TestLinkingWithOpenSpielCore() {
  std::cout << "Running open_spiel_core" << '\n';
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("kuhn_poker");
  SPIEL_CHECK_EQ(game->GetType().short_name, "kuhn_poker");
}

#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
void TestLinkingWithOpenSpielOrtools() {
  std::cout << "Running open_spiel_ortools" << '\n';
  std::shared_ptr<const open_spiel::matrix_game::MatrixGame> game =
      open_spiel::algorithms::LoadMatrixGame("matrix_rps");
  open_spiel::algorithms::ortools::ZeroSumGameSolution solution =
      open_spiel::algorithms::ortools::SolveZeroSumMatrixGame(*game);
  SPIEL_CHECK_FLOAT_NEAR(solution.values[0], 0., 1e-10);
}
#endif  // OPEN_SPIEL_BUILD_WITH_ORTOOLS

}  // namespace

int main() {
  TestLinkingWithOpenSpielCore();
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
  TestLinkingWithOpenSpielOrtools();
#endif
}
