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

#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

constexpr double kErrorTolerance = 1e-14;

void TestGameValueAndExploitability(const std::string& game_name,
                                    double expected_game_value) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  ZeroSumSequentialGameSolution solution = SolveZeroSumSequentialGame(*game);
  SPIEL_CHECK_FLOAT_NEAR(solution.game_value, expected_game_value,
                         kErrorTolerance);

  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous)
    return;
  SPIEL_CHECK_FLOAT_NEAR(Exploitability(*game, solution.policy), 0.,
                         kErrorTolerance);
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char **argv) {
  algorithms::ortools::TestGameValueAndExploitability("matrix_mp", 0.);
  algorithms::ortools::TestGameValueAndExploitability("kuhn_poker", -1 / 18.);
  algorithms::ortools::TestGameValueAndExploitability(
      "leduc_poker", -0.085606424078);
  algorithms::ortools::TestGameValueAndExploitability(
      "goofspiel(players=2,num_cards=3,imp_info=True)", 0.);
}
