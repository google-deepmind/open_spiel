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
  SequenceFormLpSpecification specification(*game);
  specification.SpecifyLinearProgram(0);
  double actual_game_value = specification.Solve();
  SPIEL_CHECK_FLOAT_NEAR(actual_game_value, expected_game_value,
                         kErrorTolerance);

  // Compute policy for the opponent.
  TabularPolicy policy0 = specification.OptimalPolicy(0);
  specification.SpecifyLinearProgram(1);
  double opponent_game_value = specification.Solve();
  SPIEL_CHECK_FLOAT_NEAR(actual_game_value + opponent_game_value, 0.,
                         kErrorTolerance);
  TabularPolicy policy1 = specification.OptimalPolicy(1);

  // Test exploitability -- this is implemented only for turn-based games.
  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) return;

  // Merge the two tables.
  std::unordered_map<std::string, ActionsAndProbs> profile_table =
      policy0.PolicyTable();
  profile_table.insert(policy1.PolicyTable().begin(),
                       policy1.PolicyTable().end());
  TabularPolicy optimal_profile(profile_table);
  SPIEL_CHECK_FLOAT_NEAR(Exploitability(*game, optimal_profile), 0.,
                         kErrorTolerance);
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::ortools::TestGameValueAndExploitability("matrix_mp", 0.);
  algorithms::ortools::TestGameValueAndExploitability("kuhn_poker", -1 / 18.);
  algorithms::ortools::TestGameValueAndExploitability("leduc_poker",
                                                      -0.085606424078);
  algorithms::ortools::TestGameValueAndExploitability(
      "goofspiel(players=2,num_cards=3,imp_info=True)", 0.);
}
