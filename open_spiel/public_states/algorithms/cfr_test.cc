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

#include "open_spiel/public_states/algorithms/cfr.h"

#include <cmath>
#include <iostream>

#include "open_spiel/public_states/public_states.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/public_states/games/kuhn_poker.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace public_states {
namespace algorithms {
namespace {

void CheckNashKuhnPoker(const Game& game, const Policy& policy) {
  const std::vector<double> game_value =
      ::open_spiel::algorithms::ExpectedReturns(
          *game.NewInitialState(), policy, /*depth_limit=*/-1,
          /*use_infostate_get_policy=*/false);

  // 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
  constexpr double nash_value = 1.0 / 18.0;
  constexpr double eps = 1e-3;

  SPIEL_CHECK_EQ(2, game_value.size());
  SPIEL_CHECK_FLOAT_NEAR(game_value[0], -nash_value, eps);
  SPIEL_CHECK_FLOAT_NEAR(game_value[1], nash_value, eps);
}

void CheckExploitabilityKuhnPoker(const Game& game, const Policy& policy) {
  SPIEL_CHECK_LE(::open_spiel::algorithms::Exploitability(game, policy), 0.05);
}

void CFRTest_KuhnPoker() {
  std::shared_ptr<const GameWithPublicStates> game =
      LoadGameWithPublicStates("kuhn_poker");
  CFRPublicStatesSolver solver(*game);
  for (int i = 0; i < 300; i++) {
    solver.RunIteration();
  }

  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  CheckNashKuhnPoker(*game->GetBaseGame(), *average_policy);
  CheckExploitabilityKuhnPoker(*game->GetBaseGame(), *average_policy);
}


void CFRPlusTest_KuhnPoker() {
  std::shared_ptr<const GameWithPublicStates> game =
      LoadGameWithPublicStates("kuhn_poker");
  CFRPlusPublicStatesSolver solver(*game);
  for (int i = 0; i < 200; i++) {
    solver.RunIteration();
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  CheckNashKuhnPoker(*game->GetBaseGame(), *average_policy);
  CheckExploitabilityKuhnPoker(*game->GetBaseGame(), *average_policy);
}

// TODO(author13): Implement terminal values for multi-player kuhn and enable
//              add multi-player CFR tests.


}  // namespace
}  // namespace algorithms
}  // namespace public_states
}  // namespace open_spiel

namespace algorithms = open_spiel::public_states::algorithms;

int main(int argc, char** argv) {
  algorithms::CFRTest_KuhnPoker();
  algorithms::CFRPlusTest_KuhnPoker();
}
