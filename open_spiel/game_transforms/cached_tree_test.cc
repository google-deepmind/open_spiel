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

#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace cached_tree {
namespace {

void BasicTests() {
  testing::LoadGameTest("cached_tree(game=kuhn_poker())");
  testing::RandomSimTest(*LoadGame("cached_tree(game=kuhn_poker())"), 10);
}

void CFRTest(const Game& game,
             int iterations,
             absl::optional<double> nash_value,
             absl::optional<double> nash_value_eps,
             absl::optional<double> exploitability_upper_bound) {
  std::cout << "Running CFR for " << iterations << " iterations on " <<
      game.ToString() << std::endl;
  algorithms::CFRSolver solver(game);
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();

  const std::vector<double> game_value =
      algorithms::ExpectedReturns(*game.NewInitialState(), *average_policy,
                                  -1);

  if (nash_value.has_value()) {
    SPIEL_CHECK_EQ(2, game_value.size());
    SPIEL_CHECK_FLOAT_NEAR((float)game_value[0], nash_value.value(),
                           nash_value_eps.value());
    SPIEL_CHECK_FLOAT_NEAR((float)game_value[1], -nash_value.value(),
                           nash_value_eps.value());
  }

  if (exploitability_upper_bound.has_value()) {
    double exploitability = algorithms::Exploitability(game, *average_policy);
    std::cout << "Exploitability: " << exploitability << std::endl;
    SPIEL_CHECK_LE(exploitability, exploitability_upper_bound.value());
  }
}

void CFRTest_KuhnPoker() {
  CFRTest(*LoadGame("cached_tree(game=kuhn_poker())"), 300, -1.0 / 18.0, 0.001,
           0.05);
}

void CFRTest_LeducPoker() {
  CFRTest(*LoadGame("cached_tree(game=leduc_poker())"), 300, -0.08, 0.05, 0.1);
}

}  // namespace
}  // namespace cached_tree
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  open_spiel::cached_tree::BasicTests();
  open_spiel::cached_tree::CFRTest_KuhnPoker();
  open_spiel::cached_tree::CFRTest_LeducPoker();
}
