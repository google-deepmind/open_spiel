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

#include "open_spiel/algorithms/ortools/lp_solver.h"

#include <memory>
#include <numeric>

#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

constexpr double kErrorTolerance = 1e-10;

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

void TestCorrelatedEquilibrium() {
  // Wikipedia example:
  // https://en.wikipedia.org/wiki/Correlated_equilibrium#An_example
  std::shared_ptr<const matrix_game::MatrixGame> chicken_dare =
      matrix_game::CreateMatrixGame({{0, 7}, {2, 6}}, {{0, 2}, {7, 6}});
  NormalFormCorrelationDevice mu =
      ComputeCorrelatedEquilibrium(*chicken_dare, CorrEqObjType::kAny, 0.0);
  for (const auto &item : mu) {
    std::cout << item.probability << " " << absl::StrJoin(item.actions, " ")
              << std::endl;
  }
  std::cout << std::endl;

  // There is a CE with 1/3 (C,C), 1/3 (D,C), and 1/3 (C,D).
  mu = ComputeCorrelatedEquilibrium(*chicken_dare,
                                    CorrEqObjType::kSocialWelfareAtLeast, 10.0);
  for (const auto &item : mu) {
    std::cout << item.probability << " " << absl::StrJoin(item.actions, " ")
              << std::endl;
  }
  std::cout << std::endl;

  std::vector<double> expected_values = ExpectedValues(*chicken_dare, mu);
  double social_welfare =
      std::accumulate(expected_values.begin(), expected_values.end(), 0.0);
  std::cout << social_welfare << std::endl;
  SPIEL_CHECK_GE(social_welfare, 10.0 - kErrorTolerance);

  // There is a better one that gets 10.5: 1/4 (C,D), 1/4 (D,C), 1/2 (C, C)
  mu = ComputeCorrelatedEquilibrium(*chicken_dare,
                                    CorrEqObjType::kSocialWelfareMax, 0);
  for (const auto &item : mu) {
    std::cout << item.probability << " " << absl::StrJoin(item.actions, " ")
              << std::endl;
  }
  std::cout << std::endl;

  expected_values = ExpectedValues(*chicken_dare, mu);
  social_welfare =
      std::accumulate(expected_values.begin(), expected_values.end(), 0.0);
  std::cout << social_welfare << std::endl;
  SPIEL_CHECK_FLOAT_NEAR(social_welfare, 10.5, kErrorTolerance);
  for (const auto &item : mu) {
    if (item.actions[0] + item.actions[1] == 1) {
      SPIEL_CHECK_FLOAT_NEAR(item.probability, 1.0 / 4.0, kErrorTolerance);
    } else if (item.actions[0] + item.actions[1] == 2) {
      SPIEL_CHECK_FLOAT_NEAR(item.probability, 1.0 / 2.0, kErrorTolerance);
    }
  }
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char **argv) {
  algorithms::ortools::TestSolveMatrixGame();
  algorithms::ortools::TestCorrelatedEquilibrium();
}
