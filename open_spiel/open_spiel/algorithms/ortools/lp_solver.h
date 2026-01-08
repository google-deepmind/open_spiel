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

#ifndef OPEN_SPIEL_ALGORITHMS_ORTOOLS_LP_SOLVER_H_
#define OPEN_SPIEL_ALGORITHMS_ORTOOLS_LP_SOLVER_H_

#include <vector>

#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/matrix_game.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

struct ZeroSumGameSolution {
  std::vector<double> values;
  std::vector<std::vector<double>> strategies;
};

enum class CorrEqObjType {
  kAny,
  kSocialWelfareMax,
  kSocialWelfareAtLeast,
};

ZeroSumGameSolution SolveZeroSumMatrixGame(
    const matrix_game::MatrixGame& matrix_game);

NormalFormCorrelationDevice ComputeCorrelatedEquilibrium(
    const NormalFormGame& normal_form_game, CorrEqObjType obj_type,
    double social_welfare_lower_bound);

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ORTOOLS_LP_SOLVER_H_
