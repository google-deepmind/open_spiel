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

#ifndef OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_
#define OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_

#include <vector>

#include "open_spiel/algorithms/infostate_tree.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

struct ZeroSumSequentialGameSolution {
  double game_value;
  TabularPolicy policy;
  std::array<std::vector<double>, 2> root_cfvs;
};

ZeroSumSequentialGameSolution SolveZeroSumSequentialGame(const Game& game);

ZeroSumSequentialGameSolution SolveZeroSumSequentialGame(
    std::shared_ptr<Observer> infostate_observer,
    absl::Span<const State*> starting_states,
    std::array<absl::Span<const float>, 2> player_ranges,
    absl::Span<const float> chance_range,
    std::optional<int> solve_only_player = {},
    bool collect_tabular_policy = true,
    bool collect_root_cfvs = false);



}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_
