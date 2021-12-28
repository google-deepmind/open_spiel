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

#ifndef OPEN_SPIEL_ALGORITHMS_VALUE_ITERATION_H_
#define OPEN_SPIEL_ALGORITHMS_VALUE_ITERATION_H_

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Value iteration algorithm: solves for the optimal value function of a game.
// The value function is solved with maximum error less than threshold,
// and it considers all states with depth at most depth_limit from the
// initial state (so if depth_limit is 0, only the root is considered).
// If depth limit is negative, all states are considered.
//
// Currently works for sequential 1-player or 2-player zero-sum games,
// with or without chance nodes.

std::map<std::string, double> ValueIteration(const Game& game, int depth_limit,
                                             double threshold);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_VALUE_ITERATION_H_
