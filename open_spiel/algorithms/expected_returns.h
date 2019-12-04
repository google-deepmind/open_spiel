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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_

#include <string>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Computes the (undiscounted) expected returns from a depth-limited search
// starting at the state and following each player's policy. Using a negative
// depth will do a full tree traversal (from the specified state).
//
// The second overloaded function acts the same way, except assumes that all of
// the players' policies are encapsulated in one joint policy.
// `provides_infostate` should be set to true if the Policy* objects passed in
// have implemented the GetStatePolicy(const std::string&) method, as this
// allows for additional optimizations. Otherwise, GetStatePolicy(const State&)
// will be called.
std::vector<double> ExpectedReturns(const State& state,
                                    const std::vector<const Policy*>& policies,
                                    int depth_limit,
                                    bool provides_infostate = true);
std::vector<double> ExpectedReturns(const State& state,
                                    const Policy& joint_policy,
                                    int depth_limit);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_
