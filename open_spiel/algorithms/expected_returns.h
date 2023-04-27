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

#ifndef OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_
#define OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_

#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Computes the (undiscounted) expected returns from a depth-limited search
// starting at the state and following each player's policy. Using a negative
// depth will do a full tree traversal (from the specified state). Using a
// prob_cut_threshold > 0 will cut the tree search if the reach probability
// goes below this value resulting in an approximate return.
//
// Policies need not be complete; any missing legal actions will be assumed to
// have zero probability.
//
// The second overloaded function acts the same way, except assumes that all of
// the players' policies are encapsulated in one joint policy.
//
// The `use_infostate_get_policy` flag indicates whether to call
// Policy::GetStatePolicy(const std::string&) rather than
// Policy::GetStatePolicy(const State&) instead for retrieving the policy at
// each information state; we use a default of true for performance reasons.
std::vector<double> ExpectedReturns(const State& state,
                                    const std::vector<const Policy*>& policies,
                                    int depth_limit,
                                    bool use_infostate_get_policy = true,
                                    float prob_cut_threshold = 0.0);
std::vector<double> ExpectedReturns(const State& state,
                                    const Policy& joint_policy, int depth_limit,
                                    bool use_infostate_get_policy = true,
                                    float prob_cut_threshold = 0.0);

// Computes the (undiscounted) expected returns from random deterministic
// policies which are specified using a seed. There should be a policy_seed per
// player. Optionally any number of policies can be provided which override
// the random deterministic policies.
//
// A deterministic policy is one that places all probability mass on a single
// action at each information state. We randomly generate a deterministic
// policy from a seed as follows:
// * Specify a policy seed for each player.
// * For each information state visited:
//   - Calculate an integer hash of the information state string.
//   - Add the move number.
//   - Add the global seed of the corresponding player.
//   - This results in a new seed per information state.
//   - Using this seed, sample an action from a uniform integer distribution.
//
// This means that an entire policy can be represented cheaply with a single
// integer and allows computing expected returns of games whose tabular policies
// may not fit in memory.
std::vector<double> ExpectedReturnsOfDeterministicPoliciesFromSeeds(
  const State& state, const std::vector<int> & policy_seeds);
std::vector<double> ExpectedReturnsOfDeterministicPoliciesFromSeeds(
  const State& state, const std::vector<int> & policy_seeds,
  const std::vector<const Policy*>& policies);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_EXPECTED_RETURNS_H_
