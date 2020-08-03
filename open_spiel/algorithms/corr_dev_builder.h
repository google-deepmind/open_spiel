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

#ifndef OPEN_SPIEL_ALGORITHMS_CORR_DEV_AGGREGATOR_H_
#define OPEN_SPIEL_ALGORITHMS_CORR_DEV_AGGREGATOR_H_

#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/policy.h"

namespace open_spiel {
namespace algorithms {

// A helper class for maintaining and building correlation devices
// (distributions over joint deterministic policies).
//
// This helper exists to serve algorithms interact with the CorrDist functions
// (see corr_dist.h), which require distributions over joint deterministic
// policies. Algorithms like CFR produce stochastic policies, so they have
// to either be converted or sampled before they can be evaluated by the
// CorrDist functions.
//
// This helper class maintains weights over joint determinstic tabular policies,
// updating each as new policies are added. A correlation device is obtained by
// normalizing the weights over all the deterministic policies being tracked.
class CorrDevBuilder {
 public:
  CorrDevBuilder(int seed = 0);

  // Add a joint policy with the specified weight.
  void AddDeterminsticJointPolicy(const TabularPolicy& policy,
                                  double weight = 1.0);

  // Take a number of sampled joint policies and add each one with a weight
  // of 1.0 / num_samples.
  void AddSampledJointPolicy(const TabularPolicy& policy, int num_samples);

  // TODO(author5): add a function here that applies NF-Reconstruct-Strategy
  // from Celli et al. '19 (https://arxiv.org/abs/1910.06228) on each player and
  // adds the resulting distribution over joint policies.

  // Return the correlation device represented by this builder.
  CorrelationDevice GetCorrelationDevice() const;

 private:
  std::mt19937 rng_;
  double total_weight_;

  // Each of these uses keys that have a canonical stringified policy as the
  // key (e.g. complete policies with sorted keys).
  absl::flat_hash_map<std::string, double> policy_weights_;
  absl::flat_hash_map<std::string, TabularPolicy> policy_map_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_CORR_DEV_AGGREGATOR_H_
