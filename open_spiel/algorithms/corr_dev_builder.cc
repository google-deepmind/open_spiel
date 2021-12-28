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

#include "open_spiel/algorithms/corr_dev_builder.h"

#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

CorrDevBuilder::CorrDevBuilder(int seed) : rng_(seed), total_weight_(0.0) {}

void CorrDevBuilder::AddDeterminsticJointPolicy(const TabularPolicy& policy,
                                                double weight) {
  std::string key = policy.ToStringSorted();
  auto iter = policy_weights_.find(key);
  if (iter == policy_weights_.end()) {
    policy_weights_[key] = weight;
    policy_map_[key] = policy;
  } else {
    iter->second += weight;
  }
  total_weight_ += weight;
}

void CorrDevBuilder::AddSampledJointPolicy(const TabularPolicy& policy,
                                           int num_samples, double weight) {
  for (int sample = 0; sample < num_samples; ++sample) {
    TabularPolicy sampled_policy;
    for (const auto& iter : policy.PolicyTable()) {
      Action sampled_action = SampleAction(iter.second, rng_).first;
      sampled_policy.SetStatePolicy(
          iter.first, ToDeterministicPolicy(iter.second, sampled_action));
    }
    AddDeterminsticJointPolicy(sampled_policy, 1.0 / num_samples * weight);
  }
}

void CorrDevBuilder::AddMixedJointPolicy(const TabularPolicy& policy,
                                         double weight) {
  std::vector<int> action_indices(policy.PolicyTable().size(), 0);
  bool done = false;
  double total_prob = 0.0;

  while (!done) {
    // Construct the joint policy and add it.
    TabularPolicy deterministic_policy;
    double prob = 1.0;
    int info_state_idx = 0;
    for (const auto& iter : policy.PolicyTable()) {
      Action action = iter.second[action_indices[info_state_idx]].first;
      prob *= GetProb(iter.second, action);
      if (prob == 0.0) {
        break;
      }
      deterministic_policy.SetStatePolicy(
          iter.first, ToDeterministicPolicy(iter.second, action));
      info_state_idx++;
    }

    SPIEL_CHECK_PROB(prob);
    if (prob > 0.0) {
      AddDeterminsticJointPolicy(deterministic_policy, prob * weight);
      total_prob += prob;
    }

    // Now, try to move to the next joint policy.
    info_state_idx = 0;
    done = true;
    for (const auto& iter : policy.PolicyTable()) {
      if (++action_indices[info_state_idx] < iter.second.size()) {
        done = false;
        break;
      } else {
        action_indices[info_state_idx] = 0;
      }
      info_state_idx++;
    }
  }

  SPIEL_CHECK_TRUE(Near(total_prob, 1.0, 1e-10));
}

CorrelationDevice CorrDevBuilder::GetCorrelationDevice() const {
  SPIEL_CHECK_GT(total_weight_, 0);
  CorrelationDevice corr_dev;
  double sum_weight = 0;
  for (const auto& key_and_policy : policy_map_) {
    double weight = policy_weights_.at(key_and_policy.first);
    sum_weight += weight;
    corr_dev.push_back({weight / total_weight_, key_and_policy.second});
  }
  SPIEL_CHECK_TRUE(Near(sum_weight, total_weight_));
  return corr_dev;
}

CorrelationDevice SampledDeterminizeCorrDev(const CorrelationDevice& corr_dev,
                                            int num_samples_per_policy) {
  CorrDevBuilder cdb;
  for (const std::pair<double, TabularPolicy>& item : corr_dev) {
    cdb.AddSampledJointPolicy(item.second, num_samples_per_policy, item.first);
  }
  return cdb.GetCorrelationDevice();
}

CorrelationDevice DeterminizeCorrDev(const CorrelationDevice& corr_dev) {
  CorrDevBuilder cdb;
  for (const std::pair<double, TabularPolicy>& item : corr_dev) {
    cdb.AddMixedJointPolicy(item.second, item.first);
  }
  return cdb.GetCorrelationDevice();
}

}  // namespace algorithms
}  // namespace open_spiel
