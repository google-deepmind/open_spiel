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

#ifndef OPEN_SPIEL_UTILS_SOFT_CONDORCET_OPTIMIZATION_H_
#define OPEN_SPIEL_UTILS_SOFT_CONDORCET_OPTIMIZATION_H_

// A C++ implementation of Soft Condorcet optimizer (see Lanctot et al. '24.
// https://arxiv.org/abs/2411.00119). This is functionally equivalent to the
// Python implementation in python/voting/soft_condorcet_optimization.py but
// runs faster.

#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"

namespace open_spiel {
namespace evaluation {

using WeightedVotePair = std::pair<int, std::vector<std::string>>;
using TupleListVote = std::vector<WeightedVotePair>;

struct WeightedVote {
  int weight;
  std::vector<std::string> vote;
};

class PreferenceProfile {
 public:
  PreferenceProfile() = default;
  void AddVote(const WeightedVote& vote);
  void AddVote(const std::vector<std::string>& vote, int weight = 1);
  void AddVoteUngrouped(const std::vector<std::string>& vote, int weight = 1);
  int RegisterAlternative(const std::string& alternative);
  const std::vector<WeightedVote>& votes() const { return votes_; }
  const std::vector<std::string>& alternatives() const { return alternatives_; }
  std::string get_alternative(int idx) const { return alternatives_[idx]; }
  int num_votes() const { return votes_.size(); }
  int num_alternatives() const { return alternatives_.size(); }
  const absl::flat_hash_map<std::string, int>& alternatives_dict() const {
    return alternatives_dict_;
  }

 private:
  // Alternative name -> index map.
  std::vector<std::string> alternatives_;
  absl::flat_hash_map<std::string, int> alternatives_dict_;
  std::vector<WeightedVote> votes_;
};

class Optimizer {
 public:
  Optimizer(const TupleListVote& votes, double rating_lower_bound,
            double rating_upper_bound, int batch_size, int rng_seed = 0,
            int compute_norm_freq = 1000, double initial_param_noise = 0.0,
            const std::vector<std::string>& alternative_names = {});
  virtual ~Optimizer() = default;

  void Step(double learning_rate, const std::vector<int>& batch);
  void RunSolver(int iterations = 1000, double learning_rate = 0.01);
  std::map<std::string, double> ratings() const;

 protected:
  virtual void ComputeGradient(const std::vector<int>& batch) = 0;
  void divide_gradient();

  PreferenceProfile profile_;
  std::mt19937_64 rng_;
  double rating_lower_bound_;
  double rating_upper_bound_;
  int batch_size_;
  int compute_norm_freq_;
  double initial_param_noise_;
  int total_iterations_;
  int num_alternatives_;
  std::vector<double> ratings_;
  std::vector<double> gradient_;
};

class SoftCondorcetOptimizer : public Optimizer {
 public:
  SoftCondorcetOptimizer(
      const TupleListVote& votes, double rating_lower_bound,
      double rating_upper_bound, int batch_size, double temperature = 1.0,
      int rng_seed = 0, int compute_norm_freq = 1000,
      double initial_param_noise = 0,
      const std::vector<std::string>& alternative_names = {});

  virtual ~SoftCondorcetOptimizer() = default;

  void ComputeGradient(const std::vector<int>& batch) override;

 private:
  double temperature_ = 1.0;
};

class FenchelYoungOptimizer : public Optimizer {
 public:
  FenchelYoungOptimizer(const TupleListVote& votes, double rating_lower_bound,
                        double rating_upper_bound, int batch_size,
                        int rng_seed = 0, int compute_norm_freq = 1000,
                        double initial_param_noise = 0, double sigma = 100.0,
                        const std::vector<std::string>& alternative_names = {});
  virtual ~FenchelYoungOptimizer() = default;

  void ComputeGradient(const std::vector<int>& batch) override;

 private:
  double sigma_;
  std::extreme_value_distribution<> gumbel_dist_;
};

}  // namespace evaluation
}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_SOFT_CONDORCET_OPTIMIZATION_H_
