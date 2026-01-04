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

#include "open_spiel/evaluation/soft_condorcet_optimization.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace evaluation {
namespace {
template <typename T>
std::vector<int> sort_indices(const std::vector<T>& v) {
  std::vector<int> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                   [&v](int i1, int i2) { return v[i1] < v[i2]; });
  return indices;
}
}  // namespace

void PreferenceProfile::AddVote(const WeightedVote& vote) {
  votes_.push_back(vote);
  for (const std::string& alt : vote.vote) {
    RegisterAlternative(alt);
  }
}

void PreferenceProfile::AddVote(const std::vector<std::string>& vote,
                                int weight) {
  AddVote(WeightedVote{weight, vote});
}

void PreferenceProfile::AddVoteUngrouped(const std::vector<std::string>& vote,
                                         int weight) {
  for (int i = 0; i < weight; ++i) {
    AddVote(WeightedVote{1, vote});
  }
}

int PreferenceProfile::RegisterAlternative(const std::string& alternative) {
  auto iter = alternatives_dict_.find(alternative);
  if (iter != alternatives_dict_.end()) {
    return iter->second;
  }
  alternatives_dict_[alternative] = alternatives_.size();
  alternatives_.push_back(alternative);
  return alternatives_.size() - 1;
}

Optimizer::Optimizer(
    const std::vector<std::pair<int, std::vector<std::string>>>& votes,
    double rating_lower_bound, double rating_upper_bound, int batch_size,
    int rng_seed, int compute_norm_freq, double initial_param_noise,
    const std::vector<std::string>& alternative_names)
    : rng_(rng_seed),
      rating_lower_bound_(rating_lower_bound),
      rating_upper_bound_(rating_upper_bound),
      batch_size_(batch_size),
      compute_norm_freq_(compute_norm_freq),
      initial_param_noise_(initial_param_noise),
      total_iterations_(0) {
  SPIEL_CHECK_GT(batch_size_, 0);
  SPIEL_CHECK_GT(rating_upper_bound, rating_lower_bound);

  if (!alternative_names.empty()) {
    for (const std::string& alt : alternative_names) {
      profile_.RegisterAlternative(alt);
    }
  }

  for (const auto& vote : votes) {
    profile_.AddVoteUngrouped(vote.second, vote.first);
  }
  double midpoint_rating =
      (rating_upper_bound - rating_lower_bound) / 2.0 + rating_lower_bound;
  num_alternatives_ = profile_.num_alternatives();
  ratings_.resize(num_alternatives_, midpoint_rating);
  gradient_.resize(num_alternatives_, 0.0);
}

void Optimizer::Step(double learning_rate, const std::vector<int>& batch) {
  ComputeGradient(batch);
  for (int a = 0; a < num_alternatives_; ++a) {
    ratings_[a] -= learning_rate * gradient_[a];
    ratings_[a] =
        std::clamp(ratings_[a], rating_lower_bound_, rating_upper_bound_);
  }
}

void Optimizer::RunSolver(int iterations, double learning_rate) {
  std::vector<int> batch(batch_size_);
  for (int i = 0; i < iterations; ++i) {
    for (int b = 0; b < batch_size_; ++b) {
      batch[b] = absl::Uniform<int>(rng_, 0, profile_.num_votes());
    }
    Step(learning_rate, batch);
    total_iterations_++;
  }
}

void Optimizer::divide_gradient() {
  for (int a = 0; a < num_alternatives_; ++a) {
    gradient_[a] /= batch_size_;
  }
}

std::map<std::string, double> Optimizer::ratings() const {
  std::map<std::string, double> ratings;
  for (int a = 0; a < num_alternatives_; ++a) {
    ratings[profile_.get_alternative(a)] = ratings_[a];
  }
  return ratings;
}

SoftCondorcetOptimizer::SoftCondorcetOptimizer(
    const TupleListVote& votes, double rating_lower_bound,
    double rating_upper_bound, int batch_size, double temperature, int rng_seed,
    int compute_norm_freq, double initial_param_noise,
    const std::vector<std::string>& alternative_names)
    : Optimizer(votes, rating_lower_bound, rating_upper_bound, batch_size,
                rng_seed, compute_norm_freq, initial_param_noise,
                alternative_names),
      temperature_(temperature) {
  SPIEL_CHECK_GT(temperature_, 0);
}

void SoftCondorcetOptimizer::ComputeGradient(const std::vector<int>& batch) {
  std::fill(gradient_.begin(), gradient_.end(), 0.0);
  for (int vote_idx : batch) {
    const WeightedVote& vote = profile_.votes()[vote_idx];
    int vote_len = vote.vote.size();
    double weight = vote.weight;
    for (int i = 0; i < vote_len; ++i) {
      int a_idx = profile_.alternatives_dict().at(vote.vote[i]);
      for (int j = i + 1; j < vote_len; ++j) {
        int b_idx = profile_.alternatives_dict().at(vote.vote[j]);
        double delta_ab = ((ratings_[b_idx] - ratings_[a_idx]) / temperature_);
        // double sigma_ab = sigmoid(delta_ab);
        double sigma_ab = 1.0 / (1.0 + std::exp(-delta_ab));
        gradient_[a_idx] -=
            (weight * sigma_ab * (1.0 - sigma_ab) / temperature_);
        gradient_[b_idx] +=
            (weight * sigma_ab * (1.0 - sigma_ab) / temperature_);
      }
    }
  }
  divide_gradient();
}

FenchelYoungOptimizer::FenchelYoungOptimizer(
    const TupleListVote& votes, double rating_lower_bound,
    double rating_upper_bound, int batch_size, int rng_seed,
    int compute_norm_freq, double initial_param_noise, double sigma,
    const std::vector<std::string>& alternative_names)
    : Optimizer(votes, rating_lower_bound, rating_upper_bound, batch_size,
                rng_seed, compute_norm_freq, initial_param_noise,
                alternative_names),
      sigma_(sigma),
      gumbel_dist_{0.0, 1.0} {}

void FenchelYoungOptimizer::ComputeGradient(const std::vector<int>& batch) {
  std::fill(gradient_.begin(), gradient_.end(), 0.0);
  for (int vote_idx : batch) {
    const WeightedVote& vote = profile_.votes()[vote_idx];
    SPIEL_CHECK_EQ(vote.weight, 1);  // Fenchel Young only works with weight 1.
    int vote_len = vote.vote.size();
    std::vector<int> alternative_ids(vote_len);
    std::vector<double> predicted_ratings(vote_len);
    std::vector<double> target_ranking(vote_len);
    std::vector<double> local_grad(vote_len);
    for (int i = 0; i < vote_len; ++i) {
      target_ranking[i] = i;
      alternative_ids[i] = profile_.alternatives_dict().at(vote.vote[i]);
      predicted_ratings[i] =
          ratings_[alternative_ids[i]] + gumbel_dist_(rng_) * sigma_;
      // Need to do this here to assemble -\tilde{\theta}_v needed below.
      predicted_ratings[i] = -predicted_ratings[i];
    }
    // ArgSort(ArgSort(-\tilde{\theta}_v))
    std::vector<int> predicted_ranking =
        sort_indices(sort_indices(predicted_ratings));
    for (int i = 0; i < vote_len; ++i) {
      local_grad[i] = predicted_ranking[i] - target_ranking[i];
      gradient_[alternative_ids[i]] += -local_grad[i];
    }
  }
  divide_gradient();
}

}  // namespace evaluation
}  // namespace open_spiel
