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

#include "open_spiel/examples/sbr_blotto/fictitious_play.h"

#include <cmath>
#include <limits>

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace blotto_fp {

inline constexpr const double kTieTolerance = 1e-9;

ActionsAndProbs UniformSequencePolicy(int num_actions) {
  ActionsAndProbs actions_and_probs;
  actions_and_probs.reserve(num_actions);
  for (Action a = 0; a < num_actions; ++a) {
    actions_and_probs.push_back({a, 1.0 / num_actions});
  }
  return actions_and_probs;
}

ActionsAndProbs RandomStatePolicy(int num_actions,
                                  const std::function<double()>& rng) {
  ActionsAndProbs actions_and_probs;
  actions_and_probs.reserve(num_actions);
  double total_weight = 0.0;
  for (Action a = 0; a < num_actions; ++a) {
    double weight = rng();
    total_weight += weight;
    actions_and_probs.push_back({a, weight});
  }
  for (Action a = 0; a < num_actions; ++a) {
    actions_and_probs[a].second /= total_weight;
  }
  return actions_and_probs;
}

FictitiousPlayProcess::FictitiousPlayProcess(std::shared_ptr<const Game> game,
                                             int seed,
                                             bool randomize_initial_policies)
    : rng_(seed),
      dist_(0.0, 1.0),
      game_(game),
      num_players_(game->NumPlayers()),
      num_actions_(game->NumDistinctActions()),
      iterations_(0),
      total_time_(absl::ZeroDuration()) {
  // Get the info states strings.
  infostate_strings_.reserve(num_players_);
  std::unique_ptr<State> state = game->NewInitialState();
  for (Player p = 0; p < num_players_; ++p) {
    SPIEL_CHECK_EQ(state->CurrentPlayer(), p);
    std::vector<Action> legal_actions = state->LegalActions();
    std::string infostate_str = state->InformationStateString();
    infostate_strings_.push_back(infostate_str);
    state->ApplyAction(legal_actions[0]);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());

  // Get number of joint actions
  num_joint_actions_ = 1;
  for (Player p = 0; p < num_players_; ++p) {
    num_joint_actions_ *= num_actions_;
  }

  // Set initial cumulative policies to uniform.
  if (randomize_initial_policies) {
    InitPoliciesRandom();
  } else {
    InitPolicies();
  }

  GetMarginalJointPolicy(&joint_policy_);
  CheckJointUtilitiesCache();

  // Make the best response computers (for full FP)
  for (int p = 0; p < num_players_; ++p) {
    best_response_computers_.push_back(std::unique_ptr<TabularBestResponse>(
        new TabularBestResponse(*game_, p, &joint_policy_)));
  }
}

void FictitiousPlayProcess::GetMarginalJointPolicy(
    TabularPolicy* policy,
    const std::vector<std::vector<double>>* weight_table) const {
  if (weight_table == nullptr) {
    weight_table = &cumulative_policies_;
  }

  for (Player p = 0; p < num_players_; ++p) {
    double prob_sum = 0.0;
    double total_weight = std::accumulate((*weight_table)[p].begin(),
                                          (*weight_table)[p].end(), 0.0);
    for (Action a = 0; a < num_actions_; ++a) {
      double prob = (*weight_table)[p][a] / total_weight;
      SPIEL_CHECK_PROB(prob);
      policy->SetProb(infostate_strings_[p], a, prob);
      prob_sum += prob;
    }
    SPIEL_CHECK_TRUE(Near(prob_sum, 1.0));
  }
}

// Get the marginalized joint policy by marginalizing the empirical joint
// policy.
void FictitiousPlayProcess::GetMarginalJointPolicyFromEmpirical(
    TabularPolicy* policy) const {
  std::vector<std::vector<double>> marginal_weights;
  marginal_weights.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    marginal_weights.push_back(std::vector<double>(num_actions_, 0));
  }

  NormalFormCorrelationDevice mu = GetEmpiricalJointPolicy();

  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    for (Player p = 0; p < num_players_; ++p) {
      marginal_weights[p][mu[ja_idx].actions[p]] += mu[ja_idx].probability;
    }
  }

  for (Player p = 0; p < num_players_; ++p) {
    double total_weight = std::accumulate(marginal_weights[p].begin(),
                                          marginal_weights[p].end(), 0.0);
    for (Action a = 0; a < num_actions_; ++a) {
      marginal_weights[p][a] /= total_weight;
    }
  }

  GetMarginalJointPolicy(policy, &marginal_weights);
}

TabularPolicy FictitiousPlayProcess::GetLatestPolicy() const {
  TabularPolicy policy;
  for (Player p = 0; p < num_players_; ++p) {
    double prob_sum = 0.0;
    for (Action a = 0; a < num_actions_; ++a) {
      double prob = std::max(GetProb(past_policies_.back()[p], a), 0.0);
      policy.SetProb(infostate_strings_[p], a, prob);
      prob_sum += prob;
    }
    SPIEL_CHECK_TRUE(Near(prob_sum, 1.0));
  }
  return policy;
}

NormalFormCorrelationDevice FictitiousPlayProcess::GetEmpiricalJointPolicy()
    const {
  double prob_sum = 0.0;
  NormalFormCorrelationDevice corr_dev;
  corr_dev.reserve(num_joint_actions_);
  double total_weight = std::accumulate(cumulative_joint_policy_.begin(),
                                        cumulative_joint_policy_.end(), 0.0);
  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    double prob = cumulative_joint_policy_[ja_idx] / total_weight;
    SPIEL_CHECK_PROB(prob);
    corr_dev.push_back({prob, IndexToJointAction(ja_idx)});
    prob_sum += prob;
  }
  SPIEL_CHECK_TRUE(Near(prob_sum, 1.0));
  SPIEL_CHECK_EQ(corr_dev.size(), num_joint_actions_);
  return corr_dev;
}

void FictitiousPlayProcess::InitPoliciesRandom() {
  // Cumulative policies per player.
  cumulative_policies_.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    std::vector<double> policy(num_actions_, 0);
    double total_weight = 0.0;
    for (int a = 0; a < num_actions_; ++a) {
      policy[a] = dist_(rng_);
      total_weight += policy[a];
    }
    for (int a = 0; a < num_actions_; ++a) {
      policy[a] /= total_weight;
    }
    cumulative_policies_.push_back(policy);
  }

  // Initial, current, and past policies
  current_joint_policy_counts_.resize(num_joint_actions_);
  std::vector<ActionsAndProbs> initial_policies;
  initial_policies.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    initial_policies.push_back(
        RandomStatePolicy(num_actions_, [this]() { return dist_(rng_); }));
  }
  past_policies_.push_back(initial_policies);

  // Cumulative joint policy.
  cumulative_joint_policy_ = std::vector<double>(num_joint_actions_, 0);
  double total_weight = 0.0;
  for (int idx = 0; idx < num_joint_actions_; ++idx) {
    cumulative_joint_policy_[idx] = dist_(rng_);
    total_weight += cumulative_joint_policy_[idx];
  }
  for (int idx = 0; idx < num_joint_actions_; ++idx) {
    cumulative_joint_policy_[idx] /= total_weight;
  }
}

void FictitiousPlayProcess::InitPolicies() {
  // Cumulative policies per player.
  cumulative_policies_.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    std::vector<double> uniform_policy(num_actions_, 1.0 / num_actions_);
    cumulative_policies_.push_back(uniform_policy);
  }

  // Initial, current, and past policies
  current_joint_policy_counts_.resize(num_joint_actions_);
  std::vector<ActionsAndProbs> initial_policies;
  initial_policies.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    initial_policies.push_back(UniformSequencePolicy(num_actions_));
  }
  past_policies_.push_back(initial_policies);

  // Cumulative joint policy.
  cumulative_joint_policy_ =
      std::vector<double>(num_joint_actions_, 1.0 / num_joint_actions_);
}

// Add appropriate weights given each players' (potentially mixed) policy
void FictitiousPlayProcess::UpdateCumulativeJointPolicy(
    const std::vector<std::vector<double>>& policies) {
  double sum_weights = 0.0;
  double sum_joint_policy = 0.0;
  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    std::vector<Action> joint_action = IndexToJointAction(ja_idx);
    double weight = 1.0;
    for (Player p = 0; p < num_players_; ++p) {
      Action action = joint_action[p];
      weight *= policies[p][action];
    }
    cumulative_joint_policy_[ja_idx] += weight;
    sum_weights += weight;
    sum_joint_policy += cumulative_joint_policy_[ja_idx];
  }
  SPIEL_CHECK_FLOAT_NEAR(sum_weights, 1.0, 1e-12);
  SPIEL_CHECK_FLOAT_NEAR(sum_joint_policy, iterations_ + 2.0, 1e-12);
}

void FictitiousPlayProcess::UpdateCumulativeJointPolicySampled(
    const std::vector<std::vector<double>>& policies, int num_samples) {
  double weight = 1.0 / num_samples;
  std::vector<absl::discrete_distribution<int>> dists;
  dists.reserve(policies.size());
  for (Player p = 0; p < policies.size(); ++p) {
    dists.push_back(absl::discrete_distribution<int>(policies[p].begin(),
                                                     policies[p].end()));
  }
  for (int sample = 0; sample < num_samples; ++sample) {
    std::vector<Action> joint_action;
    joint_action.reserve(num_players_);
    for (Player p = 0; p < policies.size(); ++p) {
      joint_action.push_back(dists[p](rng_));
    }
    int ja_idx = JointActionToIndex(joint_action);
    cumulative_joint_policy_[ja_idx] += weight;
  }
}

std::vector<double> FictitiousPlayProcess::Softmax(
    const std::vector<double>& values, double lambda) const {
  std::vector<double> new_values = values;
  for (double& new_value : new_values) {
    new_value *= lambda;
  }
  double max = *std::max_element(new_values.begin(), new_values.end());

  double denom = 0;
  for (int idx = 0; idx < values.size(); ++idx) {
    new_values[idx] = std::exp(new_values[idx] - max);
    denom += new_values[idx];
  }

  SPIEL_CHECK_GT(denom, 0);
  double prob_sum = 0.0;
  std::vector<double> policy;
  policy.reserve(new_values.size());
  for (int idx = 0; idx < values.size(); ++idx) {
    double prob = new_values[idx] / denom;
    SPIEL_CHECK_PROB(prob);
    prob_sum += prob;
    policy.push_back(prob);
  }

  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-12);
  return policy;
}

Action FictitiousPlayProcess::BestResponseAgainstEmpiricalJointPolicy(
    Player player, std::vector<double>* values) {
  double best_action_value = -10;
  Action best_action = kInvalidAction;
  // NormalFormCorrelationDevice mu = GetEmpiricalJointPolicy();
  double total_weight = std::accumulate(cumulative_joint_policy_.begin(),
                                        cumulative_joint_policy_.end(), 0.0);

  for (Action a = 0; a < num_actions_; ++a) {
    double value = 0.0;
    for (int idx = 0; idx < num_joint_actions_; ++idx) {
      // std::vector<Action> joint_action = mu[idx].actions;
      std::vector<Action> joint_action = IndexToJointAction(idx);
      joint_action[player] = a;
      int new_ja_idx = JointActionToIndex(joint_action);
      // value +=
      //    mu[idx].probability * cached_joint_utilities_[player][new_ja_idx];
      value += (cumulative_joint_policy_[idx] *
                cached_joint_utilities_[player][new_ja_idx]);
    }
    value /= total_weight;
    if (values != nullptr) {
      (*values)[a] = value;
    }
    if (value > best_action_value) {
      best_action_value = value;
      best_action = a;
    }
  }

  return best_action;
}

Action FictitiousPlayProcess::BestResponseAgainstEmpiricalMarginalizedPolicies(
    Player player, std::vector<double>* values) {
  TabularPolicy marginalized_joint_policy;
  GetMarginalJointPolicyFromEmpirical(&marginalized_joint_policy);
  best_response_computers_[player]->SetPolicy(&marginalized_joint_policy);
  TabularPolicy br = best_response_computers_[player]->GetBestResponsePolicy();
  Action br_action = GetAction(br.GetStatePolicy(infostate_strings_[player]));
  if (values != nullptr) {
    std::vector<std::pair<Action, double>> action_vals =
        best_response_computers_[player]->BestResponseActionValues(
            infostate_strings_[player]);
    values->resize(action_vals.size());
    for (const auto& iter : action_vals) {
      (*values)[iter.first] = iter.second;
    }
  }
  return br_action;
}

void FictitiousPlayProcess::CheckJointUtilitiesCache() {
  if (cached_joint_utilities_.empty()) {
    cached_joint_utilities_.reserve(num_players_);
    for (Player p = 0; p < num_players_; ++p) {
      cached_joint_utilities_.push_back(
          std::vector<double>(num_joint_actions_, 0));
    }

    for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
      std::vector<Action> joint_action = IndexToJointAction(ja_idx);
      std::unique_ptr<State> state = game_->NewInitialState();
      for (Action action : joint_action) {
        state->ApplyAction(action);
      }
      SPIEL_CHECK_TRUE(state->IsTerminal());
      std::vector<double> returns = state->Returns();
      for (Player p = 0; p < num_players_; ++p) {
        cached_joint_utilities_[p][ja_idx] = returns[p];
      }
    }
  }
}

double FictitiousPlayProcess::CCEDist() const {
  double dist = 0;
  std::vector<double> max_deviation(num_players_, -10);
  std::vector<double> exp_values(num_players_, 0);
  NormalFormCorrelationDevice corr_dev = GetEmpiricalJointPolicy();

  // First compute expected values for everyone
  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    if (corr_dev[ja_idx].probability > 0) {
      for (Player p = 0; p < num_players_; ++p) {
        exp_values[p] +=
            corr_dev[ja_idx].probability * cached_joint_utilities_[p][ja_idx];
      }
    }
  }

  // Now for each player, find the maximal deviation
  for (Player p = 0; p < num_players_; ++p) {
    for (Action a = 0; a < num_actions_; ++a) {
      double action_value = 0;
      for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
        if (corr_dev[ja_idx].probability > 0) {
          // Player p is changing to choose a instead.
          std::vector<Action> joint_action = IndexToJointAction(ja_idx);
          joint_action[p] = a;
          int other_index = JointActionToIndex(joint_action);

          action_value += corr_dev[ja_idx].probability *
                          cached_joint_utilities_[p][other_index];
        }
      }
      if (action_value > max_deviation[p]) {
        max_deviation[p] = action_value;
      }
    }
  }

  for (Player p = 0; p < num_players_; ++p) {
    double delta = std::max(max_deviation[p] - exp_values[p], 0.0);
    SPIEL_CHECK_GE(delta, 0);
    dist += delta;
  }

  return dist;
}

double FictitiousPlayProcess::NashConv() const {
  TabularPolicy marginalized_policy;
  GetMarginalJointPolicyFromEmpirical(&marginalized_policy);
  return open_spiel::algorithms::NashConv(*game_, marginalized_policy);
  // TabularPolicy marginalized_policy;
  // GetMarginalJointPolicy(&marginalized_policy);
  // return open_spiel::algorithms::NashConv(*game_, marginalized_policy);
}

int FictitiousPlayProcess::JointActionToIndex(
    const std::vector<Action>& joint_action) const {
  // Convert to a number from base num_actions_
  int index = 0;
  int digit_value = 1;
  for (int i = 0; i < joint_action.size(); ++i) {
    index += joint_action[i] * digit_value;
    digit_value *= num_actions_;
  }
  SPIEL_CHECK_LT(index, num_joint_actions_);
  return index;
}

std::vector<Action> FictitiousPlayProcess::IndexToJointAction(int index) const {
  // Convert to a number in base num_actions_
  std::vector<Action> joint_action(num_players_, kInvalidAction);
  for (int i = 0; i < num_players_; ++i) {
    joint_action[i] = index % num_actions_;
    index /= num_actions_;
  }
  SPIEL_CHECK_EQ(index, 0);
  return joint_action;
}

void FictitiousPlayProcess::IBRIteration() {
  absl::Time start = absl::Now();

  // Compute the joint policy.
  GetMarginalJointPolicy(&joint_policy_);

  // Get each player's best response, and add it to the cumulative policy.
  std::vector<TabularPolicy> br_policies(num_players_);

  std::vector<Action> joint_action(num_players_, kInvalidAction);

  for (Player p = 0; p < num_players_; ++p) {
    best_response_computers_[p]->SetPolicy(&joint_policy_);
    br_policies[p] = best_response_computers_[p]->GetBestResponsePolicy();
    Action br_action =
        GetAction(br_policies[p].GetStatePolicy(infostate_strings_[p]));
    SPIEL_CHECK_TRUE(br_action != kInvalidAction);
    std::fill(cumulative_policies_[p].begin(), cumulative_policies_[p].end(),
              0);
    cumulative_policies_[p][br_action] = 1.0;
    joint_action[p] = br_action;
  }

  std::fill(cumulative_joint_policy_.begin(), cumulative_joint_policy_.end(),
            0.0);
  cumulative_joint_policy_[JointActionToIndex(joint_action)] = 1.0;

  iterations_++;
  total_time_ += absl::Now() - start;
}

void FictitiousPlayProcess::MaxEntIBRIteration() {
  absl::Time start = absl::Now();

  // Compute the joint policy.
  GetMarginalJointPolicy(&joint_policy_);

  // Get each player's best response, and add it to the cumulative policy.
  std::vector<TabularPolicy> br_policies(num_players_);

  std::vector<std::vector<double>> policies;
  policies.reserve(num_players_);

  for (Player p = 0; p < num_players_; ++p) {
    best_response_computers_[p]->SetPolicy(&joint_policy_);
    br_policies[p] = best_response_computers_[p]->GetBestResponsePolicy();
    std::vector<Action> br_actions =
        best_response_computers_[p]->BestResponseActions(infostate_strings_[p],
                                                         1e-10);
    SPIEL_CHECK_GT(br_actions.size(), 0);
    std::fill(cumulative_policies_[p].begin(), cumulative_policies_[p].end(),
              0);

    std::vector<double> br(num_actions_, 0);

    for (Action action : br_actions) {
      double prob = 1.0 / br_actions.size();
      cumulative_policies_[p][action] = prob;
      br[action] = prob;
    }

    policies.push_back(br);
  }

  // Update empirical cumulative dist with these mixed policies
  std::fill(cumulative_joint_policy_.begin(), cumulative_joint_policy_.end(),
            0.0);
  UpdateCumulativeJointPolicy(policies);

  iterations_++;
  total_time_ += absl::Now() - start;
}

void FictitiousPlayProcess::FullFPIteration() {
  absl::Time start = absl::Now();

  std::vector<Action> joint_action(num_players_, kInvalidAction);

  for (Player p = 0; p < num_players_; ++p) {
    // Action br_action = BestResponseAgainstEmpiricalMarginalizedPolicies(p);
    Action br_action = BestResponseAgainstEmpiricalJointPolicy(p);
    SPIEL_CHECK_TRUE(br_action != kInvalidAction);
    cumulative_policies_[p][br_action] += 1.0;
    joint_action[p] = br_action;
  }

  cumulative_joint_policy_[JointActionToIndex(joint_action)] += 1.0;

  iterations_++;
  total_time_ += absl::Now() - start;
}

void FictitiousPlayProcess::SFPIteration(double lambda) {
  absl::Time start = absl::Now();

  std::vector<std::vector<double>> softmax_brs;
  softmax_brs.reserve(num_players_);

  for (Player p = 0; p < num_players_; ++p) {
    std::vector<double> values(num_actions_, 0);
    // BestResponseAgainstEmpiricalMarginalizedPolicies(p, &values);
    BestResponseAgainstEmpiricalJointPolicy(p, &values);

    std::vector<double> softmax_br = Softmax(values, lambda);
    softmax_brs.push_back(softmax_br);
  }

  for (Player p = 0; p < num_players_; ++p) {
    for (int i = 0; i < softmax_brs[p].size(); ++i) {
      cumulative_policies_[p][i] += softmax_brs[p][i];
    }
  }

  // Update empirical cumulative dist with these mixed policies
  UpdateCumulativeJointPolicy(softmax_brs);

  iterations_++;
  total_time_ += absl::Now() - start;
}

// This is FP+SBR in the paper, samples the base profiles from the average
// strategy.
void FictitiousPlayProcess::SBRIteration(int num_base_samples,
                                         int num_candidates) {
  absl::Time start = absl::Now();

  std::vector<Action> joint_action(num_players_, kInvalidAction);

  // Sample the base profiles: number by player
  std::vector<std::vector<Action>> base_samples;
  base_samples.reserve(num_base_samples);
  for (int i = 0; i < num_base_samples; ++i) {
    std::vector<Action> base_profile;
    base_profile.reserve(num_players_);
    int past_idx = static_cast<int>(dist_(rng_) * past_policies_.size());
    for (Player p = 0; p < num_players_; ++p) {
      base_profile.push_back(
          SampleAction(past_policies_[past_idx][p], dist_(rng_)).first);
    }
    base_samples.push_back(base_profile);
  }

  // Each player computes a sampled BR.
  for (Player p = 0; p < num_players_; ++p) {
    double max_return = -std::numeric_limits<double>::infinity();
    Action best_candidate = kInvalidAction;

    for (int i = 0; i < num_candidates; ++i) {
      Action sampled_candidate =
          absl::Uniform(rng_, 0u, static_cast<unsigned int>(num_actions_));
      // Compute the action's expectation.
      double return_sum = 0.0;
      for (const std::vector<Action>& base_joint_action : base_samples) {
        std::vector<Action> joint_action = base_joint_action;
        joint_action[p] = sampled_candidate;
        std::unique_ptr<State> state = game_->NewInitialState();
        // Turn-based simultaneous game, so must apply them in order.
        for (Player pp = 0; pp < num_players_; ++pp) {
          state->ApplyAction(joint_action[pp]);
        }
        return_sum += state->PlayerReturn(p);
      }
      if (return_sum / num_base_samples > max_return) {
        max_return = return_sum / num_base_samples;
        best_candidate = sampled_candidate;
      }
    }

    SPIEL_CHECK_TRUE(best_candidate != kInvalidAction);
    cumulative_policies_[p][best_candidate] += 1.0;
    joint_action[p] = best_candidate;
  }

  // Add to past policies
  std::vector<ActionsAndProbs> new_policy;
  new_policy.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    new_policy.push_back({{joint_action[p], 1.0}});
  }
  past_policies_.push_back(new_policy);

  cumulative_joint_policy_[JointActionToIndex(joint_action)] += 1.0;

  iterations_++;
  total_time_ += absl::Now() - start;
}

void FictitiousPlayProcess::AddWeight(ActionsAndProbs* policy, Action action,
                                      double weight) const {
  double prob = std::max(0.0, GetProb(*policy, action));
  SetProb(policy, action, prob + weight);
}

std::vector<Action> FictitiousPlayProcess::SampleBaseProfile(
    BaseSamplerType sampler_type) {
  std::vector<Action> base_profile;
  base_profile.reserve(num_players_);

  if (sampler_type == BaseSamplerType::kBaseUniform) {
    int past_idx = static_cast<int>(dist_(rng_) * past_policies_.size());
    for (Player p = 0; p < num_players_; ++p) {
      base_profile.push_back(
          SampleAction(past_policies_[past_idx][p], dist_(rng_)).first);
    }
    return base_profile;
  } else if (sampler_type == BaseSamplerType::kBaseLatest) {
    int past_idx = past_policies_.size() - 1;
    for (Player p = 0; p < num_players_; ++p) {
      base_profile.push_back(
          SampleAction(past_policies_[past_idx][p], dist_(rng_)).first);
    }
    return base_profile;
  } else {
    SpielFatalError("Base sampling method unrecognized.");
  }
}

Action FictitiousPlayProcess::SampleCandidate(
    Player player, CandidatesSamplerType sampler_type) {
  if (sampler_type == CandidatesSamplerType::kCandidatesInitial) {
    return absl::Uniform(rng_, 0u, static_cast<unsigned int>(num_actions_));
  } else if (sampler_type == CandidatesSamplerType::kCandidatesUniform ||
             sampler_type == CandidatesSamplerType::kCandidatesLatest) {
    int past_idx =
        sampler_type == CandidatesSamplerType::kCandidatesUniform
            ? absl::Uniform(rng_, 0u,
                            static_cast<unsigned int>(past_policies_.size()))
            : past_policies_.size() - 1;
    return SampleAction(past_policies_[past_idx][player], dist_(rng_)).first;
  } else if (sampler_type == CandidatesSamplerType::kCandidatesInitialUniform) {
    int bit = absl::Uniform(rng_, 0u, 2u);
    if (bit == 0) {
      return SampleCandidate(player, CandidatesSamplerType::kCandidatesInitial);
    } else {
      return SampleCandidate(player, CandidatesSamplerType::kCandidatesUniform);
    }
  } else if (sampler_type == CandidatesSamplerType::kCandidatesInitialLatest) {
    int bit = absl::Uniform(rng_, 0u, 2u);
    if (bit == 0) {
      return SampleCandidate(player, CandidatesSamplerType::kCandidatesInitial);
    } else {
      return SampleCandidate(player, CandidatesSamplerType::kCandidatesLatest);
    }
  } else {
    SpielFatalError("Candidate sampling method unrecognized.");
  }
}

std::vector<std::vector<Action>> FictitiousPlayProcess::SampleBaseProfiles(
    BaseSamplerType sampler_type, int num_base_samples) {
  std::vector<std::vector<Action>> base_samples;
  base_samples.reserve(num_base_samples);
  for (int i = 0; i < num_base_samples; ++i) {
    base_samples.push_back(SampleBaseProfile(sampler_type));
  }
  return base_samples;
}

Action FictitiousPlayProcess::GetBestCandidate(
    Player player, const std::vector<std::vector<Action>>& base_samples,
    int num_candidates, CandidatesSamplerType sampler_type) {
  std::vector<Action> best_candidates;
  double max_return = -std::numeric_limits<double>::infinity();

  for (int i = 0; i < num_candidates; ++i) {
    Action sampled_candidate = SampleCandidate(player, sampler_type);
    // Compute the action's expectation.
    double return_sum = 0.0;
    for (const std::vector<Action>& base_joint_action : base_samples) {
      std::vector<Action> ja_prime = base_joint_action;
      ja_prime[player] = sampled_candidate;
      int ja_idx = JointActionToIndex(ja_prime);
      return_sum += cached_joint_utilities_[player][ja_idx];
    }

    // Consider values within [ -kTieTolerance, kTieTolerance ] as tied.
    if (return_sum > max_return + kTieTolerance) {
      max_return = return_sum;
      best_candidates = {sampled_candidate};
    } else if (return_sum > max_return - kTieTolerance) {
      best_candidates.push_back(sampled_candidate);
    }
  }

  SPIEL_CHECK_GE(best_candidates.size(), 0);
  if (best_candidates.size() == 1) {
    return best_candidates[0];
  } else {
    int idx = absl::Uniform(rng_, 0u,
                            static_cast<unsigned int>(best_candidates.size()));
    return best_candidates[idx];
  }
}

// pi_i^t = \frac{1}{N} \sum_{n = 1}^N 1(a_i),  where a_i ~ SBR(\pi_b, \pi_c).
void FictitiousPlayProcess::BRPIIteration(
    BaseSamplerType base_sampling, CandidatesSamplerType candidates_sampling,
    int num_base_samples, int num_candidates, int brpi_N) {
  absl::Time start = absl::Now();

  // Clear policy counts
  std::fill(current_joint_policy_counts_.begin(),
            current_joint_policy_counts_.end(), 0.0);

  // N trials
  for (int n = 0; n < brpi_N; ++n) {
    std::vector<Action> joint_action(num_players_, kInvalidAction);

    // Sample the base profiles: number by player
    std::vector<std::vector<Action>> base_samples =
        SampleBaseProfiles(base_sampling, num_base_samples);

    // Each player computes a sampled BR.
    for (Player p = 0; p < num_players_; ++p) {
      Action best_candidate = GetBestCandidate(p, base_samples, num_candidates,
                                               candidates_sampling);
      joint_action[p] = best_candidate;
    }
    current_joint_policy_counts_[JointActionToIndex(joint_action)] += 1.0;

    // End of trial.
  }

  // Apply the 1/N to the emprical estimate of the joint policy and add them to
  // the past policies.
  std::vector<ActionsAndProbs> policies(num_players_);
  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    if (current_joint_policy_counts_[ja_idx] > 0) {
      double weight = current_joint_policy_counts_[ja_idx] / brpi_N;
      std::vector<Action> joint_action = IndexToJointAction(ja_idx);
      for (Player p = 0; p < num_players_; ++p) {
        AddWeight(&policies[p], joint_action[p], weight);
      }
    }
  }
  past_policies_.push_back(policies);

  std::fill(cumulative_joint_policy_.begin(), cumulative_joint_policy_.end(),
            0.0);
  for (int ja_idx = 0; ja_idx < num_joint_actions_; ++ja_idx) {
    std::vector<Action> joint_action = IndexToJointAction(ja_idx);
    double joint_prob = 1.0;
    for (Player p = 0; p < num_players_ && joint_prob > 0; ++p) {
      double prob = std::max(0.0, GetProb(policies[p], joint_action[p]));
      if (prob == 0) {
        joint_prob = 0;
      } else {
        joint_prob *= prob;
      }
    }
    if (joint_prob > 0) {
      cumulative_joint_policy_[ja_idx] = joint_prob;
    }
  }

  iterations_++;
  total_time_ += absl::Now() - start;
}

}  // namespace blotto_fp
}  // namespace algorithms
}  // namespace open_spiel
