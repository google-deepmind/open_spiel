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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_optional.h"

namespace open_spiel {
namespace algorithms {

// A basic structure to store the relevant quantities.
struct CFRInfoStateValues {
  CFRInfoStateValues() {}
  CFRInfoStateValues(std::vector<Action> la, double init_value)
      : legal_actions(la),
        cumulative_regrets(la.size(), init_value),
        cumulative_policy(la.size(), init_value),
        current_policy(la.size(), 1.0 / la.size()) {}
  CFRInfoStateValues(std::vector<Action> la) : CFRInfoStateValues(la, 0) {}

  void ApplyRegretMatching();  // Fills current_policy.
  bool empty() const { return legal_actions.empty(); }
  int num_actions() const { return legal_actions.size(); }

  // Samples from current policy using randomly generated z, adding epsilon
  // exploration (mixing in uniform).
  int SampleActionIndex(double epsilon, double z);

  std::vector<Action> legal_actions;
  std::vector<double> cumulative_regrets;
  std::vector<double> cumulative_policy;
  std::vector<double> current_policy;
};

// A type for tables holding CFR values.
using CFRInfoStateValuesTable =
    std::unordered_map<std::string, CFRInfoStateValues>;

// A policy that extracts the average policy from the CFR table values, which
// can be passed to tabular exploitability.
class CFRAveragePolicy : public Policy {
 public:
  // Returns the average policy from the CFR values. If a default policy is
  // passed in, then it means that it is used if the lookup fails (use nullptr
  // to not use a default policy).
  CFRAveragePolicy(const CFRInfoStateValuesTable& info_states,
                   std::shared_ptr<TabularPolicy> default_policy);
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

 private:
  const CFRInfoStateValuesTable& info_states_;
  bool default_to_uniform_;
  std::shared_ptr<TabularPolicy> default_policy_;
};

// Base class supporting different flavours of the Counterfactual Regret
// Minimization (CFR) algorithm.
//
// see https://webdocs.cs.ualberta.ca/~bowling/papers/07nips-regretpoker.pdf
// and http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
//
// The implementation is similar to the Python version:
//   open_spiel/python/algorithms/cfr.py
//
// The algorithm computes an approximate Nash policy for 2 player zero-sum
// games.
//
// CFR can be view as a policy iteration algorithm. Importantly, the policies
// themselves do not converge to a Nash policy, but their average does.
//
class CFRSolverBase {
 public:
  CFRSolverBase(const Game& game, bool initialize_cumulative_values,
                bool alternating_updates, bool linear_averaging,
                bool regret_matching_plus);

  // Performs one step of the CFR algorithm.
  void EvaluateAndUpdatePolicy();

  // Computes the average policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::unique_ptr<Policy> AveragePolicy() const {
    return std::unique_ptr<Policy>(new CFRAveragePolicy(info_states_, nullptr));
  }

 private:
  static constexpr double kInitialPositiveValue_ = 1e-5;

  std::vector<double> ComputeCounterFactualRegret(
      const State& state, const Optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities);

  std::vector<double> ComputeCounterFactualRegretForActionProbs(
      const State& state, const Optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities, const int current_player,
      const std::vector<double>& info_state_policy,
      const std::vector<Action>& legal_actions,
      std::vector<double>* child_values_out = nullptr);

  void InitializeUniformPolicy(const State& state);

  // Get the policy at this information state. The probabilities are ordered in
  // the same order as legal_actions.
  std::vector<double> GetPolicy(const std::string& info_state,
                                const std::vector<Action>& legal_actions);

  void ApplyRegretMatchingPlusReset();
  void ApplyRegretMatching();

  std::vector<double> RegretMatching(const std::string& info_state,
                                     const std::vector<Action>& legal_actions);

  bool AllPlayersHaveZeroReachProb(
      const std::vector<double>& reach_probabilities) const;

  const Game& game_;
  const bool regret_matching_plus_;
  const bool alternating_updates_;
  const bool linear_averaging_;

  const int chance_player_;
  const std::unique_ptr<State> root_state_;
  const std::vector<double> root_reach_probs_;

  // Iteration to support linear_policy.
  int iteration_ = 0;
  CFRInfoStateValuesTable info_states_;
};

// Standard CFR implementation.
//
// See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
class CFRSolver : public CFRSolverBase {
 public:
  explicit CFRSolver(const Game& game)
      : CFRSolverBase(game, /*initialize_cumulative_values=*/false,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/false,
                      /*regret_matching_plus=*/false) {}
};

// CFR+ implementation.
//
// See https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf
//
// CFR+ is CFR with the following modifications:
// - use Regret Matching+ instead of Regret Matching.
// - use alternating updates instead of simultaneous updates.
// - use linear averaging.
class CFRPlusSolver : public CFRSolverBase {
 public:
  CFRPlusSolver(const Game& game)
      : CFRSolverBase(game, /*initialize_cumulative_values=*/true,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/true,
                      /*regret_matching_plus=*/true) {}
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_
