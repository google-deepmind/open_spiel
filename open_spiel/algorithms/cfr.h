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

#ifndef OPEN_SPIEL_ALGORITHMS_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_CFR_H_

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

constexpr const char* kSerializeMetaSectionHeader = "[Meta]";
constexpr const char* kSerializeGameSectionHeader = "[Game]";
constexpr const char* kSerializeSolverTypeSectionHeader = "[SolverType]";
constexpr const char* kSerializeSolverSpecificStateSectionHeader =
    "[SolverSpecificState]";
constexpr const char* kSerializeSolverValuesTableSectionHeader =
    "[SolverValuesTable]";

// A basic structure to store the relevant quantities.
struct CFRInfoStateValues {
  CFRInfoStateValues() {}
  CFRInfoStateValues(std::vector<Action> la, double init_value)
      : legal_actions(la),
        cumulative_regrets(la.size(), init_value),
        cumulative_policy(la.size(), init_value),
        current_policy(la.size(), 1.0 / la.size()) {}
  CFRInfoStateValues(std::vector<Action> la) : CFRInfoStateValues(la, 0) {}

  // For randomized initial regrets.
  CFRInfoStateValues(std::vector<Action> la,
                     std::mt19937* rng,
                     double magnitude_scale) : CFRInfoStateValues(la, 0) {
    for (int i = 0; i < cumulative_policy.size(); ++i) {
      cumulative_regrets[i] = magnitude_scale *
          absl::Uniform<double>(*rng, 0.0, 1.0);
    }
    ApplyRegretMatching();
  }

  // Fills current_policy according to the standard application of the
  // regret-matching algorithm in the CFR papers.
  void ApplyRegretMatching();

  // Apply regret matching but over max(R^{T,+}(s,a), delta) rather than just
  // R^{T,+}(s,a). This is mostly unused but sometimes useful for debugging
  // convergence.
  void ApplyRegretMatchingAllPositive(double delta);

  bool empty() const { return legal_actions.empty(); }
  int num_actions() const { return legal_actions.size(); }

  // A string representation of the information state values.
  std::string ToString() const;

  // A less verbose string representation used for serialization purposes. The
  // double_precision parameter indicates the number of decimal places in
  // floating point numbers formatting, value -1 formats doubles with lossless,
  // non-portable bitwise representation hex strings.
  std::string Serialize(int double_precision) const;

  // Samples from current policy using randomly generated z, adding epsilon
  // exploration (mixing in uniform).
  int SampleActionIndex(double epsilon, double z);

  // Extracts the current policy. Note: assumes it is filled.
  ActionsAndProbs GetCurrentPolicy() const;

  // Return index of the action within the vector of legal_actions,
  // or exit with an error.
  int GetActionIndex(Action a);

  std::vector<Action> legal_actions;
  std::vector<double> cumulative_regrets;
  std::vector<double> cumulative_policy;
  std::vector<double> current_policy;
};

CFRInfoStateValues DeserializeCFRInfoStateValues(absl::string_view serialized);

// A type for tables holding CFR values.
using CFRInfoStateValuesTable =
    std::unordered_map<std::string, CFRInfoStateValues>;

// The result parameter is passed by pointer in order to avoid copying/moving
// the string once the table is fully serialized (CFRInfoStateValuesTable
// instances could be very large). See comments above
// CFRInfoStateValues::Serialize(double_precision) for notes about the
// double_precision parameter.
void SerializeCFRInfoStateValuesTable(
    const CFRInfoStateValuesTable& info_states, std::string* result,
    int double_precision, std::string delimiter = "<~>");

// Similarly as above, the result parameter is passed by pointer in order to
// avoid copying/moving the table once fully deserialized.
void DeserializeCFRInfoStateValuesTable(absl::string_view serialized,
                                        CFRInfoStateValuesTable* result,
                                        std::string delimiter = "<~>");

// A policy that extracts the average policy from the CFR table values, which
// can be passed to tabular exploitability.
class CFRAveragePolicy : public Policy {
 public:
  // Returns the average policy from the CFR values.
  // If a state/info state is not found, return the default policy for the
  // state/info state (or an empty policy if default_policy is nullptr).
  // If an info state has zero cumulative regret for all actions,
  // return a uniform policy.
  CFRAveragePolicy(const CFRInfoStateValuesTable& info_states,
                   std::shared_ptr<Policy> default_policy);
  ActionsAndProbs GetStatePolicy(const State& state) const override {
    return GetStatePolicy(state, state.CurrentPlayer());
  };
  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;
  TabularPolicy AsTabular() const;

 private:
  const CFRInfoStateValuesTable& info_states_;
  UniformPolicy uniform_policy_;
  std::shared_ptr<Policy> default_policy_;
  void GetStatePolicyFromInformationStateValues(
      const CFRInfoStateValues& is_vals,
      ActionsAndProbs* actions_and_probs) const;
};

// A policy that extracts the current policy from the CFR table values.
class CFRCurrentPolicy : public Policy {
 public:
  // Returns the current policy from the CFR values. If a default policy is
  // passed in, then it means that it is used if the lookup fails (use nullptr
  // to not use a default policy).
  CFRCurrentPolicy(const CFRInfoStateValuesTable& info_states,
                   std::shared_ptr<Policy> default_policy);
  ActionsAndProbs GetStatePolicy(const State& state) const override {
    return GetStatePolicy(state, state.CurrentPlayer());
  };
  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;
  TabularPolicy AsTabular() const;

 private:
  const CFRInfoStateValuesTable& info_states_;
  std::shared_ptr<Policy> default_policy_;
  ActionsAndProbs GetStatePolicyFromInformationStateValues(
      const CFRInfoStateValues& is_vals,
      ActionsAndProbs& actions_and_probs) const;
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
  CFRSolverBase(const Game& game, bool alternating_updates,
                bool linear_averaging, bool regret_matching_plus,
                bool random_initial_regrets = false, int seed = 0);
  // The constructor below is used for deserialization purposes.
  CFRSolverBase(std::shared_ptr<const Game> game, bool alternating_updates,
                bool linear_averaging, bool regret_matching_plus, int iteration,
                bool random_initial_regrets = false, int seed = 0);
  virtual ~CFRSolverBase() = default;

  // Performs one step of the CFR algorithm.
  virtual void EvaluateAndUpdatePolicy();

  // Computes the average policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::shared_ptr<Policy> AveragePolicy() const {
    return std::make_shared<CFRAveragePolicy>(info_states_, nullptr);
  }
  // Note: This can be quite large.
  TabularPolicy TabularAveragePolicy() const {
    CFRAveragePolicy policy(info_states_, nullptr);
    return policy.AsTabular();
  }

  // Computes the current policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::shared_ptr<Policy> CurrentPolicy() const {
    return std::make_shared<CFRCurrentPolicy>(info_states_, nullptr);
  }

  CFRInfoStateValuesTable& InfoStateValuesTable() { return info_states_; }

  // See comments above CFRInfoStateValues::Serialize(double_precision) for
  // notes about the double_precision parameter.
  std::string Serialize(int double_precision = -1,
                        std::string delimiter = "<~>") const;

 protected:
  std::shared_ptr<const Game> game_;

  // Iteration to support linear_policy.
  int iteration_ = 0;
  CFRInfoStateValuesTable info_states_;
  const std::unique_ptr<State> root_state_;
  const std::vector<double> root_reach_probs_;

  // Compute the counterfactual regret and update the average policy for the
  // specified player.
  // The optional `policy_overrides` can be used to specify for each player a
  // policy to use instead of the current policy. `policy_overrides=nullptr`
  // will disable this feature. Otherwise it should be a [num_players] vector,
  // and if `policy_overrides[p] != nullptr` it will be used instead of the
  // current policy. This feature exists to support CFR-BR.
  std::vector<double> ComputeCounterFactualRegret(
      const State& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities,
      const std::vector<const Policy*>* policy_overrides);

  // Update the current policy for all information states.
  void ApplyRegretMatching();

  // This method should return the type of itself so that it can be checked
  // in different deserialization methods; one method for each subtype.
  // For an example take a look at the CFRSolver::SerializeThisType() and
  // DeserializeCFRSolver() methods.
  virtual std::string SerializeThisType() const {
    SpielFatalError("Serialization of the base class is not supported.");
  }

 private:
  std::vector<double> ComputeCounterFactualRegretForActionProbs(
      const State& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities, const int current_player,
      const std::vector<double>& info_state_policy,
      const std::vector<Action>& legal_actions,
      std::vector<double>* child_values_out,
      const std::vector<const Policy*>* policy_overrides);

  void InitializeInfostateNodes(const State& state);

  // Fills `info_state_policy` to be a [num_actions] vector of the probabilities
  // found in `policy` at the given `info_state`.
  void GetInfoStatePolicyFromPolicy(std::vector<double>* info_state_policy,
                                    const std::vector<Action>& legal_actions,
                                    const Policy* policy,
                                    const std::string& info_state) const;

  // Get the policy at this information state. The probabilities are ordered in
  // the same order as legal_actions.
  std::vector<double> GetPolicy(const std::string& info_state,
                                const std::vector<Action>& legal_actions);

  void ApplyRegretMatchingPlusReset();

  std::vector<double> RegretMatching(const std::string& info_state,
                                     const std::vector<Action>& legal_actions);

  bool AllPlayersHaveZeroReachProb(
      const std::vector<double>& reach_probabilities) const;

  const bool regret_matching_plus_;
  const bool alternating_updates_;
  const bool linear_averaging_;
  const bool random_initial_regrets_;

  const int chance_player_;

  // CFR generally does not use this random number generator. However, this is
  // used for random initial regrets (and could be useful for some helper
  // methods for debugging).
  std::mt19937 rng_;
};

// Standard CFR implementation.
//
// See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
class CFRSolver : public CFRSolverBase {
 public:
  explicit CFRSolver(const Game& game)
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/false,
                      /*regret_matching_plus=*/false) {}
  // The constructor below is used for deserialization purposes.
  CFRSolver(std::shared_ptr<const Game> game, int iteration)
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/false,
                      /*regret_matching_plus=*/false, iteration) {}

 protected:
  std::string SerializeThisType() const { return "CFRSolver"; }
};

std::unique_ptr<CFRSolver> DeserializeCFRSolver(const std::string& serialized,
                                                std::string delimiter = "<~>");

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
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/true,
                      /*regret_matching_plus=*/true) {}
  // The constructor below is used for deserialization purposes.
  CFRPlusSolver(std::shared_ptr<const Game> game, int iteration)
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/false,
                      /*regret_matching_plus=*/false, iteration) {}

 protected:
  std::string SerializeThisType() const { return "CFRPlusSolver"; }
};

std::unique_ptr<CFRPlusSolver> DeserializeCFRPlusSolver(
    const std::string& serialized, std::string delimiter = "<~>");

struct PartiallyDeserializedCFRSolver {
  PartiallyDeserializedCFRSolver(std::shared_ptr<const Game> game,
                                 std::string solver_type,
                                 std::string solver_specific_state,
                                 absl::string_view serialized_cfr_values_table)
      : game(game),
        solver_type(solver_type),
        solver_specific_state(solver_specific_state),
        serialized_cfr_values_table(serialized_cfr_values_table) {}
  std::shared_ptr<const Game> game;
  std::string solver_type;
  std::string solver_specific_state;
  absl::string_view serialized_cfr_values_table;
};

PartiallyDeserializedCFRSolver PartiallyDeserializeCFRSolver(
    const std::string& serialized);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_CFR_H_
