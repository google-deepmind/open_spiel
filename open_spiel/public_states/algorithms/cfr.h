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

#ifndef OPEN_SPIEL_PUBLIC_STATES_ALGORITHMS_CFR_H_
#define OPEN_SPIEL_PUBLIC_STATES_ALGORITHMS_CFR_H_

#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/public_states/policy.h"
#include "open_spiel/public_states/public_states.h"
#include "open_spiel/spiel_utils.h"


namespace open_spiel {
namespace public_states {
namespace algorithms {

// Small epsilon typically used for fast implementation of the RM formula.
constexpr double kRmEpsilon = 1e-15;

// Structure used for public tree CFR to store relevant quantities.
// This is similar to CFRInfoStateValues, but it does it over the public tree.
struct CFRNode {
  CFRNode(std::unique_ptr<PublicState> public_state, CFRNode* parent = nullptr);
  CFRNode(const CFRNode& other);
  ~CFRNode() = default;
  CFRNode& operator=(const CFRNode& other);

  void ApplyRegretMatching();  // Fills current_policy.
  void ApplyRegretMatchingPlusReset();  // Zeroes out negative regrets.

  const std::unique_ptr<PublicState> public_state;
  CFRNode* parent;
  std::vector<std::unique_ptr<CFRNode>> children;

  // All following fields have the dimensions of:
  // [ player x private_state x private_actions ]
  std::vector<std::vector<ArrayXd>> cumulative_regrets;
  std::vector<std::vector<ArrayXd>> cumulative_policy;  // Not valid prob dist!
  std::vector<std::vector<ArrayXd>> current_policy;  // Normalized: valid dist.
};

// Calculates the average policy from the CFR values.
// If a state is not found, it returns the default policy for the
// state (or an empty policy if default_policy is nullptr).
class CFRAveragePolicyPublicStates : public PublicStatesPolicy {
 public:
  CFRAveragePolicyPublicStates(const CFRNode& root_node,
                               std::shared_ptr<Policy> default_policy);


  std::vector<ArrayXd> GetPublicStatePolicy(
      const PublicState& public_state, Player for_player) const override;
  ActionsAndProbs GetStatePolicy(
      const State& state, Player for_player) const override;

 private:
  const CFRNode* LookupPublicState(
      const CFRNode& current_node,
      const std::vector<PublicTransition>& lookup_history) const;

  const CFRNode& root_node_;
  std::shared_ptr<Policy> default_policy_;
};

// Base class supporting different flavours of the Counterfactual Regret
// Minimization (CFR) algorithm. This implementation is similar to the CFR
// algorithm for Base API, but it makes more efficient use of vectorized
// operations over the public tree.
class CFRSolverBasePublicStates {
 public:
  CFRSolverBasePublicStates(
      const GameWithPublicStates& public_game,
    bool regret_matching_plus, bool linear_averaging);
  virtual ~CFRSolverBasePublicStates() = default;

  // Performs one step of the CFR algorithm.
  virtual void RunIteration();

  // Computes the average policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::unique_ptr<Policy> AveragePolicy() const {
    return std::make_unique<CFRAveragePolicyPublicStates>(*root_node_, nullptr);
  }

 protected:
  const GameWithPublicStates& public_game_;

  // Iteration to support linear_policy.
  int iteration_ = 0;
  const std::unique_ptr<CFRNode> root_node_;

  void RunIteration(
      CFRNode* start_node, Player player,  std::vector<ReachProbs> start_probs);

  // Compute the counterfactual regrets and update the average policy
  // for the specified player.
  CfPrivValues RecursiveComputeCfRegrets(
    CFRNode* node, int alternating_player,
    std::vector<ReachProbs>& reach_probs);

 private:
  void InitializeCFRNodes(CFRNode* node);
  void RecursiveApplyRegretMatching(CFRNode* node);
  void RecursiveApplyRegretMatchingPlusReset(CFRNode* node);

  const bool regret_matching_plus_;
  const bool linear_averaging_;
};

// Standard CFR implementation.
//
// See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
class CFRPublicStatesSolver : public CFRSolverBasePublicStates {
 public:
  explicit CFRPublicStatesSolver(const GameWithPublicStates& public_game)
      : CFRSolverBasePublicStates(public_game,
                                  /*regret_matching_plus=*/false,
                                  /*linear_averaging=*/false) {}
};

// CFR+ implementation.
//
// See https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf
//
// CFR+ is CFR with the following modifications:
// - use Regret Matching+ instead of Regret Matching.
// - use alternating updates instead of simultaneous updates.
// - use linear averaging.
class CFRPlusPublicStatesSolver : public CFRSolverBasePublicStates {
 public:
  CFRPlusPublicStatesSolver(const GameWithPublicStates& public_game)
      : CFRSolverBasePublicStates(public_game,
                                  /*regret_matching_plus=*/true,
                                  /*linear_averaging=*/true) {}
};

}  // namespace algorithms
}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_ALGORITHMS_CFR_H_
