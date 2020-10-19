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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This file provides a vectorized implementation of CFR.
// This is intended for advanced usage and the code may not be as readable
// as a more basic algorithm. See cfr.h for a basic implementation.
//
// This code uses a preconstructed infostate trees of each player. It updates
// all infostates at a tree depth at once. While the current implementation is
// okay in efficiency (~10x faster than CFRSolver), it could be further
// improved:
//
// - Use contiguous memory blocks for storing regrets/current/cumul
//   for each action.
// - Skip subtrees that have zero reach probabilities.
// - Use stack allocation: most trees are small. If they exceed the allocated
//   limit (1024? nodes), use something like a linked list with these large
//   memory blocks.
//
// More todos:
// - Implement CFR+, Predictive CFR+, as local regret minimizers.
// - Provide custom leaf evaluation (neural net)
//
// If you decide to make contributions to this code, please open up an issue
// on github first. Thank you!

namespace open_spiel {
namespace algorithms {

// A helper struct that allows to propagate reach probs / cf values
// up and down the tree.
struct InfostateTreeValuePropagator {
  // Tree and the tree structure information. These must not change!
  // TODO: make some const kung-fu so we can't modify these after construction.
  /*const*/ std::unique_ptr<CFRTree> tree;
  /*const*/ std::vector<std::vector<double>> depth_branching;
  /*const*/ std::vector<std::vector<CFRNode*>> nodes_at_depth;

  // Mutable values to keep track of.
  std::vector<double> reach_probs;
  std::vector<double> cf_values;

  static void CollectTreeStructure(
      CFRNode* node, int depth,
      std::vector<std::vector<double>>* depth_branching,
      std::vector<std::vector<CFRNode*>>* nodes_at_depth);

 public:
  // Construct the value propagator, so we can use vectorized top-down
  // and bottom-up passes.
  InfostateTreeValuePropagator(std::unique_ptr<CFRTree> t);

  // Make a top-down pass, using the current policy stored in the tree nodes.
  // This computes the reach_probs_ buffer for storing cumulative product
  // of reach probabilities for leaf nodes.
  // The starting values at depth 1 must be provided externally.
  void TopDown();

  // Make a bottom-up pass, starting with the current cf_values stored
  // in the buffer. This loopss over all depths from the bottom.
  // The leaf values must be provided externally.
  void BottomUp();
};

class InfostateCFR {
 public:
  // Basic constructor for the whole game.
  InfostateCFR(const Game& game, int max_depth_limit = 1000);

  // Run CFR only at specific start states.
  InfostateCFR(absl::Span<const State*> start_states,
               absl::Span<const double> chance_reach_probs,
               const std::shared_ptr<Observer>& infostate_observer,
               int max_depth_limit = 1000);

  void RunSimultaneousIterations(int iterations);
  void RunAlternatingIterations(int iterations);

  void PrepareReachProbs();
  void PrepareReachProbs(Player pl);
  void EvaluateLeaves();
  void EvaluateLeaves(Player pl);

  // Similarly to CFRSolver, expose the InfoStateValuesTable.
  // However, this table has pointers to the values, not the actual values.
  std::unordered_map<std::string, CFRInfoStateValues const*>
    InfoStateValuesPtrTable() const;

  // Make sure we can get the average policy to compute expected values
  // and exploitability.
  class InfostateCFRAveragePolicy : public Policy {
    const InfostateCFR& cfr_;
    const std::unordered_map<
        std::string, CFRInfoStateValues const*> infostate_table_;
   public:
    InfostateCFRAveragePolicy(const InfostateCFR& cfr)
        : cfr_(cfr), infostate_table_(cfr_.InfoStateValuesPtrTable()) {}
    ActionsAndProbs GetStatePolicy(
        const std::string& info_state) const override;
  };
  std::shared_ptr<Policy> AveragePolicy() const {
    return std::make_shared<InfostateCFRAveragePolicy>(*this);
  }

 private:
  std::array<InfostateTreeValuePropagator, 2> propagators_;
  // Map from player 1 index (key) to player 0 (value).
  std::vector<int> terminal_permutation_;
  // Chance reach probs.
  std::vector<double> terminal_ch_reaches_;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> terminal_values_;

  void PrepareTerminals();
  double TerminalReachProbSum();
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
