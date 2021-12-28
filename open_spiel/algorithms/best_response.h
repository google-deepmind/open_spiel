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

#ifndef OPEN_SPIEL_ALGORITHMS_BEST_RESPONSE_H_
#define OPEN_SPIEL_ALGORITHMS_BEST_RESPONSE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// Calculates the best response to every state in the game against the given
// policy, where the best responder plays as player_id.
// This only works for two player, zero- or constant-sum sequential games, and
// raises a SpielFatalError if an incompatible game is passed to it.
//
// This implementation requires that InformationStateString for the game has
// perfect recall. Otherwise, the algorithm will still run, but the value
// returned will be wrong.
//
// A partially computed best-response can be computed when using a
// prob_cut_threshold >= 0.
class TabularBestResponse {
 public:
  TabularBestResponse(const Game& game, Player best_responder,
                      const Policy* policy,
                      const float prob_cut_threshold = -1.0);
  TabularBestResponse(
      const Game& game, Player best_responder,
      const std::unordered_map<std::string, ActionsAndProbs>& policy_table,
      const float prob_cut_threshold = -1.0);

  TabularBestResponse(TabularBestResponse&&) = default;

  // Returns the action that maximizes utility for the agent at the given
  // infostate. The infostate must correspond to a decision node for
  // best_responder.
  Action BestResponseAction(const std::string& infostate);
  Action BestResponseAction(const State& state) {
    SPIEL_CHECK_EQ(state.CurrentPlayer(), best_responder_);
    return BestResponseAction(state.InformationStateString(best_responder_));
  }

  // Returns all the actions that maximize utility for the agent at the given
  // infostate. The infostate must correspond to a decision node for
  // best_responder.
  std::vector<Action> BestResponseActions(const std::string& infostate,
                                          double tolerance);
  std::vector<Action> BestResponseActions(const State& state,
                                          double tolerance) {
    SPIEL_CHECK_EQ(state.CurrentPlayer(), best_responder_);
    return BestResponseActions(state.InformationStateString(best_responder_),
                               tolerance);
  }

  // Returns the values of all actions at this info state. The infostate must
  // correspond to a decision node for best_responder.
  std::vector<std::pair<Action, double>> BestResponseActionValues(
      const std::string& infostate);
  std::vector<std::pair<Action, double>> BestResponseActionValues(
      const State& state) {
    SPIEL_CHECK_EQ(state.CurrentPlayer(), best_responder_);
    return BestResponseActionValues(
        state.InformationStateString(best_responder_));
  }

  // Returns a map of infostates to best responses, for all information states
  // that have been calculated so far. If no best responses have been
  // calculated, then we calculate them for every state in the game.
  // When two actions have the same value, we
  // return the action with the lowest number (as an int).
  std::unordered_map<std::string, Action> GetBestResponseActions() {
    // If the best_response_actions_ cache is empty, we fill it by calculating
    // all best responses, starting at the root.
    if (best_response_actions_.empty()) Value(*root_);
    return best_response_actions_;
  }

  // Returns the computed best response as a policy object.
  TabularPolicy GetBestResponsePolicy() {
    SPIEL_CHECK_TRUE(dummy_policy_ != nullptr);
    return TabularPolicy(*dummy_policy_, GetBestResponseActions());
  }

  // Returns the expected utility for best_responder when playing the game
  // beginning at history.
  double Value(const std::string& history);
  double Value(const State& state) { return Value(state.HistoryString()); }

  // Changes the policy that we are calculating a best response to. This is
  // useful as a large amount of the data structures can be reused, causing
  // the calculation to be quicker than if we had to re-initialize the class.
  void SetPolicy(const Policy* policy) {
    policy_ = policy;
    value_cache_.clear();
    best_response_actions_.clear();
    // TODO(author1): Replace this with something that traverses the tree
    // and rebuilds the probabilities.
    infosets_ =
        GetAllInfoSets(root_->Clone(), best_responder_, policy_, &tree_);
  }

  // Set the policy given a policy table. This stores the table internally.
  void SetPolicy(
      const std::unordered_map<std::string, ActionsAndProbs>& policy_table) {
    tabular_policy_container_ = TabularPolicy(policy_table);
    SetPolicy(&tabular_policy_container_);
  }

 private:
  // For chance nodes, we recursively calculate the value of each child node,
  // and weight them by the probability of reaching each child.
  double HandleChanceCase(HistoryNode* node);

  // Calculates the value of the HistoryNode when we have to make a decision.
  // Does this by calculating the value of each possible child node and then
  // setting the value of the current node equal to the maximum (as we can just
  // choose the best child).
  double HandleDecisionCase(HistoryNode* node);

  // Calculates the value of the HistoryNode when the node is a terminal node.
  // Conveniently, the game tells us the value of every terminal node, so we
  // have nothing to do.
  double HandleTerminalCase(const HistoryNode& node) const;

  Player best_responder_;

  // Used to store a specific policy if not passed in from the caller.
  TabularPolicy tabular_policy_container_;

  // The actual policy that we are computing a best response to.
  const Policy* policy_;

  HistoryTree tree_;
  int num_players_;

  // The probability tolerance for truncating value estimation.
  float prob_cut_threshold_;

  // Maps infoset strings (from the State::InformationState method) to
  // the HistoryNodes that represent all histories with
  // the same information state, along with the counter-factual probability of
  // doing so. If the information state is a chance node, the probability comes
  // from the State::ChanceOutcomes method. If the information state is a
  // decision node for best_responder, the probability is one, following the
  // definition of counter-factual probability. Finally, if the information
  // state is a decision node for a player other than best_responder, the
  // probabilities come from their policy (i.e. policy_).
  absl::flat_hash_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
      infosets_;

  // Caches all best responses calculated so far (for each infostate).
  std::unordered_map<std::string, Action> best_response_actions_;

  // Caches all values calculated so far (for each history).
  std::unordered_map<std::string, double> value_cache_;
  std::unique_ptr<State> root_;

  // Keep a cache of an empty policy to avoid recomputing it.
  std::unique_ptr<TabularPolicy> dummy_policy_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BEST_RESPONSE_H_
