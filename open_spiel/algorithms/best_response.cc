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

#include "open_spiel/algorithms/best_response.h"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

TabularBestResponse::TabularBestResponse(const Game& game,
                                         Player best_responder,
                                         const Policy* policy,
                                         const float prob_cut_threshold)
    : best_responder_(best_responder),
      tabular_policy_container_(),
      policy_(policy),
      tree_(HistoryTree(game.NewInitialState(), best_responder_)),
      num_players_(game.NumPlayers()),
      prob_cut_threshold_(prob_cut_threshold),
      infosets_(GetAllInfoSets(game.NewInitialState(), best_responder, policy,
                               &tree_)),
      root_(game.NewInitialState()),
      dummy_policy_(new TabularPolicy(GetUniformPolicy(game))) {
  if (game.GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("The game must be turn-based.");
  }
}

TabularBestResponse::TabularBestResponse(
    const Game& game, Player best_responder,
    const std::unordered_map<std::string, ActionsAndProbs>& policy_table,
    const float prob_cut_threshold)
    : best_responder_(best_responder),
      tabular_policy_container_(policy_table),
      policy_(&tabular_policy_container_),
      tree_(HistoryTree(game.NewInitialState(), best_responder_)),
      num_players_(game.NumPlayers()),
      prob_cut_threshold_(prob_cut_threshold),
      infosets_(GetAllInfoSets(game.NewInitialState(), best_responder, policy_,
                               &tree_)),
      root_(game.NewInitialState()),
      dummy_policy_(new TabularPolicy(GetUniformPolicy(game))) {
  if (game.GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("The game must be turn-based.");
  }
}

double TabularBestResponse::HandleTerminalCase(const HistoryNode& node) const {
  return node.GetValue();
}

double TabularBestResponse::HandleDecisionCase(HistoryNode* node) {
  if (node == nullptr) SpielFatalError("HandleDecisionCase: node is null.");
  if (node->GetState()->CurrentPlayer() == best_responder_) {
    // If we're playing as the best responder, we look at every child node,
    // and pick the one with the highest expected utility to play.
    Action action = BestResponseAction(node->GetInfoState());
    HistoryNode* child = node->GetChild(action).second;
    if (child == nullptr) SpielFatalError("HandleDecisionCase: node is null.");
    return Value(child->GetHistory());
  }
  // If the other player is playing, then we can recursively compute the
  // expected utility of that node by looking at their policy.
  // We take child probabilities from the policy as that is what we are
  // calculating a best response to.
  ActionsAndProbs state_policy = policy_->GetStatePolicy(*node->GetState());
  if (state_policy.empty())
    SpielFatalError(absl::StrCat("InfoState ", node->GetInfoState(),
                                 " not found in policy."));
  if (state_policy.size() > node->NumChildren()) {
    int num_zeros = 0;
    for (const auto& a_and_p : state_policy) {
      if (Near(a_and_p.second, 0.)) ++num_zeros;
    }
    // We check here that the policy is valid, i.e. that it doesn't contain too
    // many (invalid) actions. This can only happen when the policy is built
    // incorrectly. If this is failing, you are building the policy wrong.
    if (state_policy.size() > node->NumChildren() + num_zeros) {
      std::vector<std::string> action_probs_str_vector;
      action_probs_str_vector.reserve(state_policy.size());
      for (const auto& action_prob : state_policy) {
        // TODO(b/127423396): Use absl::StrFormat.
        action_probs_str_vector.push_back(absl::StrCat(
            "(", action_prob.first, ", ", action_prob.second, ")"));
      }
      std::string action_probs_str =
          absl::StrJoin(action_probs_str_vector, " ");

      SpielFatalError(absl::StrCat(
          "Policies don't match in size, in state ",
          node->GetState()->HistoryString(), ".\nThe tree has '",
          node->NumChildren(), "' valid children, but ", state_policy.size(),
          " valid (action, prob) are available: [", action_probs_str, "]"));
    }
  }
  double value = 0;
  for (const auto& action : node->GetState()->LegalActions()) {
    const double prob = GetProb(state_policy, action);
    if (prob <= prob_cut_threshold_) continue;

    // We discard the probability here that's returned by GetChild as we
    // immediately load the probability for the given child from the policy.
    HistoryNode* child = node->GetChild(action).second;
    if (child == nullptr) SpielFatalError("HandleDecisionCase: node is null.");

    // Finally, we update value by the policy weighted value of the child.
    SPIEL_CHECK_GE(prob, 0);
    value += prob * Value(child->GetHistory());
  }
  return value;
}

double TabularBestResponse::HandleChanceCase(HistoryNode* node) {
  double value = 0;
  double prob_sum = 0;
  for (const auto& action : node->GetChildActions()) {
    std::pair<double, HistoryNode*> prob_and_child = node->GetChild(action);
    double prob = prob_and_child.first;
    prob_sum += prob;
    if (prob <= prob_cut_threshold_) continue;
    HistoryNode* child = prob_and_child.second;
    if (child == nullptr) SpielFatalError("Child is null.");

    // Verify that the probability is valid. This should always be true.
    SPIEL_CHECK_GE(prob, 0.);
    SPIEL_CHECK_LE(prob, 1.);
    value += prob * Value(child->GetHistory());
  }

  // Verify that the sum of the probabilities is 1, within tolerance.
  SPIEL_CHECK_FLOAT_EQ(prob_sum, 1.0);
  return value;
}

double TabularBestResponse::Value(const std::string& history) {
  auto it = value_cache_.find(history);
  if (it != value_cache_.end()) return it->second;
  HistoryNode* node = tree_.GetByHistory(history);
  if (node == nullptr) SpielFatalError("node returned is null.");
  double cache_value = 0;
  switch (node->GetType()) {
    case StateType::kTerminal: {
      cache_value = HandleTerminalCase(*node);
      break;
    }
    case StateType::kDecision: {
      cache_value = HandleDecisionCase(node);
      break;
    }
    case StateType::kChance: {
      cache_value = HandleChanceCase(node);
      break;
    }
    case StateType::kMeanField: {
      SpielFatalError("kMeanField not supported.");
    }
  }
  value_cache_[history] = cache_value;
  return value_cache_[history];
}

Action TabularBestResponse::BestResponseAction(const std::string& infostate) {
  auto it = best_response_actions_.find(infostate);
  if (it != best_response_actions_.end()) return it->second;
  std::vector<std::pair<HistoryNode*, double>> infoset = infosets_[infostate];

  Action best_action = -1;
  double best_value = std::numeric_limits<double>::lowest();
  // The legal actions are the same for all children, so we arbitrarily pick the
  // first one to get the legal actions from.
  for (const auto& action : infoset[0].first->GetChildActions()) {
    double value = 0;
    // Prob here is the counterfactual reach-weighted probability.
    for (const auto& state_and_prob : infoset) {
      if (state_and_prob.second <= prob_cut_threshold_) continue;
      HistoryNode* state_node = state_and_prob.first;
      HistoryNode* child_node = state_node->GetChild(action).second;
      SPIEL_CHECK_TRUE(child_node != nullptr);
      value += state_and_prob.second * Value(child_node->GetHistory());
    }
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  if (best_action == -1) SpielFatalError("No action was chosen.");
  best_response_actions_[infostate] = best_action;
  return best_action;
}

std::vector<Action> TabularBestResponse::BestResponseActions(
    const std::string& infostate, double tolerance) {
  std::vector<Action> best_actions;
  std::vector<std::pair<HistoryNode*, double>> infoset =
      infosets_.at(infostate);

  double best_value = std::numeric_limits<double>::lowest();
  // The legal actions are the same for all children, so we arbitrarily pick the
  // first one to get the legal actions from.
  for (const Action& action : infoset[0].first->GetChildActions()) {
    double value = 0;
    // Prob here is the counterfactual reach-weighted probability.
    for (const auto& [state_node, prob]  : infoset) {
      if (prob <= prob_cut_threshold_) continue;
      HistoryNode* child_node = state_node->GetChild(action).second;
      SPIEL_CHECK_TRUE(child_node != nullptr);
      value += prob * Value(child_node->GetHistory());
    }
    if (value > best_value + tolerance) {
      best_value = value;
      best_actions.clear();
      best_actions.push_back(action);
    } else if (value > best_value - tolerance) {
      best_actions.push_back(action);
    }
  }
  if (best_actions.empty()) SpielFatalError("No action was chosen.");
  return best_actions;
}

std::vector<std::pair<Action, double>>
TabularBestResponse::BestResponseActionValues(const std::string& infostate) {
  std::vector<std::pair<Action, double>> action_values;
  std::vector<std::pair<HistoryNode*, double>> infoset =
      infosets_.at(infostate);

  action_values.reserve(infoset[0].first->GetChildActions().size());
  for (Action action : infoset[0].first->GetChildActions()) {
    double value = 0;
    double normalizer = 0;

    // Prob here is the counterfactual reach-weighted probability.
    for (const auto& [state_node, prob] : infoset) {
      if (prob <= prob_cut_threshold_) continue;
      HistoryNode* child_node = state_node->GetChild(action).second;
      SPIEL_CHECK_TRUE(child_node != nullptr);
      value += prob * Value(child_node->GetHistory());
      normalizer += prob;
    }

    SPIEL_CHECK_GT(normalizer, 0);
    action_values.push_back({action, value / normalizer});
  }

  return action_values;
}

}  // namespace algorithms
}  // namespace open_spiel
