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

#include "open_spiel/algorithms/history_tree.h"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

std::unique_ptr<HistoryNode> RecursivelyBuildGameTree(
    std::unique_ptr<State> state, Player player_id,
    absl::flat_hash_map<std::string, HistoryNode*>* state_to_node) {
  std::unique_ptr<HistoryNode> node(
      new HistoryNode(player_id, std::move(state)));
  if (state_to_node == nullptr) SpielFatalError("state_to_node is null.");
  (*state_to_node)[node->GetHistory()] = node.get();
  State* state_ptr = node->GetState();
  switch (node->GetType()) {
    case StateType::kMeanField: {
      SpielFatalError("kMeanField not supported.");
    }
    case StateType::kChance: {
      double probability_sum = 0;
      for (const auto& [outcome, prob] : state_ptr->ChanceOutcomes()) {
        std::unique_ptr<State> child = state_ptr->Child(outcome);
        if (child == nullptr) {
          SpielFatalError("Can't add child; child is null.");
        }
        probability_sum += prob;
        std::unique_ptr<HistoryNode> child_node = RecursivelyBuildGameTree(
            std::move(child), player_id, state_to_node);
        node->AddChild(outcome, {prob, std::move(child_node)});
      }
      SPIEL_CHECK_FLOAT_EQ(probability_sum, 1.0);
      break;
    }
    case StateType::kDecision: {
      for (const auto& legal_action : state_ptr->LegalActions()) {
        std::unique_ptr<State> child = state_ptr->Child(legal_action);

        // Note: The probabilities here are meaningless if state.CurrentPlayer()
        // != player_id, as we'll be getting the probabilities from the policy
        // during the call to Value. For state.CurrentPlayer() == player_id,
        // the probabilities are equal to 1. for every action as these are
        // *counter-factual* probabilities, which ignore the probability of
        // the player that we are playing as.
        node->AddChild(legal_action,
                       {1., RecursivelyBuildGameTree(
                                std::move(child), player_id, state_to_node)});
      }
      break;
    }
    case StateType::kTerminal: {
      // As we assign terminal utilities to node.value in the constructor of
      // HistoryNode, we don't have anything to do here.
      break;
    }
  }
  return node;
}

}  // namespace

HistoryNode::HistoryNode(Player player_id, std::unique_ptr<State> game_state)
    : state_(std::move(game_state)),
      history_(state_->HistoryString()),
      type_(state_->GetType()) {
  // Unless it's the opposing player's turn, we always view the game from the
  // view of player player_id.
  if (type_ == StateType::kDecision && state_->CurrentPlayer() != player_id) {
    infostate_ = state_->InformationStateString();
  } else if (type_ == StateType::kChance) {
    infostate_ = kChanceNodeInfostateString;
  } else if (type_ == StateType::kTerminal) {
    infostate_ = kTerminalNodeInfostateString;
  } else {
    infostate_ = state_->InformationStateString(player_id);
  }
  // Compute & store the legal actions so we can check that all actions we're
  // adding are legal.
  for (Action action : state_->LegalActions()) legal_actions_.insert(action);
  if (type_ == StateType::kTerminal) value_ = state_->PlayerReturn(player_id);
}

void HistoryNode::AddChild(
    Action outcome, std::pair<double, std::unique_ptr<HistoryNode>> child) {
  if (!legal_actions_.count(outcome)) SpielFatalError("Child is not legal.");
  if (child.second == nullptr) {
    SpielFatalError("Error inserting child; child is null.");
  }
  if (child.first < 0. || child.first > 1.) {
    SpielFatalError(absl::StrCat(
        "AddChild error: Probability for child must be in [0, 1], not: ",
        child.first));
  }
  child_info_[outcome] = std::move(child);
  if (child_info_.size() > legal_actions_.size()) {
    SpielFatalError("More children than legal actions.");
  }
}

std::pair<double, HistoryNode*> HistoryNode::GetChild(Action outcome) {
  auto it = child_info_.find(outcome);
  if (it == child_info_.end()) {
    SpielFatalError("Error getting child; action not found.");
  }
  // it->second.first is the probability associated with outcome, so as it is a
  // probability, it must be in [0, 1].
  SPIEL_CHECK_GE(it->second.first, 0.);
  SPIEL_CHECK_LE(it->second.first, 1.);
  std::pair<double, HistoryNode*> child =
      std::make_pair(it->second.first, it->second.second.get());
  if (child.second == nullptr) {
    SpielFatalError("Error getting child; child is null.");
  }
  return child;
}

std::vector<Action> HistoryNode::GetChildActions() const {
  std::vector<Action> actions;
  actions.reserve(child_info_.size());
  for (const auto& [action, _] : child_info_) actions.push_back(action);
  return actions;
}

HistoryNode* HistoryTree::GetByHistory(const std::string& history) {
  auto it = state_to_node_.find(history);
  if (it == state_to_node_.end()) {
    SpielFatalError(absl::StrCat("Node is null for history: '", history, "'"));
  }
  return it->second;
}

std::vector<std::string> HistoryTree::GetHistories() {
  std::vector<std::string> histories;
  histories.reserve(state_to_node_.size());
  for (const auto& [history, _] : state_to_node_) histories.push_back(history);
  return histories;
}

// Builds game tree consisting of all decision nodes for player_id.
HistoryTree::HistoryTree(std::unique_ptr<State> state, Player player_id) {
  root_ =
      RecursivelyBuildGameTree(std::move(state), player_id, &state_to_node_);
}

ActionsAndProbs GetSuccessorsWithProbs(const State& state,
                                       Player best_responder,
                                       const Policy* policy) {
  if (state.CurrentPlayer() == best_responder) {
    ActionsAndProbs state_policy;
    for (const auto& legal_action : state.LegalActions()) {
      // Counterfactual reach probabilities exclude the player's
      // actions, hence return probability 1.0 for every action.
      state_policy.push_back({legal_action, 1.});
    }
    return state_policy;
  } else if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    // Finally, we look at the policy we are finding a best response to, and
    // get our probabilities from there.
    auto state_policy = policy->GetStatePolicy(state);
    if (state_policy.empty()) {
      SpielFatalError(state.InformationStateString() + " not found in policy.");
    }
    return state_policy;
  }
}

// TODO(author1): If this is a bottleneck, it should be possible
// to pass the probabilities-so-far into the call, and get everything right
// the first time, without recursion. The recursion is simpler, however.
std::vector<std::pair<std::unique_ptr<State>, double>> DecisionNodes(
    const State& parent_state, Player best_responder, const Policy* policy) {
  // If the state is terminal, then there are no more decisions to be made,
  // so we're done.
  if (parent_state.IsTerminal()) return {};

  std::vector<std::pair<std::unique_ptr<State>, double>> states_and_probs;
  // We only consider states where the best_responder is making a decision.
  if (parent_state.CurrentPlayer() == best_responder) {
    states_and_probs.push_back({parent_state.Clone(), 1.});
  }
  ActionsAndProbs actions_and_probs =
      GetSuccessorsWithProbs(parent_state, best_responder, policy);
  for (open_spiel::Action action : parent_state.LegalActions()) {
    std::unique_ptr<State> child = parent_state.Child(action);

    // We recurse here to get the correct probabilities for all children.
    // This could probably be done in a cleaner, more performant way, but as
    // this is only done once, at the start of the exploitability calculation,
    // this is fine for now.
    std::vector<std::pair<std::unique_ptr<State>, double>> children =
        DecisionNodes(*child, best_responder, policy);
    const double policy_prob = GetProb(actions_and_probs, action);
    SPIEL_CHECK_GE(policy_prob, 0);
    for (auto& [state, prob] : children) {
      states_and_probs.push_back(
          {std::move(state),
           // We weight the child probabilities by the probability of taking
           // the action that would lead to them.
           policy_prob * prob});
    }
  }
  return states_and_probs;
}

absl::flat_hash_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
GetAllInfoSets(std::unique_ptr<State> state, Player best_responder,
               const Policy* policy, HistoryTree* tree) {
  absl::flat_hash_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
      infosets;
  // We only need decision nodes, as there's no decision to be made at chance
  // nodes (we randomly sample from the different outcomes there).
  std::vector<std::pair<std::unique_ptr<State>, double>> states_and_probs =
      DecisionNodes(*state, best_responder, policy);
  infosets.reserve(states_and_probs.size());
  for (const auto& [state, prob] : states_and_probs) {
    // We look at each decision from the perspective of the best_responder.
    std::string infostate = state->InformationStateString(best_responder);
    infosets[infostate].push_back({tree->GetByHistory(*state), prob});
  }
  return infosets;
}

}  // namespace algorithms
}  // namespace open_spiel
