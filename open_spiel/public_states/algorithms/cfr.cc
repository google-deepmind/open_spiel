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

#include "open_spiel/public_states/algorithms/cfr.h"
#include <memory>
#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/fog/observation_history.h"
#include "open_spiel/public_states/public_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/functional.h"

namespace open_spiel {
namespace public_states {
namespace algorithms {

// -----------------------------------------------------------------------------
// CFRNode : a wrapper for public state and CFR-related values.
// -----------------------------------------------------------------------------

CFRNode::CFRNode(
    std::unique_ptr<PublicState> state, CFRNode* parent_node)
    : public_state(std::move(state)), parent(parent_node) {
  // We need to store regrets / policies only for public states
  // where a player acts.
  if (!public_state->IsPlayer()) return;

  const int num_players = public_state->GetPublicGame()->NumPlayers();
  const std::vector<Player> acting_players = public_state->ActingPlayers();
  const std::vector<int> privates_per_player =
      public_state->NumDistinctPrivateInformations();

  cumulative_regrets.resize(num_players);
  cumulative_policy.resize(num_players);
  current_policy.resize(num_players);

  for (const auto& acting_player : acting_players) {
    const int num_privates = privates_per_player[acting_player];
    const std::vector<int> num_actions =
        public_state->CountPrivateActions(acting_player);
    SPIEL_CHECK_EQ(num_actions.size(), num_privates);

    cumulative_regrets[acting_player].reserve(num_privates);
    cumulative_policy[acting_player].reserve(num_privates);
    current_policy[acting_player].reserve(num_privates);
    for (int i = 0; i < num_privates; i++) {
      cumulative_regrets[acting_player].push_back(
          ArrayXd::Zero(num_actions[i]));
      cumulative_policy[acting_player].push_back(
          ArrayXd::Zero(num_actions[i]));
      current_policy[acting_player].push_back(
          ArrayXd::Constant(num_actions[i], 1. / num_actions[i]));
    }
  }
}

CFRNode::CFRNode(const CFRNode& other)
    : public_state(other.public_state->Clone()), parent(other.parent) {
  for (const auto& other_child : other.children) {
    children.push_back(std::make_unique<CFRNode>(*other_child));
  }
}

CFRNode& CFRNode::operator=(const CFRNode& other) {
  if (this != &other) *this = CFRNode(other);
  return *this;
}

void CFRNode::ApplyRegretMatching() {
  const int num_players = public_state->GetPublicGame()->NumPlayers();
  SPIEL_DCHECK_EQ(cumulative_regrets.size(), num_players);
  SPIEL_DCHECK_EQ(cumulative_policy.size(), num_players);
  SPIEL_DCHECK_EQ(current_policy.size(), num_players);

  for (int player = 0; player < num_players; player++) {
    const int num_privates = cumulative_regrets[player].size();
    for (int pr = 0; pr < num_privates; pr++) {
      SPIEL_DCHECK_TRUE(public_state->IsPlayerActing(player));
      SPIEL_DCHECK_EQ(cumulative_regrets[player].size(), num_privates);
      SPIEL_DCHECK_EQ(cumulative_policy[player].size(), num_privates);
      SPIEL_DCHECK_EQ(current_policy[player].size(), num_privates);

      double positive_regret_sum =
          cumulative_regrets[player][pr].max(kRmEpsilon).sum();
      current_policy[player][pr] =
          cumulative_regrets[player][pr].max(kRmEpsilon) / positive_regret_sum;
    }
  }
}

void CFRNode::ApplyRegretMatchingPlusReset() {
  const int num_players = public_state->GetPublicGame()->NumPlayers();
  for (int player = 0; player < num_players; player++) {
    const int num_privates = cumulative_regrets[player].size();
    for (int pr = 0; pr < num_privates; pr++) {
      cumulative_regrets[player][pr] = cumulative_regrets[player][pr].max(0.);
    }
  }
}


// -----------------------------------------------------------------------------
// CFRAveragePolicyPublicStates
// -----------------------------------------------------------------------------

CFRAveragePolicyPublicStates::CFRAveragePolicyPublicStates(
    const CFRNode& root_node, std::shared_ptr<Policy> default_policy)
    : root_node_(root_node), default_policy_(default_policy) {}

const CFRNode* CFRAveragePolicyPublicStates::LookupPublicState(
    const CFRNode& current_node,
    const std::vector<PublicTransition>& lookup_history) const {
  const std::vector<PublicTransition>& current_history =
      current_node.public_state->GetPublicObservationHistory();

  if (lookup_history.size() == current_history.size()) {
    SPIEL_CHECK_EQ(lookup_history, current_history);
    return &current_node;
  }

  SPIEL_CHECK_LT(current_history.size(), lookup_history.size());
  // Check if the current node has some children that lead to lookup state.
  for (const auto& child : current_node.children) {
     const std::vector<PublicTransition>& child_history =
      child->public_state->GetPublicObservationHistory();
    if (child_history.back() == lookup_history.at(child_history.size() - 1)) {
      return LookupPublicState(*child, lookup_history);
    }
  }
  return nullptr;
}

std::vector<ArrayXd> CFRAveragePolicyPublicStates::GetPublicStatePolicy(
    const PublicState& public_state, Player for_player) const {
  SPIEL_CHECK_TRUE(public_state.IsPlayerActing(for_player));
  const CFRNode* node = LookupPublicState(
      root_node_, public_state.GetPublicObservationHistory());
  if (!node) return {};

  const std::vector<ArrayXd>& cumul_policy =
      node->cumulative_policy[for_player];
  const int num_privates = cumul_policy.size();
  std::vector<ArrayXd> avg_policy(num_privates);
  for (int pr = 0; pr < num_privates; pr++) {
    avg_policy[pr] = cumul_policy[pr] / cumul_policy[pr].sum();
  }
  return avg_policy;
}

ActionsAndProbs CFRAveragePolicyPublicStates::GetStatePolicy(
    const State& state, Player for_player) const {
  SPIEL_CHECK_TRUE(state.IsPlayerNode());
  SPIEL_CHECK_TRUE(state.IsPlayerActing(for_player));
  PublicObservationHistory public_history(state);
  const CFRNode* node = LookupPublicState(
      root_node_, public_history.History());
  if (!node) {
     if (default_policy_) {
       return default_policy_->GetStatePolicy(state);
     } else {
       return {};
     }
  }

  SPIEL_CHECK_TRUE(node->public_state->IsPlayer());
  SPIEL_CHECK_TRUE(node->public_state->IsPlayerActing(for_player));
  const std::unique_ptr<PrivateInformation> private_info =
      node->public_state->GetPrivateInformation(state, for_player);
  const int network_index = private_info->NetworkIndex();
  const ArrayXd& cumul_policy =
      node->cumulative_policy.at(for_player).at(network_index)
      + kRmEpsilon;  // Add a small value for easier normalization.
  const std::vector<Action>& actions = state.LegalActions();
  SPIEL_CHECK_EQ(actions.size(), cumul_policy.size());

  ActionsAndProbs policy;
  policy.reserve(actions.size());
  Zip(actions.begin(), actions.end(), cumul_policy.data(), policy);
  NormalizePolicy(&policy);

  return policy;
}

// -----------------------------------------------------------------------------
// CFRSolverBasePublicStates
// -----------------------------------------------------------------------------


CFRSolverBasePublicStates::CFRSolverBasePublicStates(
    const GameWithPublicStates& public_game,
    bool regret_matching_plus,
    bool linear_averaging)
    : public_game_(public_game),
      root_node_(std::make_unique<CFRNode>(
          public_game_.NewInitialPublicState())),
      regret_matching_plus_(regret_matching_plus),
      linear_averaging_(linear_averaging) {
  InitializeCFRNodes(root_node_.get());
}

void CFRSolverBasePublicStates::InitializeCFRNodes(CFRNode* node) {
  const PublicState& public_state = *node->public_state;
  for (const auto& transition : public_state.LegalTransitions()) {
    node->children.push_back(std::make_unique<CFRNode>(
        public_state.Child(transition), node));
    InitializeCFRNodes(node->children.back().get());
  }
}

void CFRSolverBasePublicStates::RunIteration() {
  ++iteration_;
  for (int player = 0; player < public_game_.NumPlayers(); player++) {
    RunIteration(root_node_.get(), player, public_game_.NewInitialReachProbs());
  }
}

void CFRSolverBasePublicStates::RunIteration(
    CFRNode* start_node, Player player, std::vector<ReachProbs> start_probs) {
  RecursiveComputeCfRegrets(start_node, player, start_probs);
  if (regret_matching_plus_) {
    RecursiveApplyRegretMatchingPlusReset(start_node);
  }
  RecursiveApplyRegretMatching(start_node);
}

void CFRSolverBasePublicStates::RecursiveApplyRegretMatching(CFRNode* node) {
  if (node->public_state->IsPlayer()) node->ApplyRegretMatching();
  for (int i = 0; i < node->children.size(); i++) {
    RecursiveApplyRegretMatching(node->children[i].get());
  }
}

void CFRSolverBasePublicStates::RecursiveApplyRegretMatchingPlusReset(
    CFRNode* node) {
  if (node->public_state->IsPlayer()) node->ApplyRegretMatchingPlusReset();
  for (int i = 0; i < node->children.size(); i++) {
    RecursiveApplyRegretMatchingPlusReset(node->children[i].get());
  }
}

CfPrivValues CFRSolverBasePublicStates::RecursiveComputeCfRegrets(
    CFRNode* node, int alternating_player,
    std::vector<ReachProbs>& reach_probs) {
  const PublicState& current_state = *node->public_state;

  if (current_state.IsTerminal()) {
    SPIEL_CHECK_TRUE(node->cumulative_regrets.empty());
    SPIEL_CHECK_TRUE(node->cumulative_policy.empty());
    SPIEL_CHECK_TRUE(node->current_policy.empty());
    SPIEL_CHECK_TRUE(node->children.empty());
    return current_state.TerminalCfValues(reach_probs, alternating_player);
  }

  if (current_state.IsChance()) {
    SPIEL_CHECK_TRUE(node->cumulative_regrets.empty());
    SPIEL_CHECK_TRUE(node->cumulative_policy.empty());
    SPIEL_CHECK_TRUE(node->current_policy.empty());
    std::vector<CfPrivValues> children_values;
    children_values.reserve(node->children.size());
    for (int i = 0; i < node->children.size(); ++i) {
      std::unique_ptr<CFRNode>& child = node->children[i];
      const PublicState& child_state = *child->public_state;

      // Compute reach probabilities of each player.
      std::vector<ReachProbs> child_reach;
      child_reach.reserve(public_game_.NumPlayers());
      for (int player = 0; player < public_game_.NumPlayers(); player++) {
        child_reach.push_back(
            // Chance states do not use players' strategies,
            // so we do not need to supply them.
            current_state.ComputeReachProbs(
                /*transition=*/child_state.LastTransition(),
                /*strategy=*/{},
                reach_probs[player]));
      }

      children_values.push_back(RecursiveComputeCfRegrets(
          child.get(), alternating_player, child_reach));
    }

    std::vector<CfActionValues> action_values =
        current_state.ComputeCfActionValues(children_values);

    // Chance states do not use players' strategies,
    // so we do not need to supply them.
    return current_state.ComputeCfPrivValues(
        action_values, /*children_policy=*/{});
  }

  SPIEL_CHECK_TRUE(current_state.IsPlayer());
  SPIEL_CHECK_EQ(node->cumulative_regrets.size(), public_game_.NumPlayers());
  SPIEL_CHECK_EQ(node->cumulative_policy.size(), public_game_.NumPlayers());
  SPIEL_CHECK_EQ(node->current_policy.size(), public_game_.NumPlayers());

  std::vector<CfPrivValues> children_values;
  children_values.reserve(node->children.size());
  for (int i = 0; i < node->children.size(); ++i) {
    std::unique_ptr<CFRNode>& child = node->children[i];
    const PublicState& child_state = *child->public_state;

    // Compute reach probabilities of each player.
    std::vector<ReachProbs> child_reach;
    child_reach.reserve(public_game_.NumPlayers());
    for (int player = 0; player < public_game_.NumPlayers(); player++) {
      SPIEL_CHECK_TRUE(!current_state.IsPlayerActing(player)
                       || !node->current_policy[player].empty());
      child_reach.push_back(
          current_state.ComputeReachProbs(
              /*transition=*/child_state.LastTransition(),
              /*strategy=*/node->current_policy[player],
              reach_probs[player]));
    }

    children_values.push_back(RecursiveComputeCfRegrets(
        child.get(), alternating_player, child_reach));
  }

  std::vector<CfActionValues> action_values =
      current_state.ComputeCfActionValues(children_values);

  CfPrivValues current_values = current_state.ComputeCfPrivValues(
      /*children_values=*/action_values,
      /*children_policy=*/node->current_policy[alternating_player]);

  SPIEL_CHECK_EQ(action_values.size(), current_values.cfvs.size());

  if (!current_state.IsPlayerActing(alternating_player)) {
    return current_values;
  }

  for (int i = 0; i < current_values.cfvs.size(); ++i) {
    const ArrayXd rm_regret = action_values[i].cfavs - current_values.cfvs[i];

    // Update regrets
    node->cumulative_regrets[alternating_player][i] += rm_regret;

    // Update average policy.
    SPIEL_CHECK_EQ(reach_probs[alternating_player].probs.size(),
                   node->current_policy[alternating_player].size());
    const double private_reach_prob = reach_probs[alternating_player].probs[i];
    const ArrayXd policy_update =  node->current_policy[alternating_player][i]
        * private_reach_prob;
    if (linear_averaging_) {
        node->cumulative_policy[alternating_player][i] +=
            iteration_ * policy_update;
    } else {
        node->cumulative_policy[alternating_player][i] += policy_update;
    }
  }
  return current_values;
}

}  // namespace algorithms
}  // namespace public_states
}  // namespace open_spiel
