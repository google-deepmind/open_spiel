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

#include "open_spiel/algorithms/tabular_best_response_mdp.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/policy.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {
namespace {
constexpr double kSolveTolerance = 1e-12;
}  // namespace

MDPNode::MDPNode(const std::string& node_key)
    : terminal_(false), total_weight_(0), children_(), value_(0),
      node_key_(node_key) {}

void MDPNode::IncTransitionWeight(Action a, MDPNode *child, double weight) {
  SPIEL_CHECK_TRUE(child != nullptr);
  children_[a][child] += weight;
}

MDP::MDP() : terminal_node_uid_(0), num_nonterminal_nodes_(0) {
  node_map_[kRootKey] = absl::make_unique<MDPNode>(std::string(kRootKey));
  node_map_[kRootKey]->add_weight(1.0);
}

MDPNode *MDP::CreateTerminalNode(const std::string &node_key) {
  ++terminal_node_uid_;
  MDPNode *terminal_node = LookupOrCreateNode(node_key, true);
  terminal_node->set_terminal(true);
  return terminal_node;
}

MDPNode *MDP::LookupOrCreateNode(const std::string &node_key, bool terminal) {
  const auto &iter = node_map_.find(node_key);
  if (iter != node_map_.end()) {
    return iter->second.get();
  } else {
    MDPNode *new_node = new MDPNode(node_key);
    node_map_[node_key].reset(new_node);
    if (!terminal) {
      num_nonterminal_nodes_++;
    }
    return new_node;
  }
}

double MDP::Solve(double tolerance, TabularPolicy *br_policy) {
  double delta = 0;

  do {
    delta = 0.0;
    for (auto &key_and_node : node_map_) {
      MDPNode *node = key_and_node.second.get();

      if (node->terminal()) {
        continue;
      }

      double max_value = -std::numeric_limits<double>::infinity();
      Action max_action = kInvalidAction;
      double node_weight = node->total_weight();
      SPIEL_CHECK_GE(node_weight, 0.0);

      // Compute Bellman value from children.
      for (const auto &action_and_child : node->children()) {
        double action_value = 0.0;
        Action action = action_and_child.first;

        for (auto &child_value : node->children()[action]) {
          MDPNode *child = child_value.first;
          double transition_weight = child_value.second;
          SPIEL_CHECK_TRUE(child != nullptr);
          double prob = transition_weight / node_weight;
          if (std::isnan(prob)) {
            // When transition_weight = node_weight = 0, set to 0
            prob = 0.0;
          }

          SPIEL_CHECK_PROB(prob);
          action_value += prob * child->value();
        }

        if (action_value > max_value) {
          max_value = action_value;
          max_action = action;
        }
      }

      SPIEL_CHECK_NE(max_action, kInvalidAction);
      delta += std::abs(node->value() - max_value);
      node->set_value(max_value);

      // Set the best response to the maximum-value action, if it's non-null.
      if (node->node_key() != kRootKey) {
        ActionsAndProbs br_state_policy;
        for (const auto &[action, child] : node->children()) {
          SetProb(&br_state_policy, action, action == max_action ? 1.0 : 0.0);
        }
        br_policy->SetStatePolicy(node->node_key(), br_state_policy);
      }
    }
  } while (delta > tolerance);

  return RootNode()->value();
}

double
TabularBestResponseMDP::OpponentReach(const std::vector<double>& reach_probs,
                                      Player p) const {
  double product = 1.0;
  for (int i = 0; i < reach_probs.size(); i++) {
    if (p != i) {
      product *= reach_probs[i];
    }
  }
  return product;
}

void TabularBestResponseMDP::BuildMDPs(
    const State &state, const std::vector<double>& reach_probs,
    const std::vector<MDPNode*>& parent_nodes,
    const std::vector<Action>& parent_actions, Player only_for_player) {
  if (state.IsTerminal()) {
    std::vector<double> terminal_values = state.Returns();
    for (Player p = 0; p < game_.NumPlayers(); ++p) {
      if (only_for_player == kInvalidPlayer || only_for_player == p) {
        std::string node_key = state.ToString();
        MDPNode *node = mdps_.at(p)->CreateTerminalNode(node_key);
        node->set_value(terminal_values[p]);
        double opponent_reach = OpponentReach(reach_probs, p);
        SPIEL_CHECK_GE(opponent_reach, 0.0);
        SPIEL_CHECK_LE(opponent_reach, 1.0);
        // Following line is not actually necessary because the weight of a leaf
        // is never in a denominator for a transition probability, but we
        // include it to keep the semantics of the values consistent across the
        // ISMDP.
        node->add_weight(opponent_reach);
        MDPNode *parent_node = parent_nodes[p];
        SPIEL_CHECK_TRUE(parent_node != nullptr);
        parent_node->IncTransitionWeight(parent_actions[p], node,
                                         opponent_reach);
      }
    }
  } else if (state.IsChanceNode()) {
    ActionsAndProbs outcomes_and_probs = state.ChanceOutcomes();
    for (const auto &[outcome, prob] : outcomes_and_probs) {
      std::unique_ptr<State> state_copy = state.Clone();
      state_copy->ApplyAction(outcome);
      std::vector<double> new_reach_probs = reach_probs;
      // Chance prob is at the end of the vector.
      new_reach_probs[game_.NumPlayers()] *= prob;
      BuildMDPs(*state_copy, new_reach_probs, parent_nodes, parent_actions,
                only_for_player);
    }
  } else if (state.IsSimultaneousNode()) {
    // Several nodes are created: one for each player as the maximizer.
    std::vector<std::string> node_keys(num_players_);
    std::vector<MDPNode*> nodes(num_players_, nullptr);
    std::vector<double> opponent_reaches(num_players_, 1.0);
    std::vector<ActionsAndProbs> fixed_state_policies(num_players_);

    for (Player player = 0; player < num_players_; ++player) {
      if (only_for_player == kInvalidPlayer || only_for_player == player) {
        node_keys[player] = GetNodeKey(state, player);
        nodes[player] = mdps_.at(player)->LookupOrCreateNode(node_keys[player]);
        opponent_reaches[player] = OpponentReach(reach_probs, player);

        SPIEL_CHECK_GE(opponent_reaches[player], 0.0);
        SPIEL_CHECK_LE(opponent_reaches[player], 1.0);
        nodes[player]->add_weight(opponent_reaches[player]);

        MDPNode* parent_node = parent_nodes[player];
        SPIEL_CHECK_TRUE(parent_node != nullptr);
        parent_node->IncTransitionWeight(parent_actions[player], nodes[player],
                                         opponent_reaches[player]);
      }

      if (only_for_player == kInvalidPlayer || only_for_player != player) {
        fixed_state_policies[player] =
            fixed_policy_.GetStatePolicy(state, player);
      }
    }

    // Traverse over the list of joint actions. For each one, first deconstruct
    // the actions, and then recurse once for each player as the maximizer with
    // the others as the fixed policies.
    const auto& sim_move_state = down_cast<const SimMoveState&>(state);
    for (Action joint_action : state.LegalActions()) {
      std::vector<Action> actions =
          sim_move_state.FlatJointActionToActions(joint_action);

      std::unique_ptr<State> state_copy = state.Clone();
      state_copy->ApplyAction(joint_action);

      std::vector<double> new_reach_probs = reach_probs;
      std::vector<MDPNode*> new_parent_nodes = parent_nodes;
      std::vector<Action> new_parent_actions = parent_actions;
      for (Player player = 0; player < num_players_; ++player) {
        if (only_for_player == kInvalidPlayer || only_for_player != player) {
          double action_prob = GetProb(fixed_state_policies[player],
                                       actions[player]);
          SPIEL_CHECK_PROB(action_prob);
          new_reach_probs[player] *= action_prob;
        }

        if (only_for_player == kInvalidPlayer || only_for_player == player) {
          new_parent_nodes[player] = nodes[player];
        }

        new_parent_actions[player] = actions[player];
      }

      BuildMDPs(*state_copy, new_reach_probs, new_parent_nodes,
                new_parent_actions, only_for_player);
    }
  } else {
    // Normal decisions node.
    std::vector<Action> legal_actions = state.LegalActions();
    Player player = state.CurrentPlayer();
    ActionsAndProbs state_policy;  // Fixed joint policy we're responding to.
    MDPNode* node = nullptr;

    // Check to see if we need to build this node.
    if (only_for_player == kInvalidPlayer || only_for_player == player) {
      std::string node_key = GetNodeKey(state, player);

      node = mdps_.at(player)->LookupOrCreateNode(node_key);
      double opponent_reach = OpponentReach(reach_probs, player);

      SPIEL_CHECK_GE(opponent_reach, 0.0);
      SPIEL_CHECK_LE(opponent_reach, 1.0);
      node->add_weight(opponent_reach);
      MDPNode *parent_node = parent_nodes[player];
      SPIEL_CHECK_TRUE(parent_node != nullptr);
      parent_node->IncTransitionWeight(parent_actions[player], node,
                                       opponent_reach);
    }

    // Get the fixed policy all the time if building all MDPs, or only at
    // opponent nodes otherwise
    if (only_for_player == kInvalidPlayer || only_for_player != player) {
        state_policy = fixed_policy_.GetStatePolicy(state);
    }

    for (Action action : legal_actions) {
      std::unique_ptr<State> state_copy = state.Clone();
      state_copy->ApplyAction(action);

      std::vector<double> new_reach_probs = reach_probs;
      std::vector<MDPNode*> new_parent_nodes = parent_nodes;

      // If building all MDPs at once, modify reach probs in all cases.
      // Otherwise, only at opponent nodes.
      if (only_for_player == kInvalidPlayer || only_for_player != player) {
        double action_prob = GetProb(state_policy, action);
        SPIEL_CHECK_PROB(action_prob);
        new_reach_probs[player] *= action_prob;
      }

      // If building all MDPs at once, modify parent nodes for that MDP.
      // Otherwise, only do it for the player we're building the MDP for.
      if (only_for_player == kInvalidPlayer || only_for_player == player) {
        new_parent_nodes[player] = node;
      }

      std::vector<Action> new_parent_actions = parent_actions;
      new_parent_actions[player] = action;

      BuildMDPs(*state_copy, new_reach_probs, new_parent_nodes,
                new_parent_actions, only_for_player);
    }
  }
}

std::string TabularBestResponseMDP::GetNodeKey(const State &state,
                                               Player player) const {
  switch (game_.GetType().information) {
    case GameType::Information::kImperfectInformation:
    case GameType::Information::kOneShot:
      return state.InformationStateString(player);
    case GameType::Information::kPerfectInformation:
      return state.ObservationString(player);
    default:
      SpielFatalError("Information type not supported.");
  }
}

TabularBestResponseMDP::TabularBestResponseMDP(const Game &game,
                                               const Policy &fixed_policy)
    : game_(game), fixed_policy_(fixed_policy),
      num_players_(game.NumPlayers()) {}

int TabularBestResponseMDP::TotalNumNonterminals() const {
  int total_num_nonterminals = 0;
  for (Player p = 0; p < num_players_; ++p) {
    total_num_nonterminals += mdps_[p]->NumNonTerminalNodes();
  }
  return total_num_nonterminals;
}

int TabularBestResponseMDP::TotalSize() const {
  int total_size = 0;
  for (Player p = 0; p < num_players_; ++p) {
    total_size += mdps_[p]->TotalSize();
  }
  return total_size;
}

TabularBestResponseMDPInfo TabularBestResponseMDP::ComputeBestResponses() {
  TabularBestResponseMDPInfo br_info(num_players_);

  // Initialize IS-MDPs for each player, if necessary.
  if (mdps_.empty()) {
    for (Player p = 0; p < num_players_; p++) {
      mdps_.push_back(absl::make_unique<MDP>());
    }
  }

  std::vector<MDPNode*> parent_nodes;
  parent_nodes.reserve(num_players_);
  for (Player p = 0; p < num_players_; p++) {
    parent_nodes.push_back(mdps_[p]->RootNode());
  }
  std::vector<double> reach_probs(num_players_ + 1, 1.0);  // include chance.
  std::vector<Action> parent_actions(num_players_, 0);

  std::unique_ptr<State> initial_state = game_.NewInitialState();
  BuildMDPs(*initial_state, reach_probs, parent_nodes, parent_actions);

  for (Player p = 0; p < num_players_; p++) {
    br_info.br_values[p] =
        mdps_[p]->Solve(kSolveTolerance, &br_info.br_policies[p]);
  }

  return br_info;
}

TabularBestResponseMDPInfo
TabularBestResponseMDP::ComputeBestResponse(Player max_player) {
  TabularBestResponseMDPInfo br_info(num_players_);

  if (mdps_.empty()) {
    mdps_.resize(num_players_);
    mdps_[max_player] = absl::make_unique<MDP>();
  }

  std::vector<MDPNode*> parent_nodes(num_players_, nullptr);
  parent_nodes[max_player] = mdps_[max_player]->RootNode();
  std::vector<double> reach_probs(num_players_ + 1, 1.0);  // include chance.
  std::vector<Action> parent_actions(num_players_, 0);

  std::unique_ptr<State> initial_state = game_.NewInitialState();
  BuildMDPs(*initial_state, reach_probs, parent_nodes, parent_actions,
            max_player);

  br_info.br_values[max_player] =
        mdps_[max_player]->Solve(kSolveTolerance,
                                 &br_info.br_policies[max_player]);
  return br_info;
}

TabularBestResponseMDPInfo TabularBestResponseMDP::NashConv() {
  TabularBestResponseMDPInfo br_info = ComputeBestResponses();
  std::unique_ptr<State> state = game_.NewInitialState();
  br_info.on_policy_values =
      ExpectedReturns(*state, fixed_policy_,
                      /*depth_limit*/ -1, /*use_infostate_get_policy*/ false);
  for (Player p = 0; p < num_players_; ++p) {
    br_info.deviation_incentives[p] =
        br_info.br_values[p] - br_info.on_policy_values[p];
    br_info.nash_conv += br_info.deviation_incentives[p];
  }
  return br_info;
}

TabularBestResponseMDPInfo TabularBestResponseMDP::Exploitability() {
  SPIEL_CHECK_TRUE(game_.GetType().utility == GameType::Utility::kZeroSum ||
                   game_.GetType().utility == GameType::Utility::kConstantSum);
  TabularBestResponseMDPInfo br_info = ComputeBestResponses();
  br_info.nash_conv = absl::c_accumulate(br_info.br_values, 0.0);
  br_info.exploitability =
      (br_info.nash_conv - game_.UtilitySum()) / num_players_;
  return br_info;
}

}  // namespace algorithms
}  // namespace open_spiel
