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

#include "open_spiel/algorithms/fsicfr.h"

#include <algorithm>
#include <limits>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

constexpr const int kNumPlayers = 2;

FSICFRNode::FSICFRNode() : psum(kNumPlayers, 0) {}

void FSICFRNode::AddChild(Action action, int chance_id, FSICFRNode* child) {
  children[{action, chance_id}] = child->id;
  if (std::find(child->parent_ids.begin(), child->parent_ids.end(), id) ==
      child->parent_ids.end()) {
    child->parent_ids.push_back(id);
  }
}

std::string FSICFRNode::ToString() {
  return absl::StrCat(id, " P", player, " T?", terminal, ": ", string_key);
}

void FSICFRNode::ApplyRegretMatching() {
  double pos_rsum = 0.0;
  for (int a = 0; a < legal_actions.size(); ++a) {
    pos_rsum += std::max(0.0, regrets[a]);
  }
  for (int a = 0; a < legal_actions.size(); ++a) {
    strategy[a] = pos_rsum > 0 ? std::max(0.0, regrets[a]) / pos_rsum
                               : 1.0 / legal_actions.size();
    SPIEL_CHECK_PROB(strategy[a]);
  }
}

FSICFRNode* FSICFRGraph::GetOrCreateDecisionNode(
    const std::vector<Action>& legal_actions,
    const std::string& info_state_string, Player player, int max_predecessors,
    int chance_id) {
  auto iter = string_key_to_node_id_map_.find(info_state_string);
  if (iter != string_key_to_node_id_map_.end()) {
    return &nodes_[iter->second];
  } else {
    FSICFRNode node;
    node.terminal = false;
    node.string_key = info_state_string;
    node.player = player;
    node.chance_id = chance_id;
    node.max_predecessors = max_predecessors;
    node.legal_actions = legal_actions;
    node.psum = {0.0, 0.0};
    node.strategy =
        std::vector<double>(legal_actions.size(), 1.0 / legal_actions.size());
    node.regrets =
        std::vector<double>(legal_actions.size(), 1.0 / legal_actions.size());
    node.ssum = std::vector<double>(legal_actions.size(), 0.0);
    node.id = nodes_.size();
    string_key_to_node_id_map_[info_state_string] = node.id;
    nodes_.push_back(node);
    return &nodes_[node.id];
  }
}

FSICFRNode* FSICFRGraph::GetOrCreateTerminalNode(
    const std::string& terminal_string_key, double p0_utility,
    int max_predecessors) {
  auto iter = string_key_to_node_id_map_.find(terminal_string_key);
  if (iter != string_key_to_node_id_map_.end()) {
    return &nodes_[iter->second];
  } else {
    FSICFRNode node;
    node.terminal = true;
    node.string_key = terminal_string_key;
    node.p0_utility = p0_utility;
    node.max_predecessors = max_predecessors;
    node.id = nodes_.size();
    string_key_to_node_id_map_[terminal_string_key] = node.id;
    nodes_.push_back(node);
    return &nodes_[node.id];
  }
}

void FSICFRGraph::TopSort() {
  int max_value = -1;
  int cur_value = 0;
  bool done = false;
  int num_nodes = 0;

  while (!done) {
    num_nodes = 0;
    for (int i = 0; i < nodes_.size(); ++i) {
      max_value = std::max(max_value, nodes_[i].max_predecessors);
      if (nodes_[i].max_predecessors == cur_value) {
        // std::cout << nodes_[i].max_predecessors << " "
        //           << nodes_[i].string_key << std::endl;
        ordered_ids_.push_back(i);
        num_nodes++;
      }
    }

    cur_value++;
    if (cur_value > max_value) {
      done = true;
    }
  }

  SPIEL_CHECK_EQ(nodes_.size(), ordered_ids_.size());
}

FSICFRSolver::FSICFRSolver(const Game& game, int seed,
                           const std::vector<int>& chance_outcome_ranges,
                           const FSICFRGraph* graph)
    : game_(game),
      rng_(seed),
      total_iterations_(0),
      chance_outcome_ranges_(chance_outcome_ranges),
      sampled_chance_outcomes_(game.NumPlayers()),
      graph_(graph) {}

void FSICFRSolver::RunIteration() {
  // Predetermine chance outcomes (one per player).
  for (int i = 0; i < sampled_chance_outcomes_.size(); ++i) {
    sampled_chance_outcomes_[i] =
        absl::Uniform<int>(rng_, 0, chance_outcome_ranges_[i]);
    SPIEL_CHECK_GE(sampled_chance_outcomes_[i], 0);
    SPIEL_CHECK_LT(sampled_chance_outcomes_[i], chance_outcome_ranges_[i]);
  }
  ForwardPass();
  BackwardPass();
  total_iterations_++;
}

void FSICFRSolver::RunIterations(int n) {
  for (int i = 0; i < n; ++i) {
    RunIteration();
  }
}

void FSICFRSolver::ForwardPass() {
  bool done_first = false;
  for (int idx = 0; idx < graph_->size(); ++idx) {
    int node_id = graph_->ordered_node_id(idx);
    FSICFRNode* node = graph_->GetNode(node_id);
    if (!node->terminal &&
        node->chance_id == sampled_chance_outcomes_[node->player]) {
      if (!done_first) {
        node->visits = 1;
        node->psum = {1.0, 1.0};
        done_first = true;
      }
      node->ApplyRegretMatching();
      double my_reach = node->psum[node->player];
      int opp_chance_id = sampled_chance_outcomes_[1 - node->player];
      for (int a = 0; a < node->legal_actions.size(); ++a) {
        node->ssum[a] += my_reach * node->strategy[a];
        Action action = node->legal_actions[a];
        auto iter = node->children.find({action, opp_chance_id});
        SPIEL_CHECK_TRUE(iter != node->children.end());
        int child_id = iter->second;
        FSICFRNode* child = graph_->GetNode(child_id);
        if (!child->terminal) {
          child->visits += node->visits;
          SPIEL_CHECK_GT(child->visits, 0);
          for (int p : {0, 1}) {
            child->psum[p] +=
                node->psum[p] * (node->player == p ? node->strategy[a] : 1.0);
            SPIEL_CHECK_GE(child->psum[p], 0);
          }
        }
      }
    }
  }
}

void FSICFRSolver::BackwardPass() {
  for (int idx = graph_->size() - 1; idx >= 0; --idx) {
    int node_id = graph_->ordered_node_id(idx);
    FSICFRNode* node = graph_->GetNode(node_id);
    if (!node->terminal &&
        node->chance_id == sampled_chance_outcomes_[node->player]) {
      node->v = 0;
      int opp_chance_id = sampled_chance_outcomes_[1 - node->player];
      std::vector<double> values(node->legal_actions.size(), 0);
      double opp_reach = node->psum[1 - node->player];
      for (int a = 0; a < node->legal_actions.size(); ++a) {
        Action action = node->legal_actions[a];
        auto iter = node->children.find({action, opp_chance_id});
        SPIEL_CHECK_TRUE(iter != node->children.end());
        int child_id = iter->second;
        FSICFRNode* child = graph_->GetNode(child_id);
        if (child->terminal) {
          SPIEL_CHECK_TRUE(child->p0_utility == -1 || child->p0_utility == 1);
          values[a] =
              node->player == 0 ? child->p0_utility : -child->p0_utility;
        } else {
          values[a] = node->player == child->player ? child->v : -child->v;
        }
        node->v += node->strategy[a] * values[a];
      }
      for (int a = 0; a < node->legal_actions.size(); ++a) {
        node->regrets[a] = (node->T * node->regrets[a] +
                            node->visits * opp_reach * (values[a] - node->v)) /
                           (node->T + node->visits);
      }
      node->T += node->visits;
      node->visits = 0;
      node->psum[0] = 0;
      node->psum[1] = 0;
    }
  }
}

TabularPolicy FSICFRSolver::GetAveragePolicy() const {
  TabularPolicy policy;
  for (int idx = 0; idx < graph_->size(); ++idx) {
    FSICFRNode* node = graph_->GetNode(idx);
    if (!node->terminal) {
      ActionsAndProbs state_policy;
      double denom = std::accumulate(node->ssum.begin(), node->ssum.end(), 0.0);
      SPIEL_CHECK_GE(denom, 0.0);
      for (int a = 0; a < node->legal_actions.size(); ++a) {
        Action action = node->legal_actions[a];
        double prob = denom > 0 ? node->ssum[a] / denom
                                : 1.0 / node->legal_actions.size();
        SPIEL_CHECK_PROB(prob);
        state_policy.push_back({action, prob});
      }
      policy.SetStatePolicy(node->string_key, state_policy);
    }
  }
  return policy;
}

}  // namespace algorithms
}  // namespace open_spiel
