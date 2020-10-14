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


#include "open_spiel/algorithms/infostate_tree_cfr.h"


namespace open_spiel {
namespace algorithms {


ActionsAndProbs InfostateCFR::InfostateCFRAveragePolicy::GetStatePolicy(
    const std::string& info_state) const {
  const CFRInfoStateValues& vs = *infostate_table_.at(info_state);
  double sum_prob = 0.0;
  for (int i = 0; i < vs.num_actions(); ++i) {
    sum_prob += vs.cumulative_policy[i];
  }

  ActionsAndProbs out;
  out.reserve(vs.num_actions());
  for (int i = 0; i < vs.num_actions(); ++i) {
    if (sum_prob > 0) {
      out.push_back({vs.legal_actions[i],
                     vs.cumulative_policy[i] / sum_prob});
    } else {
      // Return a uniform policy at this node
      out.push_back({vs.legal_actions[i],
                     vs.cumulative_policy[i] / vs.num_actions()});
    }
  }
  return out;
}

void InfostateTreeValuePropagator::TopDown() {
  SPIEL_CHECK_EQ(nodes_at_depth.size(), depth_branching.size());
  const int tree_depth = nodes_at_depth.size();
  reach_probs[0] = 1.;  // Root reach probability of the player.
  // Loop over all depths, except for the initial one.
  for (int d = 1; d < tree_depth; d++) {
    // Loop over all parents of current nodes.
    // We do it in reverse, i.e. from the last parent index to the first one.
    // As we update reach probs, we overwrite the same buffer so we lose the
    // current reach. However, because the tree is balanced and the usage
    // of the buffer only monotically grows with depth, doing it in reverse we
    // do not overwrite the current reach prob.
    SPIEL_DCHECK_EQ(depth_branching[d].size(), nodes_at_depth[d].size());
    int right_offset = nodes_at_depth[d].size();
    for (int parent_idx = nodes_at_depth[d - 1].size() - 1;
         parent_idx >= 0; parent_idx--) {
      const double current_reach = reach_probs[parent_idx];
      const int num_children = depth_branching[d - 1][parent_idx];
      right_offset -= num_children;
      CFRNode& node = *(nodes_at_depth[d - 1][parent_idx]);
      if (node.Type() == kDecisionInfostateNode) {
        const std::vector<double>& policy = node->current_policy;
        const std::vector<double>& regrets = node->cumulative_regrets;
        std::vector<double>& avg_policy = node->cumulative_policy;

        SPIEL_DCHECK_EQ(policy.size(), num_children);
        // Copy the policy and update with reach probs.
        // Update cumulative policy, as we now have the appropriate reaches.
        for (int i = 0; i < num_children; i++) {
          avg_policy[i] += policy[i] * current_reach;
        }

        for (int i = 0; i < num_children; i++) {
          reach_probs[right_offset + i] = policy[i] * current_reach;
        }

      } else {
        SPIEL_DCHECK_EQ(node.Type(), kObservationInfostateNode);
        // Copy only the reach probs.
        for (int i = 0; i < num_children; i++) {
          reach_probs[right_offset + i] = current_reach;
        }
      }
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(right_offset, 0);
  }
}
void InfostateTreeValuePropagator::BottomUp() {
  SPIEL_CHECK_EQ(nodes_at_depth.size(), depth_branching.size());
  const int tree_depth = nodes_at_depth.size();
  // Loop over all depths, except for the last one, as it is already set
  // by calling the leaf evaluation.
  for (int d = tree_depth - 2; d >= 0; d--) {
    // Loop over all parents of current nodes.
    // We do it in forward mode, i.e. from the first parent index to the last
    // one. As we update cf values, we overwrite the same buffer, so we lose
    // the children values. However, because the tree is balanced and
    // the usage of the buffer only monotically grows with depth, doing it in
    // forward we do not overwrite the parent's node cf value.
    int left_offset = 0;
    // Loop over all parents of current nodes.
    SPIEL_DCHECK_EQ(depth_branching[d].size(), nodes_at_depth[d].size());
    for (int parent_idx = 0; parent_idx < nodes_at_depth[d].size();
         parent_idx++) {
      const int num_children = depth_branching[d][parent_idx];
      CFRNode& node = *(nodes_at_depth[d][parent_idx]);
      double node_sum = 0.;
      if (node.Type() == kDecisionInfostateNode) {
        std::vector<double>& regrets = node->cumulative_regrets;
        std::vector<double>& policy = node->current_policy;
        SPIEL_DCHECK_EQ(policy.size(), num_children);
        SPIEL_DCHECK_EQ(regrets.size(), num_children);
        // Propagate child values by multiplying with current policy.
        for (int i = 0; i < num_children; i++) {
          node_sum += policy[i] * cf_values[left_offset + i];
        }
        // TODO: abstract away RM!
        // Update regrets.
        for (int i = 0; i < num_children; i++) {
          regrets[i] += cf_values[left_offset + i] - node_sum;
        }
        // Apply RM: compute current policy.
        double sum_positive_regrets = 0.;
        for (int i = 0; i < num_children; i++) {
          if (regrets[i] > 0) {
            sum_positive_regrets += regrets[i];
          }
        }
        for (int i = 0; i < num_children; ++i) {
          if (sum_positive_regrets > 0) {
            policy[i] = regrets[i] > 0
                        ? regrets[i] / sum_positive_regrets
                        : 0;
          } else {
            policy[i] = 1.0 / num_children;
          }
        }
      } else {
        SPIEL_DCHECK_EQ(node.Type(), kObservationInfostateNode);
        // Just sum the child values, no policy weighing is needed.
        for (int i = 0; i < num_children; i++) {
          node_sum += cf_values[left_offset + i];
        }
      }

      cf_values[parent_idx] = node_sum;
      left_offset += num_children;
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(left_offset, nodes_at_depth[d + 1].size());
  }
}
void
InfostateTreeValuePropagator::CollectTreeStructure(CFRNode* node, int depth,
                                                   std::vector<std::vector<
                                                       double>>* depth_branching,
                                                   std::vector<std::vector<
                                                       CFRNode*>>* nodes_at_depth) {
  // This CFR variant works only with leaf nodes being terminal nodes.
  SPIEL_CHECK_TRUE(node->NumChildren() > 0
                       || node->Type() == kTerminalInfostateNode);
  (*depth_branching)[depth].push_back(node->NumChildren());
  (*nodes_at_depth)[depth].push_back(node);

  for (CFRNode& child : *node)
    CollectTreeStructure(&child, depth + 1, depth_branching, nodes_at_depth);
}
InfostateTreeValuePropagator::InfostateTreeValuePropagator(CFRTree t)
    : tree(std::move(t)) {
  // We need to make sure that all terminals are at the same depth so that
  // we can propagate the computation of reach probs / cf values in a single
  // vector. This requirement is typically satisfied by most domains already
  // during the tree construction anyway.
  if (!tree.IsBalanced()) tree.Rebalance();

  depth_branching.resize(tree.TreeHeight() + 1);
  nodes_at_depth.resize(tree.TreeHeight() + 1);
  CollectTreeStructure(tree.MutableRoot(), 0,
                       &depth_branching, &nodes_at_depth);

  const int max_nodes_across_depths = nodes_at_depth.back().size();
  cf_values = std::vector<double>(max_nodes_across_depths);
  reach_probs = std::vector<double>(max_nodes_across_depths);
}
InfostateCFR::InfostateCFR(const Game& game, int max_depth_limit)
    : propagators_({CFRTree(game, 0, max_depth_limit),
                    CFRTree(game, 1, max_depth_limit)}) {
  PrepareTerminals();
}
InfostateCFR::InfostateCFR(absl::Span<const State*> start_states,
                           absl::Span<const double> chance_reach_probs,
                           const std::shared_ptr<Observer>& infostate_observer,
                           int max_depth_limit)
    : propagators_({CFRTree(start_states, chance_reach_probs,
                            infostate_observer, 0, max_depth_limit),
                    CFRTree(start_states, chance_reach_probs,
                            infostate_observer, 1, max_depth_limit)}) {
  PrepareTerminals();
}
void InfostateCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    propagators_[0].TopDown();
    propagators_[1].TopDown();
    SPIEL_DCHECK_TRUE(fabs(TerminalReachProbSum() - 1.0) < 1e-10);

    EvaluateLeaves();

    propagators_[0].BottomUp();
    propagators_[1].BottomUp();
    SPIEL_DCHECK_TRUE(
        fabs(propagators_[0].cf_values[0] + propagators_[1].cf_values[0])
            < 1e-10);
  }
}
void InfostateCFR::RunAlternatingIterations(int iterations) {
  // Warm up reach probs buffers.
  propagators_[0].TopDown();
  propagators_[1].TopDown();

  for (int t = 0; t < iterations; ++t) {
    for (int i = 0; i < 2; ++i) {
      propagators_[1-i].TopDown();
      EvaluateLeaves(i);
      propagators_[i].BottomUp();
    }
  }
}
void InfostateCFR::EvaluateLeaves() {
  auto& prop = propagators_;
  SPIEL_DCHECK_EQ(prop[0].cf_values.size(), prop[1].cf_values.size());
  for (int i = 0; i < prop[0].cf_values.size(); ++i) {
    const int j = terminal_permutation_[i];
    prop[0].cf_values[i] =   terminal_values_[i] * prop[1].reach_probs[j];
    prop[1].cf_values[j] = - terminal_values_[i] * prop[0].reach_probs[i];
  }
}
void InfostateCFR::EvaluateLeaves(Player pl) {
  auto& prop = propagators_;
  SPIEL_DCHECK_EQ(prop[0].cf_values.size(), prop[1].cf_values.size());
  if (pl == 0) {
    for (int i = 0; i < prop[0].cf_values.size(); ++i) {
      const int j = terminal_permutation_[i];
      prop[0].cf_values[i] =   terminal_values_[i] * prop[1].reach_probs[j];
    }
  } else {
    for (int i = 0; i < prop[0].cf_values.size(); ++i) {
      const int j = terminal_permutation_[i];
      prop[1].cf_values[j] = - terminal_values_[i] * prop[0].reach_probs[i];
    }
  }
}
std::unordered_map<std::string, CFRInfoStateValues const*>
InfostateCFR::InfoStateValuesPtrTable() const {
  std::unordered_map<std::string, CFRInfoStateValues const*> vec_ptable;
  CollectTable(propagators_[0].tree.Root(), &vec_ptable);
  CollectTable(propagators_[1].tree.Root(), &vec_ptable);
  return vec_ptable;
}
void InfostateCFR::PrepareTerminals() {
  std::array<absl::Span<CFRNode*>, 2> leaf_nodes = {
      absl::MakeSpan(propagators_[0].nodes_at_depth.back()),
      absl::MakeSpan(propagators_[1].nodes_at_depth.back())
  };
  SPIEL_CHECK_EQ(leaf_nodes[0].size(), leaf_nodes[1].size());
  const int num_terminals = leaf_nodes[0].size();
  terminal_values_.reserve(num_terminals);
  terminal_ch_reaches_.reserve(num_terminals);
  terminal_permutation_.reserve(num_terminals);

  using History = absl::Span<const Action>;
  std::map<History, int> player1_map;
  for (int i = 0; i < num_terminals; ++i) {
    player1_map[leaf_nodes[1][i]->TerminalHistory()] = i;
  }
  SPIEL_CHECK_EQ(player1_map.size(), leaf_nodes[1].size());

  for (int i = 0; i < num_terminals; ++i) {
    const CFRNode const* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const CFRNode const* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const CFRNode const* leaf = leaf_nodes[0][i];
    const double v = leaf->TerminalValue();
    const double chn = leaf->TerminalChanceReachProb();
    terminal_values_.push_back(v * chn);
    terminal_ch_reaches_.push_back(chn);
    terminal_permutation_.push_back(permutation_index);
  }
  SPIEL_DCHECK_EQ(
  // A quick check to see if the permutation is ok
  // by computing the arithmetic sum.
      std::accumulate(terminal_permutation_.begin(),
                      terminal_permutation_.end(), 0),
      num_terminals * (num_terminals - 1) / 2);
}
double InfostateCFR::TerminalReachProbSum() {
  const int num_terminals = terminal_values_.size();
  double reach_sum = 0.;
  for (int i = 0; i < num_terminals; ++i) {
    const int j = terminal_permutation_[i];
    const double leaf_reach = terminal_ch_reaches_[i]
        * propagators_[0].reach_probs[i]
        * propagators_[1].reach_probs[j];
    SPIEL_CHECK_LE(leaf_reach, 1.0);
    reach_sum += leaf_reach;
  }
  return reach_sum;
}
void InfostateCFR::CollectTable(const CFRNode& node,
                                std::unordered_map<std::string,
                                                   const CFRInfoStateValues*>* out) const {
  if (node.Type() == kDecisionInfostateNode) {
    (*out)[node.infostate_string_] = &node.values();
  }
  for (const auto& child : node) CollectTable(child, out);
}
}  // namespace algorithms
}  // namespace open_spiel
