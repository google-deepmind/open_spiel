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

#include "open_spiel/algorithms/infostate_tree.h"

#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/action_view.h"

namespace open_spiel {
namespace algorithms {

using internal::kUndefinedNodeId;

InfostateNode::InfostateNode(const InfostateTree& tree, InfostateNode* parent,
                             int incoming_index, InfostateNodeType type,
                             const std::string& infostate_string,
                             double terminal_utility,
                             double terminal_ch_reach_prob, size_t depth,
                             std::vector<Action> legal_actions,
                             std::vector<Action> terminal_history)
    : tree_(tree),
      parent_(parent),
      incoming_index_(incoming_index),
      type_(type),
      infostate_string_(infostate_string),
      terminal_utility_(terminal_utility),
      terminal_chn_reach_prob_(terminal_ch_reach_prob),
      depth_(depth),
      legal_actions_(std::move(legal_actions)),
      terminal_history_(std::move(terminal_history)) {
  // Implications for kTerminalNode
  SPIEL_DCHECK_TRUE(type_ != kTerminalInfostateNode || parent_);
  // Implications for kDecisionNode
  SPIEL_DCHECK_TRUE(type_ != kDecisionInfostateNode || parent_);
  // Implications for kObservationNode
  SPIEL_DCHECK_TRUE(!(type_ == kObservationInfostateNode && parent_ &&
                      parent_->type() == kDecisionInfostateNode) ||
                    (incoming_index_ >= 0 &&
                     incoming_index_ < parent_->legal_actions().size()));
}

InfostateNode* InfostateNode::AddChild(std::unique_ptr<InfostateNode> child) {
  SPIEL_CHECK_EQ(child->parent_, this);
  children_.push_back(std::move(child));
  return children_.back().get();
}

InfostateNode* InfostateNode::GetChild(
    const std::string& infostate_string) const {
  for (const std::unique_ptr<InfostateNode>& child : children_) {
    if (child->infostate_string() == infostate_string) return child.get();
  }
  return nullptr;
}

std::ostream& InfostateNode::operator<<(std::ostream& os) const {
  if (!parent_) return os << 'x';
  return os << parent_ << ',' << incoming_index_;
}

std::string InfostateNode::MakeCertificate() const {
  if (type_ == kTerminalInfostateNode) return "{}";

  std::vector<std::string> certificates;
  for (InfostateNode* child : child_iterator()) {
    certificates.push_back(child->MakeCertificate());
  }
  std::sort(certificates.begin(), certificates.end());

  std::string open, close;
  if (type_ == kDecisionInfostateNode) {
    open = "[";
    close = "]";
  } else if (type_ == kObservationInfostateNode) {
    open = "(";
    close = ")";
  }

  return absl::StrCat(
      open, absl::StrJoin(certificates.begin(), certificates.end(), ""), close);
}

void InfostateNode::RebalanceSubtree(int target_depth, int current_depth) {
  SPIEL_DCHECK_LE(current_depth, target_depth);
  depth_ = current_depth;

  if (is_leaf_node() && target_depth != current_depth) {
    // Prepare the chain of dummy observations.
    depth_ = target_depth;
    std::unique_ptr<InfostateNode> node = Release();
    InfostateNode* node_parent = node->parent();
    int position_in_leaf_parent = node->incoming_index();
    std::unique_ptr<InfostateNode> chain_head =
        std::unique_ptr<InfostateNode>(new InfostateNode(
            /*tree=*/tree_, /*parent=*/nullptr,
            /*incoming_index=*/position_in_leaf_parent,
            kObservationInfostateNode,
            /*infostate_string=*/kFillerInfostate,
            /*terminal_utility=*/NAN, /*terminal_ch_reach_prob=*/NAN,
            current_depth, /*legal_actions=*/{}, /*terminal_history=*/{}));
    InfostateNode* chain_tail = chain_head.get();
    for (int i = 1; i < target_depth - current_depth; ++i) {
      chain_tail =
          chain_tail->AddChild(std::unique_ptr<InfostateNode>(new InfostateNode(
              /*tree=*/tree_, /*parent=*/chain_tail,
              /*incoming_index=*/0, kObservationInfostateNode,
              /*infostate_string=*/kFillerInfostate,
              /*terminal_utility=*/NAN, /*terminal_ch_reach_prob=*/NAN,
              current_depth + i, /*legal_actions=*/{},
              /*terminal_history=*/{})));
    }
    chain_tail->children_.push_back(nullptr);

    // First put the node to the chain. If we did it in reverse order,
    // i.e chain to parent and then node to the chain, the node would
    // become freed.
    auto* node_ptr = node.get();
    node_ptr->SwapParent(std::move(node), /*target=*/chain_tail, 0);
    auto* chain_head_ptr = chain_head.get();
    chain_head_ptr->SwapParent(std::move(chain_head), /*target=*/node_parent,
                               position_in_leaf_parent);
  }

  for (std::unique_ptr<InfostateNode>& child : children_) {
    child->RebalanceSubtree(target_depth, current_depth + 1);
  }
}

std::unique_ptr<InfostateNode> InfostateNode::Release() {
  SPIEL_DCHECK_TRUE(parent_);
  SPIEL_DCHECK_TRUE(parent_->children_.at(incoming_index_).get() == this);
  return std::move(parent_->children_.at(incoming_index_));
}

void InfostateNode::SwapParent(std::unique_ptr<InfostateNode> self,
                               InfostateNode* target, int at_index) {
  // This node is still who it thinks it is :)
  SPIEL_DCHECK_TRUE(self.get() == this);
  target->children_.at(at_index) = std::move(self);
  this->parent_ = target;
  this->incoming_index_ = at_index;
}

InfostateTree::InfostateTree(const std::vector<const State*>& start_states,
                             const std::vector<double>& chance_reach_probs,
                             std::shared_ptr<Observer> infostate_observer,
                             Player acting_player, int max_move_ahead_limit)
    : acting_player_(acting_player),
      infostate_observer_(std::move(infostate_observer)),
      root_(MakeRootNode()) {
  SPIEL_CHECK_FALSE(start_states.empty());
  SPIEL_CHECK_EQ(start_states.size(), chance_reach_probs.size());
  SPIEL_CHECK_GE(acting_player_, 0);
  SPIEL_CHECK_LT(acting_player_, start_states[0]->GetGame()->NumPlayers());
  SPIEL_CHECK_TRUE(infostate_observer_->HasString());

  int start_max_move_number = 0;
  for (const State* start_state : start_states) {
    start_max_move_number =
        std::max(start_max_move_number, start_state->MoveNumber());
  }

  for (int i = 0; i < start_states.size(); ++i) {
    RecursivelyBuildTree(root_.get(), /*depth=*/1, *start_states[i],
                         start_max_move_number + max_move_ahead_limit,
                         chance_reach_probs[i]);
  }

  // Operations to make after building the tree.
  RebalanceTree();
  nodes_at_depths_.resize(tree_height() + 1);
  CollectNodesAtDepth(mutable_root(), 0);
  LabelNodesWithIds();
}

void InfostateTree::RebalanceTree() {
  root_->RebalanceSubtree(tree_height(), 0);
}

void InfostateTree::CollectNodesAtDepth(InfostateNode* node, size_t depth) {
  nodes_at_depths_[depth].push_back(node);
  for (InfostateNode* child : node->child_iterator())
    CollectNodesAtDepth(child, depth + 1);
}

std::ostream& InfostateTree::operator<<(std::ostream& os) const {
  return os << "Infostate tree for player " << acting_player_ << ".\n"
            << "Tree height: " << tree_height_ << '\n'
            << "Root branching: " << root_branching_factor() << '\n'
            << "Number of decision infostate nodes: " << num_decisions() << '\n'
            << "Number of sequences: " << num_sequences() << '\n'
            << "Number of leaves: " << num_leaves() << '\n'
            << "Tree certificate: " << '\n'
            << root().MakeCertificate() << '\n';
}

std::unique_ptr<InfostateNode> InfostateTree::MakeNode(
    InfostateNode* parent, InfostateNodeType type,
    const std::string& infostate_string, double terminal_utility,
    double terminal_ch_reach_prob, size_t depth,
    const State* originating_state) {
  auto legal_actions =
      originating_state && originating_state->IsPlayerActing(acting_player_)
          ? originating_state->LegalActions(acting_player_)
          : std::vector<Action>();
  auto terminal_history = originating_state && originating_state->IsTerminal()
                              ? originating_state->History()
                              : std::vector<Action>();
  // Instantiate node using new to make sure that we can call
  // the private constructor.
  auto node = std::unique_ptr<InfostateNode>(new InfostateNode(
      *this, parent, parent->num_children(), type, infostate_string,
      terminal_utility, terminal_ch_reach_prob, depth, std::move(legal_actions),
      std::move(terminal_history)));
  return node;
}

std::unique_ptr<InfostateNode> InfostateTree::MakeRootNode() const {
  return std::unique_ptr<InfostateNode>(new InfostateNode(
      /*tree=*/*this, /*parent=*/nullptr, /*incoming_index=*/0,
      /*type=*/kObservationInfostateNode,
      /*infostate_string=*/kDummyRootNodeInfostate,
      /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN,
      /*depth=*/0, /*legal_actions=*/{}, /*terminal_history=*/{}));
}

void InfostateTree::UpdateLeafNode(InfostateNode* node, const State& state,
                                   size_t leaf_depth,
                                   double chance_reach_probs) {
  tree_height_ = std::max(tree_height_, leaf_depth);
  node->corresponding_states_.push_back(state.Clone());
  node->corresponding_ch_reaches_.push_back(chance_reach_probs);
}

void InfostateTree::RecursivelyBuildTree(InfostateNode* parent, size_t depth,
                                         const State& state, int move_limit,
                                         double chance_reach_prob) {
  if (state.IsTerminal())
    return BuildTerminalNode(parent, depth, state, chance_reach_prob);
  else if (state.IsPlayerActing(acting_player_))
    return BuildDecisionNode(parent, depth, state, move_limit,
                             chance_reach_prob);
  else
    return BuildObservationNode(parent, depth, state, move_limit,
                                chance_reach_prob);
}

void InfostateTree::BuildTerminalNode(InfostateNode* parent, size_t depth,
                                      const State& state,
                                      double chance_reach_prob) {
  const double terminal_utility = state.Returns()[acting_player_];
  InfostateNode* terminal_node = parent->AddChild(
      MakeNode(parent, kTerminalInfostateNode,
               infostate_observer_->StringFrom(state, acting_player_),
               terminal_utility, chance_reach_prob, depth, &state));
  UpdateLeafNode(terminal_node, state, depth, chance_reach_prob);
}

void InfostateTree::BuildDecisionNode(InfostateNode* parent, size_t depth,
                                      const State& state, int move_limit,
                                      double chance_reach_prob) {
  SPIEL_DCHECK_EQ(parent->type(), kObservationInfostateNode);
  std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);
  InfostateNode* decision_node = parent->GetChild(info_state);
  const bool is_leaf_node = state.MoveNumber() >= move_limit;

  if (decision_node) {
    // The decision node has been already constructed along with children
    // for each action: these are observation nodes.
    // Fetches the observation child and goes deeper recursively.
    SPIEL_DCHECK_EQ(decision_node->type(), kDecisionInfostateNode);

    if (is_leaf_node) {  // Do not build deeper.
      return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);
    }

    if (state.IsSimultaneousNode()) {
      const ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        InfostateNode* observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child, move_limit,
                               chance_reach_prob);
        }
      }
    } else {
      std::vector<Action> legal_actions = state.LegalActions(acting_player_);
      for (int i = 0; i < legal_actions.size(); ++i) {
        InfostateNode* observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);
        std::unique_ptr<State> child = state.Child(legal_actions.at(i));
        RecursivelyBuildTree(observation_node, depth + 2, *child, move_limit,
                             chance_reach_prob);
      }
    }
  } else {  // The decision node was not found yet.
    decision_node = parent->AddChild(MakeNode(
        parent, kDecisionInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));

    if (is_leaf_node) {  // Do not build deeper.
      return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);
    }

    // Build observation nodes right away after the decision node.
    // This is because the player might be acting multiple times in a row:
    // each time it might get some observations that branch the infostate
    // tree.

    if (state.IsSimultaneousNode()) {
      ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        // We build a dummy observation node.
        // We can't ask for a proper infostate string or an originating state,
        // because such a thing is not properly defined after only a partial
        // application of actions for the sim move state
        // (We need to supply all the actions).
        InfostateNode* observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     /*infostate_string=*/kFillerInfostate,
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     /*originating_state=*/nullptr));

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          // Only now we can advance the state, when we have all actions.
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child, move_limit,
                               chance_reach_prob);
        }
      }
    } else {  // Not a sim move node.
      for (Action a : state.LegalActions()) {
        std::unique_ptr<State> child = state.Child(a);
        InfostateNode* observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     infostate_observer_->StringFrom(*child, acting_player_),
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     child.get()));
        RecursivelyBuildTree(observation_node, depth + 2, *child, move_limit,
                             chance_reach_prob);
      }
    }
  }
}

void InfostateTree::BuildObservationNode(InfostateNode* parent, size_t depth,
                                         const State& state, int move_limit,
                                         double chance_reach_prob) {
  SPIEL_DCHECK_TRUE(state.IsChanceNode() ||
                    !state.IsPlayerActing(acting_player_));
  const bool is_leaf_node = state.MoveNumber() >= move_limit;
  const std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);

  InfostateNode* observation_node = parent->GetChild(info_state);
  if (!observation_node) {
    observation_node = parent->AddChild(MakeNode(
        parent, kObservationInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
  }
  SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);

  if (is_leaf_node) {  // Do not build deeper.
    return UpdateLeafNode(observation_node, state, depth, chance_reach_prob);
  }

  if (state.IsChanceNode()) {
    for (std::pair<Action, double> action_prob : state.ChanceOutcomes()) {
      std::unique_ptr<State> child = state.Child(action_prob.first);
      RecursivelyBuildTree(observation_node, depth + 1, *child, move_limit,
                           chance_reach_prob * action_prob.second);
    }
  } else {
    for (Action a : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(a);
      RecursivelyBuildTree(observation_node, depth + 1, *child, move_limit,
                           chance_reach_prob);
    }
  }
}
int InfostateTree::root_branching_factor() const {
  return root_->num_children();
}

std::shared_ptr<InfostateTree> MakeInfostateTree(const Game& game,
                                                 Player acting_player,
                                                 int max_move_limit) {
  // Uses new instead of make_shared, because shared_ptr is not a friend and
  // can't call private constructors.
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      {game.NewInitialState().get()}, /*chance_reach_probs=*/{1.},
      game.MakeObserver(kInfoStateObsType, {}), acting_player, max_move_limit));
}

std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<InfostateNode*>& start_nodes, int max_move_ahead_limit) {
  std::vector<const InfostateNode*> const_nodes(start_nodes.begin(),
                                                start_nodes.end());
  return MakeInfostateTree(const_nodes, max_move_ahead_limit);
}

std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<const InfostateNode*>& start_nodes,
    int max_move_ahead_limit) {
  SPIEL_CHECK_FALSE(start_nodes.empty());
  const InfostateNode* some_node = start_nodes[0];
  const InfostateTree& originating_tree = some_node->tree();
  SPIEL_DCHECK_TRUE([&]() {
    for (const InfostateNode* node : start_nodes) {
      if (!node) return false;
      if (!node->is_leaf_node()) return false;
      if (node->depth() != some_node->depth()) return false;
      if (&node->tree() != &originating_tree) return false;
    }
    return true;
  }());

  // We reserve a larger number of states, as infostate nodes typically contain
  // a large number of States. (8 is an arbitrary choice though).
  std::vector<const State*> start_states;
  start_states.reserve(start_nodes.size() * 8);
  std::vector<double> chance_reach_probs;
  chance_reach_probs.reserve(start_nodes.size() * 8);

  for (const InfostateNode* node : start_nodes) {
    for (int i = 0; i < node->corresponding_states_size(); ++i) {
      start_states.push_back(node->corresponding_states()[i].get());
      chance_reach_probs.push_back(node->corresponding_chance_reach_probs()[i]);
    }
  }

  // Uses new instead of make_shared, because shared_ptr is not a friend and
  // can't call private constructors.
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      start_states, chance_reach_probs, originating_tree.infostate_observer_,
      originating_tree.acting_player_, max_move_ahead_limit));
}

std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<const State*>& start_states,
    const std::vector<double>& chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player acting_player,
    int max_move_ahead_limit) {
  return std::shared_ptr<InfostateTree>(
      new InfostateTree(start_states, chance_reach_probs, infostate_observer,
                        acting_player, max_move_ahead_limit));
}
SequenceId InfostateTree::empty_sequence() const {
  return root().sequence_id();
}
absl::optional<DecisionId> InfostateTree::DecisionIdForSequence(
    const SequenceId& sequence_id) const {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode* node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  if (node->is_root_node()) {
    return {};
  } else {
    return node->parent_->decision_id();
  }
}
absl::optional<InfostateNode*> InfostateTree::DecisionForSequence(
    const SequenceId& sequence_id) {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode* node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  if (node->is_root_node()) {
    return {};
  } else {
    return node->parent_;
  }
}
bool InfostateTree::IsLeafSequence(const SequenceId& sequence_id) const {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode* node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  return node->start_sequence_id() == node->end_sequence_id();
}
std::vector<DecisionId> InfostateTree::DecisionIdsWithParentSeq(
    const SequenceId& sequence_id) const {
  std::vector<DecisionId> out;
  const InfostateNode* observation_node = sequences_.at(sequence_id.id());
  std::stack<const InfostateNode*> open_set;
  for (const InfostateNode* child : observation_node->child_iterator()) {
    open_set.push(child);
  }
  while (!open_set.empty()) {
    const InfostateNode* node = open_set.top();
    open_set.pop();
    if (node->type() == kDecisionInfostateNode &&
        node->sequence_id() == sequence_id) {
      out.push_back(node->decision_id());
    } else {
      for (const InfostateNode* child : node->child_iterator()) {
        open_set.push(child);
      }
    }
  }
  return out;
}

void InfostateTree::LabelNodesWithIds() {
  // Idea of labeling: label the leaf sequences first, and continue up the tree.
  size_t sequence_index = 0;
  size_t decision_index = 0;

  // Do not label leaf nodes with sequences.
  const int start_depth = nodes_at_depths_.size() - 2;

  for (int depth = start_depth; depth >= 0; --depth) {
    for (InfostateNode* node : nodes_at_depths_[depth]) {
      if (node->type() != kDecisionInfostateNode) continue;
      decision_infostates_.push_back(node);
      node->decision_id_ = DecisionId(decision_index++, this);

      for (InfostateNode* child : node->child_iterator()) {
        sequences_.push_back(child);
        child->sequence_id_ = SequenceId(sequence_index++, this);
      }
      // We could use sequence_index to set start and end sequences for
      // the decision infostate right away here, however we'd like to make
      // sure to label correctly all nodes in the tree.
    }
  }
  // Finally label the last sequence (an empty sequence) in the root node.
  sequences_.push_back(mutable_root());
  mutable_root()->sequence_id_ = SequenceId(sequence_index, this);

  CollectStartEndSequenceIds(mutable_root(), mutable_root()->sequence_id());
}

// Make a recursive call to assign the parent's sequences appropriately.
// Collect pairs of (start, end) sequence ids from children and propagate
// them up the tree. In case that deep nodes (close to the leaves) do not
// have any child decision nodes, set the (start, end) to the parent sequence.
// In this way the range iterator will be empty (start==end) and well defined.
std::pair<size_t, size_t> InfostateTree::CollectStartEndSequenceIds(
    InfostateNode* node, const SequenceId parent_sequence) {
  size_t min_index = kUndefinedNodeId;  // This is a large number.
  size_t max_index = 0;
  const SequenceId propagate_sequence_id =
      node->sequence_id_.is_undefined()
          ? parent_sequence
          : node->sequence_id();  // This becomes the parent for next nodes.

  for (InfostateNode* child : node->child_iterator()) {
    auto [min_child, max_child] =
        CollectStartEndSequenceIds(child, propagate_sequence_id);
    min_index = std::min(min_child, min_index);
    max_index = std::max(max_child, max_index);
  }

  if (min_index != kUndefinedNodeId) {
    SPIEL_CHECK_LE(min_index, max_index);
    node->start_sequence_id_ = SequenceId(min_index, this);
    node->end_sequence_id_ = SequenceId(max_index + 1, this);
  } else {
    node->start_sequence_id_ = propagate_sequence_id;
    node->end_sequence_id_ = propagate_sequence_id;
  }

  if (node->sequence_id_.is_undefined()) {
    // Propagate children limits.
    node->sequence_id_ = parent_sequence;
    return {min_index, max_index};
  } else {
    // We have hit a defined sequence id, propagate it up.
    return {node->sequence_id_.id(), node->sequence_id_.id()};
  }
}

std::pair<double, SfStrategy> InfostateTree::BestResponse(
    TreeplexVector<double>&& gradient) const {
  SPIEL_CHECK_EQ(this, gradient.tree());
  SPIEL_CHECK_EQ(num_sequences(), gradient.size());
  SfStrategy response(this);

  // 1. Compute counterfactual best response
  // (i.e. in all infostates, even unreachable ones)
  SequenceId current(0, this);
  const double init_value = -std::numeric_limits<double>::infinity();
  while (current.id() <= empty_sequence().id()) {
    double max_value = init_value;
    SequenceId max_id = current;
    const InfostateNode* node = observation_infostate(current);
    for (current = node->start_sequence_id();
         current != node->end_sequence_id(); current.next()) {
      if (gradient[current] > max_value) {
        max_value = gradient[current];
        max_id = current;
      }
    }
    if (init_value != max_value) {
      gradient[node->sequence_id()] += max_value;
      response[max_id] = 1.;
    }
    current.next();
  }
  SPIEL_CHECK_EQ(current.id(), empty_sequence().id() + 1);

  // 2. Prune away unreachable subtrees.
  //
  // This can be done with a more costly recursion.
  // Instead we make a more cache-friendly double pass through the response
  // vector: we increment the visited path by 1, resulting in a value of 2.
  // Then we zero-out all values but 2.
  current = empty_sequence();
  response[current] = 2.;
  while (!IsLeafSequence(current)) {
    for (SequenceId seq : observation_infostate(current)->AllSequenceIds()) {
      if (response[seq] == 1.) {
        current = seq;
        response[seq] += 1.;
        break;
      }
    }
  }
  for (SequenceId seq : response.range()) {
    response[seq] = response[seq] == 2. ? 1. : 0.;
  }
  SPIEL_DCHECK_TRUE(IsValidSfStrategy(response));
  return {gradient[empty_sequence()], response};
}

double InfostateTree::BestResponseValue(LeafVector<double>&& gradient) const {
  // Loop over all heights.
  for (int d = tree_height_ - 1; d >= 0; d--) {
    int left_offset = 0;
    // Loop over all parents of current nodes.
    for (int parent_idx = 0; parent_idx < nodes_at_depths_[d].size();
         parent_idx++) {
      const InfostateNode* node = nodes_at_depths_[d][parent_idx];
      const int num_children = node->num_children();
      const Range<LeafId> children_range =
          gradient.range(left_offset, left_offset + num_children);
      const LeafId parent_id(parent_idx, this);

      if (node->type() == kDecisionInfostateNode) {
        double max_value = std::numeric_limits<double>::min();
        for (LeafId id : children_range) {
          max_value = std::fmax(max_value, gradient[id]);
        }
        gradient[parent_id] = max_value;
      } else {
        SPIEL_DCHECK_EQ(node->type(), kObservationInfostateNode);
        double sum_value = 0.;
        for (LeafId id : children_range) {
          sum_value += gradient[id];
        }
        gradient[parent_id] = sum_value;
      }
      left_offset += num_children;
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(left_offset, nodes_at_depths_[d + 1].size());
  }
  const LeafId root_id(0, this);
  return gradient[root_id];
}

DecisionId InfostateTree::DecisionIdFromInfostateString(
    const std::string& infostate_string) const {
  for (InfostateNode* node : decision_infostates_) {
    if (node->infostate_string() == infostate_string)
      return node->decision_id();
  }
  return kUndefinedDecisionId;
}

bool CheckSum(const SfStrategy& strategy, SequenceId id, double expected_sum) {
  if (fabs(strategy[id] - expected_sum) > 1e-13) {
    return false;
  }

  const InfostateTree* tree = strategy.tree();
  if (tree->IsLeafSequence(id)) {
    return true;
  }

  double actual_sum = 0.;
  const InfostateNode* node = tree->observation_infostate(id);
  for (SequenceId sub_seq : node->AllSequenceIds()) {
    actual_sum += strategy[sub_seq];
  }
  if (fabs(actual_sum - expected_sum) > 1e-13) {
    return false;
  }

  for (SequenceId sub_seq : node->AllSequenceIds()) {
    if (!CheckSum(strategy, sub_seq, strategy[sub_seq])) {
      return false;
    }
  }
  return true;
}

bool IsValidSfStrategy(const SfStrategy& strategy) {
  return CheckSum(strategy, strategy.tree()->empty_sequence(), 1.);
}

}  // namespace algorithms
}  // namespace open_spiel
