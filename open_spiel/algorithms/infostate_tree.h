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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/action_view.h"

// This file contains an utility algorithm that builds an infostate tree
// for specified acting player, starting at some histories in the game.
//
// The identification of infostates is based on tensors from an information
// state observer.
//
// As infostate node can be extended to contain arbitrary values, it is
// implemented with curiously recurring template pattern (CRTP). The common
// usage for CFR is provided under a CFRTree and CFRNode respectively.

namespace open_spiel {
namespace algorithms {

// We use the nomenclature from the [Predictive CFR] paper.
//
// In _decision nodes_, the acting player selects actions.
// The _observation nodes_ can correspond to State that is a chance node,
// opponent's node, but importantly, also to the acting player's node,
// as the player may have discovered something as a result of its action
// in the previous decision node. Additionally, we use _terminal nodes_,
// which correspond to a single State terminal history.
//
// You can retrieve observation tensors for decision nodes via Tensor() method.
// The terminal nodes store player's utility as well as cumulative chance reach
// probability.
//
// [Predictive CFR] https://arxiv.org/pdf/2007.14358.pdf.
enum InfostateNodeType {
  kDecisionInfostateNode,
  kObservationInfostateNode,
  kTerminalInfostateNode
};

// Forward declarations.
template<class Node> class InfostateTree;
template<class Self> class InfostateNode;

template<class Self>
class InfostateNode {
 public:
  InfostateNode(const InfostateTree<Self>& tree, Self* parent,
                int incoming_index, InfostateNodeType type,
                absl::Span<float> tensor, double terminal_value,
                double terminal_ch_reach_prob, const State* originating_state)
      : tree_(tree), parent_(parent),
        incoming_index_(incoming_index), type_(type),
        // Copy the tensor.
        tensor_(tensor.begin(), tensor.end()),
        terminal_value_(terminal_value),
        terminal_chn_reach_prob_(terminal_ch_reach_prob) {

    // Implications for kTerminalNode
    SPIEL_DCHECK_TRUE(type != kTerminalInfostateNode || originating_state);
    SPIEL_DCHECK_TRUE(type != kTerminalInfostateNode || parent);
    SPIEL_DCHECK_TRUE(type != kTerminalInfostateNode || !tensor.empty());
    // Implications for kDecisionNode
    SPIEL_DCHECK_TRUE(type != kDecisionInfostateNode || originating_state);
    SPIEL_DCHECK_TRUE(type != kDecisionInfostateNode || parent);
    SPIEL_DCHECK_TRUE(type != kDecisionInfostateNode || !tensor.empty());
    // Implications for kObservationNode
    SPIEL_DCHECK_TRUE(
      !(type == kObservationInfostateNode && parent
          && parent->Type() == kDecisionInfostateNode)
      || (incoming_index >= 0 && incoming_index < parent->LegalActions().size())
    );

    if (type == kDecisionInfostateNode) {
      legal_actions_ = originating_state->LegalActions(tree_.GetPlayer());
    }
  }
  InfostateNode(InfostateNode&&) = default;
  virtual ~InfostateNode() = default;

  [[nodiscard]] const InfostateTree<Self>& Tree() const { return tree_; }
  [[nodiscard]] Self* Parent() const { return parent_; }
  int IncomingIndex() const { return incoming_index_; }
  const InfostateNodeType& Type() const { return type_; }
  bool IsLeafNode() const { return children_.empty(); }
  bool IsRootNode() const { return !parent_; }
  absl::Span<const float> Tensor() const {
    // Avoid working with empty tensors. Use HasTensor() first to check.
    SPIEL_CHECK_TRUE(HasTensor());
    return tensor_;
  }
  bool HasTensor() const { return !tensor_.empty(); }
  double TerminalValue() const {
    SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
    return terminal_value_;
  }
  double TerminalChanceReachProb() const {
    SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
    return terminal_chn_reach_prob_;
  }
  absl::Span<const Action> LegalActions() const {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return absl::MakeSpan(legal_actions_);
  }
  const std::vector<std::unique_ptr<State>>& CorrespondingStates() const {
    return corresponding_states_;
  }
  const std::vector<double>& CorrespondingChanceReaches() const {
    return corresponding_ch_reaches_;
  }
  [[nodiscard]] Self* AddChild(std::unique_ptr<Self> child) {
    SPIEL_CHECK_EQ(child->parent_, this);
    children_.push_back(std::move(child));
    return children_.back().get();
  }
  [[nodiscard]] Self* GetChild(absl::Span<float> tensor) const {
    for (const std::unique_ptr<Self>& child : children_) {
      if (child->Tensor() == tensor) return child.get();
    }
    return nullptr;
  }
  [[nodiscard]] Self* ChildAt(int i) const { return children_.at(i).get(); }
  int NumChildren() const { return children_.size(); }
  [[nodiscard]] const Self* FindNode(absl::Span<float> tensor_lookup) const {
    if (tensor_ == tensor_lookup)
      return open_spiel::down_cast<const Self*>(this);
    for (Self& child : *this) {
      if (const Self* node = child.FindNode(tensor_lookup)) {
        return node;
      }
    }
    return nullptr;
  }
  // Intended only for debug purposes.
  std::string ToString() const {
    if (!parent_) return "";
    if (!parent_->parent_) return std::to_string(incoming_index_);
    return absl::StrCat(parent_->ToString(), ",", incoming_index_);
  }

  // Iterate over children and expose references to the children
  // (instead of unique_ptrs).
  class ChildIterator {
    int pos_;
    const std::vector<std::unique_ptr<Self>>& children_;
   public:
    ChildIterator(const std::vector<std::unique_ptr<Self>>& children,
                  int pos = 0) : pos_(pos), children_(children) {}
    ChildIterator& operator++() { pos_++; return *this; }
    bool operator==(ChildIterator other) const { return pos_ == other.pos_; }
    bool operator!=(ChildIterator other) const { return !(*this == other); }
    [[nodiscard]] Self& operator*() { return *children_[pos_]; }
    ChildIterator begin() const { return *this; }
    ChildIterator end() const {
      return ChildIterator(children_, children_.size());
    }
  };
  ChildIterator child_iterator() const { return ChildIterator(children_); }

  void Rebalance(int max_depth, int current_depth) {
    SPIEL_DCHECK_LE(current_depth, max_depth);
    if (IsLeafNode() && max_depth != current_depth) {
      // Prepare the chain of dummy observations.
      std::unique_ptr<Self> node = Release();
      Self* node_parent = node->Parent();
      int position_in_leaf_parent = node->IncomingIndex();
      std::unique_ptr<Self> chain_head =
          std::unique_ptr<Self>(new Self(
              /*tree=*/tree_, /*parent=*/nullptr,
              /*incoming_index=*/position_in_leaf_parent,
              kObservationInfostateNode,
              /*tensor=*/{}, /*terminal_value=*/NAN,
              /*terminal_ch_reach_prob=*/NAN, /*originating_state=*/nullptr));
      Self* chain_tail = chain_head.get();
      for (int i = 1; i < max_depth - current_depth; ++i) {
        chain_tail = chain_tail->AddChild(
            std::unique_ptr<Self>(new Self(
                /*tree=*/tree_, /*parent=*/chain_tail,
                /*incoming_index=*/0, kObservationInfostateNode,
                /*tensor=*/{}, /*terminal_value=*/NAN,
                /*terminal_ch_reach_prob=*/NAN,
                /*originating_state=*/nullptr)));
      }
      chain_tail->children_.push_back(nullptr);

      // First put the node to the chain. If we did it in reverse order,
      // i.e chain to parent and then node to the chain, the node would
      // become freed.
      node->SwapParent(std::move(node), /*target=*/chain_tail, 0);
      chain_head->SwapParent(std::move(chain_head), /*target=*/node_parent,
                             position_in_leaf_parent);
    }

    for (std::unique_ptr<Self>& child : children_) {
      child->Rebalance(max_depth, current_depth + 1);
    }
  }

 private:
  // Get the unique_ptr for this node. The usage is intended only for tree
  // balance manipulation.
  std::unique_ptr<Self> Release() {
    SPIEL_DCHECK_TRUE(parent_);
    SPIEL_DCHECK_TRUE(parent_->children_.at(incoming_index_).get() == this);
    return std::move(parent_->children_.at(incoming_index_));
  }

  // Change the parent of this node by inserting it at at index
  // of the new parent. The node at the existing position will be freed.
  // We pass the unique ptr of itself, because calling Release might be
  // undefined: the node we want to swap a parent for can be root of a subtree.
  void SwapParent(std::unique_ptr<Self> self, Self* target, int at_index) {
    // This node is still who it thinks it is :)
    SPIEL_DCHECK_TRUE(self.get() == this);
    target->children_.at(at_index) = std::move(self);
    this->parent_ = target;
    this->incoming_index_ = at_index;
  }

 protected:
  // Needed for adding corresponding_states_ during tree traversal.
  friend class InfostateTree<Self>;

  const InfostateTree<Self>& tree_;
  // Pointer to the parent node.
  // This is not const so that we can change it when we SwapParent().
  Self* parent_;
  // Position of this node in the parent's children, i.e. it should hold that
  //   parent_->children_.at(incoming_index_).get() == this.
  // For decision nodes this corresponds also to the
  //   State::LegalActions(player_).at(incoming_index_)
  // This is not const so that we can change it when we SwapParent().
  int incoming_index_;
  const InfostateNodeType type_;
  const std::vector<float> tensor_;
  const double terminal_value_;
  const double terminal_chn_reach_prob_;
  // Only for decision nodes.
  std::vector<Action> legal_actions_;
  // Children infostate nodes. Notice the node owns its children.
  std::vector<std::unique_ptr<Self>> children_;
  // Optionally store States that correspond to this infostate node.
  std::vector<std::unique_ptr<State>> corresponding_states_;
  std::vector<double> corresponding_ch_reaches_;
};

template<class Node>
class InfostateTree final {
 public:
  // Creates an infostate tree for a player based on the initial state
  // of the game, up to some move limit.
  InfostateTree(const Game& game, Player acting_player,
                int max_move_limit = 1000)
      : player_(acting_player),
        infostate_observer_(game.MakeObserver(kInfoStateObsType, {})),
        root_(/*tree=*/*this, /*parent=*/nullptr, /*incoming_index=*/0,
              /*type=*/kObservationInfostateNode, /*tensor=*/{},
              /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN,
              /*originating_state=*/nullptr),
        observation_(std::move(CreateObservation(game))) {
    std::unique_ptr<State> root_state = game.NewInitialState();
    RecursivelyBuildTree(&root_, /*depth=*/1, *root_state,
                         max_move_limit, /*chance_reach_prob=*/1.);
  }

  // Creates an infostate tree for a player based on some start states,
  // using an infostate observer to provide tensor observations,
  // up to some move limit from the deepest start state.
  InfostateTree(
      absl::Span<const State*> start_states,
      absl::Span<const float> chance_reach_probs,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_move_ahead_limit = 1000)
      : player_(acting_player),
        infostate_observer_(std::move(infostate_observer)),
        // Root is just a dummy node, and has a tensor full of zeros.
        // It cannot be retrieved via Get* methods, only by using the Root()
        // method.
        root_(/*tree=*/*this, /*parent=*/nullptr, /*incoming_index=*/0,
              /*type=*/kObservationInfostateNode, /*tensor=*/{},
              /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN,
              /*originating_state=*/nullptr),
      observation_(std::move(CreateObservation(*start_states.at(0)))) {
    SPIEL_CHECK_EQ(start_states.size(), chance_reach_probs.size());

    int start_max_move_number = 0;
    for (const State* start_state : start_states) {
      start_max_move_number = std::max(start_max_move_number,
                                       start_state->MoveNumber());
    }

    for (int i = 0; i < start_states.size(); ++i) {
      RecursivelyBuildTree(
          &root_, /*depth=*/1, *start_states[i],
          start_max_move_number + max_move_ahead_limit,
          chance_reach_probs[i]);
    }
  }

  [[nodiscard]] const Node& Root() const { return root_; }
  [[nodiscard]] Node* MutableRoot() { return &root_; }
  Player GetPlayer() const { return player_; }
  const Observer& GetObserver() const { return *infostate_observer_; }
  int TreeHeight() const { return tree_height_; }
  bool IsBalanced() const { return is_tree_balanced_; }

  // Identify node that corresponds to this tensor observation.
  // If the node is not found, returns a nullptr.
  [[nodiscard]] const Node* FindNode(absl::Span<float> tensor_lookup) const {
    return root_.FindNode(tensor_lookup);
  }

  // Makes sure that all tree leaves are at the same height.
  // It inserts a linked list of dummy observation nodes with appropriate length
  // to balance all the leaves. In the worst case this makes the tree about 2x
  // as large (in the number of nodes).
  void Rebalance() {
    root_.Rebalance(TreeHeight(), 0);
    is_tree_balanced_ = true;
  }

  // Iterate over all leaves.
  class LeavesIterator {
    const InfostateTree* tree_;
    const Node* current_;
   public:
    LeavesIterator(const InfostateTree* tree, const Node* current)
    : tree_(tree), current_(current) {
      SPIEL_CHECK_TRUE(current_);
      SPIEL_CHECK_TRUE(current_->IsLeafNode() || current_->IsRootNode());
    }
    LeavesIterator& operator++() {
      if (!current_->Parent()) SpielFatalError("All leaves have been iterated!");
      SPIEL_CHECK_TRUE(current_->IsLeafNode());
      int child_idx;
      do {  // Find some parent that was not fully traversed.
        SPIEL_DCHECK_LT(current_->IncomingIndex(),
                        current_->Parent()->NumChildren());
        SPIEL_DCHECK_EQ(current_->Parent()->ChildAt(current_->IncomingIndex()),
                        current_);
        child_idx = current_->IncomingIndex();
        current_ = current_->Parent();
      } while (current_->Parent()
            && child_idx + 1 == current_->NumChildren());
      // We traversed the whole tree and we got the root node.
      if (!current_->Parent() && child_idx + 1 == current_->NumChildren())
        return *this;
      // Choose the next sibling node.
      current_ = current_->ChildAt(child_idx + 1);
      // Find the first leaf.
      while (!current_->IsLeafNode()) {
        current_ = current_->ChildAt(0);
      }
      return *this;
    }
    bool operator==(LeavesIterator other) const {
      return current_ == other.current_;
    }
    bool operator!=(LeavesIterator other) const { return !(*this == other); }
    [[nodiscard]] const Node& operator*() const { return *current_; }
    LeavesIterator begin() const { return *this; }
    LeavesIterator end() const {
      return LeavesIterator(tree_, &(current_->Tree().Root()));
    }
  };
  LeavesIterator leaves_iterator() const {
    // Find the first leaf.
    const Node* node = &root_;
    while (!node->IsLeafNode()) node = node->ChildAt(0);
    return LeavesIterator(this, node);
  }
  // Expensive. Use only for debugging.
  int CountLeaves() const {
    int cnt = 0;
    for (const Node& n : leaves_iterator()) cnt++;
    return cnt;
  }

 private:
  const Player player_;
  const std::shared_ptr<Observer> infostate_observer_;
  Node root_;
  Observation observation_;

  // A value that helps to determine if the tree is balanced.
  int tree_height_ = -1;
  // We call a tree balanced if all leaves are in the same depth.
  bool is_tree_balanced_ = true;

  // Create observation here, so that we can run a number of checks,
  // which cannot be done in the initialization list.
  Observation CreateObservation(const State& start_state) const {
    SPIEL_CHECK_TRUE(infostate_observer_->HasTensor());
    const std::shared_ptr<const Game>& game = start_state.GetGame();
    SPIEL_CHECK_GE(player_, 0);
    SPIEL_CHECK_LT(player_, game->NumPlayers());
    return Observation(*game, infostate_observer_);
  }
  Observation CreateObservation(const Game& game) const {
    SPIEL_CHECK_GE(player_, 0);
    SPIEL_CHECK_LT(player_, game.NumPlayers());
    return Observation(game, infostate_observer_);
  }

  // Utility function whenever we create a new node for the tree.
  std::unique_ptr<Node> MakeNode(
      Node* parent, InfostateNodeType type, absl::Span<float> tensor,
      double terminal_value, double terminal_ch_reach_prob,
      const State* originating_state) {
    return std::make_unique<Node>(
        *this, parent, parent->NumChildren(), type,
        tensor, terminal_value, terminal_ch_reach_prob, originating_state);
  }

  // Track and update information about tree balance.
  void UpdateLeafNode(Node* node, const State& state,
                      int leaf_depth, double chance_reach_probs) {
    if (tree_height_ != -1 && is_tree_balanced_) {
      is_tree_balanced_ = tree_height_ == leaf_depth;
    }
    tree_height_ = std::max(tree_height_, leaf_depth);
    node->corresponding_states_.push_back(state.Clone());
    node->corresponding_ch_reaches_.push_back(chance_reach_probs);
  }

  void RecursivelyBuildTree(Node* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob) {
    observation_.SetFrom(state, player_);

    if (state.IsTerminal())
      return BuildTerminalNode(parent, depth, state, chance_reach_prob);
    else if (state.IsPlayerActing(player_))
      return BuildDecisionNode(parent, depth, state, move_limit,
                               chance_reach_prob);
    else
      return BuildObservationNode(parent, depth, state, move_limit,
                                  chance_reach_prob);
  }

  void BuildTerminalNode(Node* parent, int depth, const State& state,
                         double chance_reach_prob) {
    const double terminal_value = state.Returns()[player_];
    Node* terminal_node = parent->AddChild(MakeNode(
        parent, kTerminalInfostateNode, observation_.Tensor(), terminal_value,
        chance_reach_prob, &state));
    UpdateLeafNode(terminal_node, state, depth, chance_reach_prob);
  }

  void BuildDecisionNode(Node* parent, int depth, const State& state,
                         int move_limit, double chance_reach_prob) {
    SPIEL_DCHECK_EQ(parent->Type(), kObservationInfostateNode);
    Node* decision_node = parent->GetChild(observation_.Tensor());
    const bool is_leaf_node = state.MoveNumber() >= move_limit;

    if (decision_node) {
      // The decision node has been already constructed along with children
      // for each action: these are observation nodes.
      // Fetches the observation child and goes deeper recursively.
      SPIEL_DCHECK_EQ(decision_node->Type(), kDecisionInfostateNode);

      if (is_leaf_node)  // Do not build deeper.
        return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);

      if (state.IsSimultaneousNode()) {
        const ActionView action_view(state);
        for (int i = 0; i < action_view.legal_actions[player_].size(); ++i) {
          Node* observation_node = decision_node->ChildAt(i);
          SPIEL_DCHECK_EQ(observation_node->Type(),
                          kObservationInfostateNode);

          for (Action flat_actions : action_view.fixed_action(player_, i)) {
            std::unique_ptr<State> child = state.Child(flat_actions);
            RecursivelyBuildTree(observation_node, depth + 2, *child,
                                 move_limit, chance_reach_prob);
          }
        }
      } else {
        std::vector<Action> legal_actions = state.LegalActions(player_);
        for (int i = 0; i < legal_actions.size(); ++i) {
          Node* observation_node = decision_node->ChildAt(i);
          SPIEL_DCHECK_EQ(observation_node->Type(),
                          kObservationInfostateNode);
          std::unique_ptr<State> child = state.Child(legal_actions.at(i));
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }
      }
    } else {  // The decision node was not found yet.
      decision_node = parent->AddChild(MakeNode(
          parent, kDecisionInfostateNode, observation_.Tensor(),
          /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN, &state));

      if (is_leaf_node)  // Do not build deeper.
        return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);

      // Build observation nodes right away after the decision node.
      // This is because the player might be acting multiple times in a row:
      // each time it might get some observations that branch the infostate
      // tree.

      if (state.IsSimultaneousNode()) {
        ActionView action_view(state);
        for (int i = 0; i < action_view.legal_actions[player_].size(); ++i) {
          // We build a dummy observation node.
          // We can't ask for a proper tensor or an originating state, because
          // such a thing is not properly defined after only a partial
          // application of actions for the sim move state (We need to supply
          // all the actions).
          Node* observation_node = decision_node->AddChild(MakeNode(
              decision_node, kObservationInfostateNode, /*tensor=*/{},
              /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN,
              /*originating_state=*/nullptr));

          for (Action flat_actions : action_view.fixed_action(player_, i)) {
            // Only now we can advance the state, when we have all actions.
            std::unique_ptr<State> child = state.Child(flat_actions);
            RecursivelyBuildTree(observation_node, depth + 2, *child,
                                 move_limit, chance_reach_prob);
          }

        }
      } else {  // Not a sim move node.
        for (Action a : state.LegalActions()) {
          std::unique_ptr<State> child = state.Child(a);
          observation_.SetFrom(*child, player_);
          Node* observation_node = decision_node->AddChild(MakeNode(
              decision_node, kObservationInfostateNode, observation_.Tensor(),
              /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN,
              child.get()));
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }
      }
    }
  }

  void BuildObservationNode(Node* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob) {
    SPIEL_DCHECK_TRUE(state.IsChanceNode() || !state.IsPlayerActing(player_));
    const bool is_leaf_node = state.MoveNumber() >= move_limit;

    Node* observation_node = parent->GetChild(observation_.Tensor());
    if (!observation_node) {
      observation_node = parent->AddChild(MakeNode(
          parent, kObservationInfostateNode, observation_.Tensor(),
          /*terminal_value=*/NAN, /*chance_reach_prob=*/NAN, &state));
    }
    SPIEL_DCHECK_EQ(observation_node->Type(), kObservationInfostateNode);

    if (is_leaf_node)  // Do not build deeper.
      return UpdateLeafNode(observation_node, state, depth, chance_reach_prob);

    if (state.IsChanceNode()) {
      for (std::pair<Action, double> action_prob : state.ChanceOutcomes()) {
        std::unique_ptr<State> child = state.Child(action_prob.first);
        RecursivelyBuildTree(observation_node, depth + 1, *child,
                             move_limit,
                             chance_reach_prob * action_prob.second);
      }
    } else {
      for (Action a : state.LegalActions()) {
        std::unique_ptr<State> child = state.Child(a);
        RecursivelyBuildTree(observation_node, depth + 1, *child,
                             move_limit, chance_reach_prob);
      }
    }
  }
};

// Provide convenient types for usage in CFR-based algorithms.
class CFRNode;
using CFRTree = InfostateTree<CFRNode>;

class CFRNode : public InfostateNode</*Self=*/CFRNode> {
 public:
  CFRInfoStateValues values_;  // TODO: use just floats.
  std::vector<Action> terminal_history_;
  std::string infostate_string_;
  CFRNode(const CFRTree& tree, CFRNode* parent, int incoming_index,
          InfostateNodeType type, absl::Span<float> tensor,
          double terminal_value, double terminal_chn_reach_prob,
          const State* originating_state) :
      InfostateNode<CFRNode>(
          tree, parent, incoming_index, type, tensor, terminal_value,
          terminal_chn_reach_prob, originating_state)  {
    SPIEL_DCHECK_TRUE(
        !(originating_state && type == kDecisionInfostateNode)
            || originating_state->IsPlayerActing(tree.GetPlayer()));
    if (originating_state) {
      if (type_ == kDecisionInfostateNode) {
        values_ = CFRInfoStateValues(
            originating_state->LegalActions(tree.GetPlayer()));
        infostate_string_ = Tree().GetObserver().StringFrom(
            *originating_state, Tree().GetPlayer());
      }
      if (type_ == kTerminalInfostateNode) {
        terminal_history_ = originating_state->History();
      }
    }
  }

  // Provide a convenient operator to access the values.
  CFRInfoStateValues* operator->() {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return &values_;
  }
  // Provide a const getter as well.
  const CFRInfoStateValues& values() const {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return values_;
  }
  absl::Span<const Action> TerminalHistory() const {
    SPIEL_DCHECK_EQ(type_, kTerminalInfostateNode);
    return absl::MakeSpan(terminal_history_);
  }
};

inline void CollectInfostateLookupTable(
    const CFRNode& node,
    std::unordered_map<std::string, const CFRInfoStateValues*>* out) {
  if (node.Type() == kDecisionInfostateNode) {
    (*out)[node.infostate_string_] = &node.values();
  }
  for (const CFRNode& child : node.child_iterator()) {
    CollectInfostateLookupTable(child, out);
  }
}


}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
