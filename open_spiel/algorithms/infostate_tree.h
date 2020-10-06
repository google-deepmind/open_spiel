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

// This file contains an utility algorithm that builds an infostate tree
// for some acting player, starting at some histories in the game.
//
// The identification of infostates is based on tensors from an information
// state observer.
//
// As infostate node may need to contain arbitrary values, it is implemented
// with templates. The common usage for CFR is provided under a CFRTree
// and CFRNode respectively.

namespace open_spiel {
namespace algorithms {

// We use the nomenclature from the [Predictive CFR] paper.
//
// In _decision nodes_, the acting player selects actions.
// The _observation nodes_ can correspond to State that is a chance node,
// opponent's node, but importantly, also to the acting player's node,
// as the player may have discovered something as a result of its action
// in the previous decision node.
//
// Additionally, we use _terminal nodes_, which correspond to a single State
// terminal history. The player's utility is multiplied by chance reach
// probability to get an expected value of the terminal history.
//
// [Predictive CFR] https://arxiv.org/pdf/2007.14358.pdf.
enum InfostateNodeType {
  kDecisionNode,
  kObservationNode,
  kTerminalNode
};

// Forward declarations.
template<class NodeContents> class InfostateTree;
template<class Contents> class InfostateNode;

template<class Contents>
class InfostateNode {
 public:
  InfostateNode(InfostateTree<Contents>* tree, InfostateNode* parent,
                int incoming_index, InfostateNodeType type,
                absl::Span<float> tensor, double terminal_value,
                const State* originating_state, Player infostate_tree_player)
      : tree_(tree), parent_(parent),
        incoming_index_(incoming_index), type_(type),
        // Copy the tensor.
        tensor_(tensor.begin(), tensor.end()),
        terminal_value_(terminal_value) {
    if (type == kDecisionNode)
      legal_actions_ = originating_state->LegalActions(infostate_tree_player);
  }

  InfostateNode* Parent() const { return parent_; }
  InfostateTree<Contents>* Tree() const { return tree_; }
  const InfostateNodeType& Type() const { return type_; }
  absl::Span<const float> Tensor() const { return tensor_; }
  double TerminalValue() {
    SPIEL_CHECK_EQ(type_, kTerminalNode);
    return terminal_value_;
  }
  absl::Span<const Action> LegalActions() const {
    SPIEL_CHECK_EQ(type_, kDecisionNode);
    return absl::MakeSpan(legal_actions_);
  }
  InfostateNode* AddChild(std::unique_ptr<InfostateNode> child) {
    children_.push_back(std::move(child));
    return children_.back().get();
  }
  InfostateNode* GetChild(absl::Span<float> tensor) {
    for (const std::unique_ptr<InfostateNode>& child : children_) {
      if (child->Tensor() == tensor) return child.get();
    }
    return nullptr;
  }
  InfostateNode* ChildAt(int i) { return children_.at(i).get(); }

  long NumChildren() const { return children_.size(); }

  std::vector<int> GetSequence() const {
    std::vector<int> sequence;
    InfostateNode* parent = parent_;
    while(parent) {
      sequence.push_back(incoming_index_);
      parent = parent->parent_;
    }
    return sequence;
  }

  // Provide a convenient way to access the Contents
  // without calling some getter.
  Contents* operator->() { return &content_; }

  // Iterate over children.
  class iterator {
    long pos_;
    const std::vector<std::unique_ptr<InfostateNode>>& children_;
   public:
    iterator(const std::vector<std::unique_ptr<InfostateNode>>& children,
             long pos = 0) : pos_(pos), children_(children) {}
    iterator& operator++() { pos_++; return *this; }
    bool operator==(iterator other) const { return pos_ == other.pos_; }
    bool operator!=(iterator other) const { return !(*this == other); }
    InfostateNode& operator*() { return *children_[pos_]; }
  };
  iterator begin() const { return iterator(children_); }
  iterator end() const { return iterator(children_, children_.size()); }

 private:
  const InfostateTree<Contents>* tree_;
  const InfostateNode* parent_;
  const int incoming_index_;
  const InfostateNodeType type_;
  const std::vector<float> tensor_;
  const double terminal_value_;
  std::vector<Action> legal_actions_;
  Contents content_;
  std::vector<std::unique_ptr<InfostateNode>> children_;
};

template<class NodeContents>
class InfostateTree {
  using Node = InfostateNode<NodeContents>;

 public:

  // Creates an infostate tree for a player based on initial state of a game
  // up to some depth limit.
  InfostateTree(const Game& game, Player acting_player,
                int max_depth_limit = 1000)
      : player_(acting_player),
        infostate_observer_(game.MakeObserver(kInfoStateObsType, {})),
        root_(/*tree=*/this, /*parent=*/nullptr, /*incoming_index=*/0,
              /*type=*/kObservationNode, /*tensor=*/{}, /*terminal_value=*/0,
              /*originating_state=*/nullptr, /*acting_player=*/player_),
        observation_(std::move(CreateObservation(game))) {
    std::unique_ptr<State> root_state = game.NewInitialState();
    RecursivelyBuildTree(&root_, /*depth=*/1, *root_state,
                         max_depth_limit, /*chance_reach_prob=*/1.);
  }

  // Create an infostate tree for a player based on some start states,
  // using an infostate observer to provide tensor observations,
  // up to some depth limit from the deepest start state.
  //
  // The root node is a dummy observation node, so that we can have one
  // infostate tree instead of a forest of infostate trees.
  InfostateTree(
      absl::Span<const State*> start_states,
      absl::Span<const double> chance_reach_probs,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_depth_limit = 1000)
      : player_(acting_player),
        infostate_observer_(std::move(infostate_observer)),
        // Root is just a dummy node, and has a tensor full of zeros.
        // It cannot be retrieved via Get* methods, only by using the Root()
        // method.
        root_(/*tree=*/this, /*parent=*/nullptr, /*incoming_index=*/0,
              /*type=*/kObservationNode, /*tensor=*/{}, /*terminal_value=*/0,
              /*originating_state=*/nullptr, /*acting_player=*/player_),
      observation_(std::move(CreateObservation(*start_states.at(0)))) {
    SPIEL_CHECK_EQ(start_states.size(), chance_reach_probs.size());

    int start_max_depth = 0;
    for (const State* start_state : start_states) {
      start_max_depth = std::max(start_max_depth, start_state->MoveNumber());
    }

    for (int i = 0; i < start_states.size(); ++i) {
      RecursivelyBuildTree(
          &root_, /*depth=*/1, *start_states[i],
          start_max_depth + max_depth_limit,
          chance_reach_probs[i]);
    }
  }

  Node* Root() { return &root_; }
  Player GetPlayer() const { return player_; }
  int TreeHeight() const { return tree_height_; }
  bool IsBalanced() const { return is_tree_balanced_; }
  // Convenient methods to directly access decision nodes by observation
  // or State. This is useful for looking up solved policy, but not neccessary
  // for the infostate tree construction or running CFR iterations.
  Node* GetByCompressed(const std::string& observation) const {
    auto it = lookup_table_.find(observation);
    return it == lookup_table_.end() ? nullptr : it->second;
  }
  Node* GetByState(const State& state) {
    SPIEL_CHECK_TRUE(state.IsPlayerActing(player_));
    observation_.SetFrom(state, player_);
    std::string compressed = observation_.Compress();
    return GetByCompressed(compressed);
  }

 private:
  const Player player_;
  const std::shared_ptr<Observer> infostate_observer_;
  Node root_;
  Observation observation_;

  // A value that helps to determine if the tree is balanced.
  int tree_height_ = -1;
  // We call an infostate tree balanced if all terminal nodes
  // are in the same depth.
  bool is_tree_balanced_ = true;

  // Store compressed observations for fast lookup of decision nodes
  // in the lookup table.
  std::unordered_map<std::string, Node*> lookup_table_;

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

  // Call this function whenever we create a new node for the tree.
  std::unique_ptr<Node> MakeNode(
      Node* parent, InfostateNodeType type, absl::Span<float> tensor,
      double terminal_value, const State* originating_state) {
    return std::make_unique<Node>(
        this, parent, parent->NumChildren(), type,
        tensor, terminal_value, originating_state, player_);
  }

  void UpdateBalanceInfo(int leaf_depth) {
    // Track information about tree balance.
    if (tree_height_ != -1 && is_tree_balanced_) {
      is_tree_balanced_ = tree_height_ == leaf_depth;
    }
    tree_height_ = std::max(tree_height_, leaf_depth);
  }

  void RecursivelyBuildTree(Node* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob) {
    observation_.SetFrom(state, player_);

    // Create terminal nodes.
    if (state.IsTerminal()) {
      double terminal_value = state.Returns()[player_] * chance_reach_prob;
      parent->AddChild(MakeNode(parent, kTerminalNode, observation_.Tensor(),
                                terminal_value, &state));
      return UpdateBalanceInfo(depth);
    }

    // Create decision nodes.
    if (state.IsPlayerActing(player_)) {
      Node* decision_node = parent->GetChild(observation_.Tensor());

      if (decision_node) {
        // The decision node has been already constructed along with children
        // for each action: these are observation nodes.
        // Fetches the observation child and goes deeper recursively.
        SPIEL_DCHECK_EQ(decision_node->Type(), kDecisionNode);

        if (state.MoveNumber() >= move_limit)  // Do not build deeper.
          return UpdateBalanceInfo(depth);

        std::vector<Action> legal_actions = state.LegalActions();
        for (int i = 0; i < legal_actions.size(); ++i) {
          Node* observation_node = decision_node->ChildAt(i);
          SPIEL_DCHECK_EQ(observation_node->Type(), kObservationNode);
          std::unique_ptr<State> child = state.Child(legal_actions.at(i));
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }
      } else {
        decision_node = parent->AddChild(MakeNode(
            parent, kDecisionNode, observation_.Tensor(), 0, &state));
        lookup_table_.insert({observation_.Compress(), decision_node});

        if (state.MoveNumber() >= move_limit)  // Do not build deeper.
          return UpdateBalanceInfo(depth);

        // Build observation nodes right away after the decision node.
        // This is because the player might be acting multiple times in a row:
        // each time it might get some observations that branch the infostate
        // tree.
        for (Action a : state.LegalActions()) {
          std::unique_ptr<State> child = state.Child(a);
          observation_.SetFrom(*child, player_);
          Node* observation_node = decision_node->AddChild(MakeNode(
              parent, kObservationNode, observation_.Tensor(),
              /*terminal_value=*/0, child.get()));
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }
      }
      return;
    }

    // Finally, create observation nodes.
    SPIEL_DCHECK_TRUE(state.IsChanceNode()
                          || !state.IsPlayerActing(player_));
    Node* observation_node = parent->GetChild(observation_.Tensor());
    if (!observation_node) {
      observation_node = parent->AddChild(MakeNode(
          parent, kObservationNode, observation_.Tensor(), 0, &state));
    }
    SPIEL_DCHECK_EQ(observation_node->Type(), kObservationNode);

    if (state.MoveNumber() >= move_limit)  // Do not build deeper.
      return UpdateBalanceInfo(depth);

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
using CFRTree = InfostateTree</*NodeContents=*/CFRInfoStateValues>;
using CFRNode = InfostateNode</*Contents=*/CFRInfoStateValues>;

// Specialize CFRNode, because we construct the content
// of CFRInfoStateValues differently for decision nodes.
template<> CFRNode::InfostateNode(
    CFRTree* tree, CFRNode* parent, int incoming_index, InfostateNodeType type,
    absl::Span<float> tensor, double terminal_value,
    const State* originating_state, Player player) :
  tree_(tree), parent_(parent), incoming_index_(incoming_index), type_(type),
  tensor_(tensor.begin(), tensor.end()),
  terminal_value_(terminal_value),
  content_(originating_state && type == kDecisionNode
    ? CFRInfoStateValues(originating_state->LegalActions(player))
    : CFRInfoStateValues()) {
    // Do not save legal actions, as they are already saved
    // within CFRInfoStateValues. Instead, change the LegalActions
    // implementation to refer to these values directly.
    SPIEL_DCHECK_TRUE(
       !(originating_state && type == kDecisionNode)
       || originating_state->IsPlayerActing(player));
  }
template<> absl::Span<const Action> CFRNode::LegalActions() const {
  SPIEL_CHECK_EQ(type_, kDecisionNode);
  return content_.legal_actions;
}

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
