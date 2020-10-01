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
// state observer. In general, the implementation is not restricted to only
// this observer type.
//
// As infostate node may need contain arbitrary values, it is implemented
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
// Additionally, we use _terminal nodes_, which do not necessarily correspond
// to a single State terminal history, but can span a number of them.
// This is useful for aggregating over terminal histories in games like Poker,
// where players may not learn what were the exact card combinations when
// a player folds.
//
// [Predictive CFR] https://arxiv.org/pdf/2007.14358.pdf.
enum InfostateNodeType {
  kDecisionNode,
  kObservationNode,
  kTerminalNode
};

InfostateNodeType GetInfostateNodeType(State* state, Player player) {
  if (!state) return kObservationNode;  // i.e. dummy root node.
  if (state->IsTerminal()) return kTerminalNode;
  if (state->IsPlayerActing(player)) return kDecisionNode;
  return kObservationNode;
}

template<class Contents>
class InfostateNode {
 public:
  InfostateNode(InfostateNode* parent, State* originating_state,
                Player acting_player, absl::Span<float> tensor)
      : parent_(parent),
        type_(GetInfostateNodeType(originating_state, acting_player)),
        // Copy the tensor.
        tensor_(tensor.begin(), tensor.end()) {}

  const InfostateNodeType& Type() const { return type_; }
  absl::Span<const float> Tensor() const { return tensor_; }
  InfostateNode* Parent() const { return parent_; }
  InfostateNode* AddChild(std::unique_ptr<InfostateNode> child) {
    children_.push_back(std::move(child));
    return children_.back().get();
  }

  // Provide a convenient way to access the Content
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
  Contents content_;
  InfostateNode* parent_ = nullptr;
  InfostateNodeType type_;
  std::vector<float> tensor_;
  std::vector<std::unique_ptr<InfostateNode>> children_;
};

template<class NodeContents>
class InfostateTree {
  using Node = InfostateNode<NodeContents>;

 public:
  // Create an infostate tree for a player based on some start states,
  // using an infostate observer to provide tensor observations,
  // up to some depth limit from the deepest start state.
  //
  // The root node is a dummy observation node, so that we can have one
  // infostate tree instead of a forest of infostate trees.
  InfostateTree(
      std::vector<std::unique_ptr<State>> start_states,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_depth_limit = 1000) :
      player_(acting_player),
      infostate_observer_(std::move(infostate_observer)),
      observation_(std::move(CreateObservation(*start_states.at(0)))),
      // Root is just a dummy node, and has a tensor full of zeros.
      // It cannot be retrieved via Get* methods, only by using the Root()
      // method.
      root_(/*parent=*/nullptr, /*originating_state=*/nullptr,
            /*acting_player=*/player_, /*tensor=*/observation_.Tensor()) {

    int start_max_depth = 0;
    for (const std::unique_ptr<State>& start_state : start_states) {
      start_max_depth = std::max(start_max_depth, start_state->MoveNumber());
    }

    for (std::unique_ptr<State>& start_state : start_states) {
      RecursivelyBuildTree(&root_, std::move(start_state),
                           start_max_depth + max_depth_limit);
    }
  }

  Node* Root() { return &root_; }

  Node* GetByCompressed(const std::string& observation) const {
    auto it = lookup_table_.find(observation);
    return it == lookup_table_.end() ? nullptr : it->second;
  }

  Node* GetByState(const State& state) {
    observation_.SetFrom(state, player_);
    std::string compressed = observation_.Compress();
    return GetByCompressed(compressed);
  }

 private:
  Player player_;
  std::shared_ptr<Observer> infostate_observer_;
  Observation observation_;
  Node root_;
  // Store compressed observations for fast lookup in the lookup table.
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

  void RecursivelyBuildTree(Node* parent, std::unique_ptr<State> state,
                            int move_limit) {
    observation_.SetFrom(*state, player_);
    std::string compressed = observation_.Compress();
    Node* node = GetByCompressed(compressed);
    if (!node) {
      node = parent->AddChild(std::make_unique<Node>(
          parent, state.get(), player_, observation_.Tensor()));
      lookup_table_.insert({std::move(compressed), node});
    }

    // Do not build deeper.
    if (state->MoveNumber() >= move_limit)
      return;

    for (Action a : state->LegalActions()) {
      RecursivelyBuildTree(node, state->Child(a), move_limit);
    }
  }
};

// Provide convenient types for usage in CFR-based algorithms.
using CFRTree = InfostateTree</*NodeContents=*/CFRInfoStateValues>;
using CFRNode = InfostateNode</*Contents=*/CFRInfoStateValues>;

// Specialize CFRNode, because we construct the content
// of CFRInfoStateValues differently for decision nodes.
template<> CFRNode::InfostateNode(
    CFRNode *parent, State* originating_state, Player acting_player,
    absl::Span<float> tensor) :
  content_(originating_state && originating_state->IsPlayerActing(acting_player)
           ? CFRInfoStateValues(originating_state->LegalActions(acting_player))
           : CFRInfoStateValues()),
  parent_(parent),
  type_(GetInfostateNodeType(originating_state, acting_player)),
  tensor_(tensor.begin(), tensor.end()) {}

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
