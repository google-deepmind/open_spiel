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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

// This file contains data structures used in imperfect information games.
// Specifically, we implement an infostate tree, a representation of a game
// from the perspective of an acting player.
//
// The information-state tree [1] contains information states, which describe
// where the player is a) acting, b) getting observations, or c) receiving
// terminal utilities (when the game ends). See `InfostateNodeType` for more
// details.
//
// The tree can be constructed with a depth limit, so we make a distinction
// between leaf nodes and non-leaf nodes. All terminal nodes are leaf nodes.
//
// The identification of infostates is based on strings from an information
// state observer, i.e. one that is constructed using `kInfoStateObsType`.
//
// As algorithms typically need to store information associated to specific
// nodes of the tree, we provide following indexing mechanisms (see the classes
// below for more details):
//
// - `DecisionId` refers to a decision infostate where the player acts.
// - `SequenceId` refers to an observation infostate that follows the decision
//    infostate after some action.
// - `LeafId` refers to an infostate node which is a leaf.
//
// All of these ids are very cheap (they are just typed `size_t`s).
// They can be used to get a pointer to the corresponding infostate node.
//
// To enable some algorithmic optimizations we construct the trees "balanced".
// We call a _balanced_ tree one which has all leaf nodes at the same depth.
// To make the tree balanced, we may need to pad "dummy" observation nodes as
// prefixes for the (previously too shallow) leafs. This is not too expensive,
// as most games are balanced by default due to game rules.
//
// [1]: Rethinking Formal Models of Partially Observable Multiagent Decision
//      Making https://arxiv.org/abs/1906.11110

namespace open_spiel {
namespace algorithms {

// To categorize infostate nodes we use nomenclature from [2]:
//
// - In _decision nodes_, the acting player selects actions.
// - In _observation nodes_ the acting player receives observations.
//   They can correspond to State that is a chance node, or opponent's node.
//   Importantly, they can correspond also to the acting player's node,
//   as the player may have discovered something as a result of its action
//   in the previous decision node. (This is especially important for the tree
//   construction in simultaneous-move games).
// - Additionally, we introduce _terminal nodes_, which correspond to a single
//   terminal history.
//
// The terminal nodes store player's utility as well as cumulative chance reach
// probability.
//
// [2]: Faster Game Solving via Predictive Blackwell Approachability:
//      Connecting Regret Matching and Mirror Descent
//      https://arxiv.org/abs/2007.14358
enum InfostateNodeType {
  kDecisionInfostateNode,
  kObservationInfostateNode,
  kTerminalInfostateNode
};

// Representing the game via infostates leads actually to a graph structure
// of a forest (a collection of trees), as the player may be acting for the
// first time in distinct situations. We trivially make it into a proper tree
// by introducing a "dummy" root node, which we set as an observation node.
// It can be interpreted as "the player observes the start of the game".
// This node also corresponds to the empty sequence.
// Following is the infostate string for this node.
constexpr const char* kDummyRootNodeInfostate = "(root)";

// Sometimes we need to create infostate nodes that do not have a corresponding
// game State, and therefore we cannot retrieve their string representations.
// This happens in simultaneous move games or if we rebalance game trees.
constexpr const char* kFillerInfostate = "(fill)";

// Forward declaration.
class InfostateTree;

namespace internal {

// An implementation detail - Not to be used directly.
//
// We use various indexing schemes (SequenceId, DecisionId, LeafId) to access
// specific nodes in the tree. Not all nodes can have an Id defined, for example
// a DecisionId is not defined for decision nodes. In this case they will
// default to the following value.
constexpr size_t kUndefinedNodeId = -1;  // This is a large number.

// An implementation detail - Not to be used directly.
//
// Create an indexing of specific infostate nodes.
//
// In release-mode the implementation is as cheap as the underlying size_t
// identifier. Therefore it is preferable to pass the Ids by copy and not
// by pointers / references.
// Most importantly in debug-mode we add checks to make sure that we are using
// the ids on appropriate trees and we do not try to index any opponents' trees.
//
// We use CRTP as it allows us to reuse the implementation for derived classes.
template <class Self>
class NodeId {
  size_t identifier_ = kUndefinedNodeId;
#ifndef NDEBUG  // Allow additional automatic debug-time checks.
  const InfostateTree* tree_ = nullptr;

 public:
  NodeId(size_t id_value, const InfostateTree* tree_ptr)
      : identifier_(id_value), tree_(tree_ptr) {}
  NodeId& operator=(Self&& rhs) {
    SPIEL_CHECK_TRUE(
        tree_ == rhs.tree_ ||
        // The NodeId may be uninitialized, so allow to copy the rhs tree.
        tree_ == nullptr && rhs.tree_ != nullptr);
    identifier_ = rhs.id();
    tree_ == rhs.tree_;
    return *this;
  }
  bool operator==(const Self& rhs) const {
    SPIEL_CHECK_EQ(tree_, rhs.tree_);
    return id() == rhs.id();
  }
  bool operator!=(const Self& rhs) const {
    SPIEL_CHECK_EQ(tree_, rhs.tree_);
    return id() != rhs.id();
  }
  bool BelongsToTree(const InfostateTree* other) const {
    return tree_ == other;
  }
#else

 public:
  // Do not save the tree pointer, but expose the same interface
  // so it's easy to use.
  NodeId(size_t id_value, const InfostateTree*) : identifier_(id_value) {}
  Self& operator=(Self&& rhs) {
    identifier_ = rhs.id();
    return this;
  }
  bool operator==(const Self& rhs) const { return id() == rhs.id(); }
  bool operator!=(const Self& rhs) const { return id() != rhs.id(); }
  // BelongsToTree is not implemented on purpose:
  // It must not be called in release mode -- used only by DCHECK statements.
#endif
  constexpr NodeId() {}
  size_t id() const {
    SPIEL_CHECK_NE(identifier_, kUndefinedNodeId);
    return identifier_;
  }
  bool is_undefined() const { return identifier_ == kUndefinedNodeId; }
  void next() {
    SPIEL_CHECK_NE(identifier_, kUndefinedNodeId);
    ++identifier_;
  }
};

}  // namespace internal

// `SequenceId` refers to an observation infostate that follows the decision
// infostate after following some action. It indexes the decision space of
// an agent, and its strategy can formulated in terms of values associated with
// the agent's sequences. See `TreeplexVector` for more details.
// The smallest sequence ids correspond to the deepest nodes and the highest
// value corresponds to the empty sequence.
class SequenceId final : public internal::NodeId<SequenceId> {
  using NodeId<SequenceId>::NodeId;
};
// When the tree is still under construction and a node doesn't
// have a final sequence id assigned yet, we use this value.
constexpr SequenceId kUndefinedSequenceId = SequenceId();

// `DecisionId` refers to an infostate node where the player acts,
// i.e. an infostate node with the type `kDecisionInfostateNode`.
class DecisionId final : public internal::NodeId<DecisionId> {
  using NodeId<DecisionId>::NodeId;
};
// When a node isn't a decision infostate, we use this value instead.
constexpr DecisionId kUndefinedDecisionId = DecisionId();

// `LeafId` refers to an infostate node which is a leaf. Note that this can be
// an arbitrary infostate node type. A kTerminalInfostateNode is always
// a leaf node.
// Note that leaf decision nodes do not have assigned any `DecisionId`, and
// similarly leaf observation nodes do not have assigned any `SequenceId`.
class LeafId final : public internal::NodeId<LeafId> {
  using internal::NodeId<LeafId>::NodeId;
};
// When a node isn't a leaf, we use this value instead.
constexpr LeafId kUndefinedLeafId = LeafId();

// Each of the Ids can be used to index an appropriate vector.
// See below for an implementation.
template <typename T>
class TreeplexVector;
template <typename T>
class LeafVector;
template <typename T>
class DecisionVector;
using SfStrategy = TreeplexVector<double>;

// A convenience iterator over a contiguous range of node ids.
template <class Id>
class RangeIterator {
  size_t id_;
  const InfostateTree* tree_;

 public:
  RangeIterator(size_t id, const InfostateTree* tree) : id_(id), tree_(tree) {}
  RangeIterator& operator++() {
    ++id_;
    return *this;
  }
  bool operator!=(const RangeIterator& other) const {
    return id_ != other.id_ || tree_ != other.tree_;
  }
  Id operator*() { return Id(id_, tree_); }
};
template <class Id>
class Range {
  const size_t start_;
  const size_t end_;
  const InfostateTree* tree_;

 public:
  Range(size_t start, size_t end, const InfostateTree* tree)
      : start_(start), end_(end), tree_(tree) {
    SPIEL_CHECK_LE(start_, end_);
  }
  RangeIterator<Id> begin() const { return RangeIterator<Id>(start_, tree_); }
  RangeIterator<Id> end() const { return RangeIterator<Id>(end_, tree_); }
};

// Forward declaration.
class InfostateNode;

// Creates an infostate tree for a player based on the initial state
// of the game, up to some move limit.
std::shared_ptr<InfostateTree> MakeInfostateTree(const Game& game,
                                                 Player acting_player,
                                                 int max_move_limit = 1000);

// Creates an infostate tree for a player based on some start states,
// up to some move limit from the deepest start state.
std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<const State*>& start_states,
    const std::vector<double>& chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player acting_player,
    int max_move_ahead_limit = 1000);

// Creates an infostate tree based on some leaf infostate nodes coming from
// another infostate tree, up to some move limit.
// This is useful for easily constructing (depth-limited) tree continuations.
std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<const InfostateNode*>& start_nodes,
    int max_move_ahead_limit = 1000);

// C++17 does not allow implicit conversion of non-const pointers to const
// pointers within a vector - explanation: https://stackoverflow.com/a/2102415
// This just adds const to the pointers and calls the other MakeInfostateTree.
std::shared_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<InfostateNode*>& start_nodes,
    int max_move_ahead_limit = 1000);

class InfostateTree final {
  // Note that only MakeInfostateTree is allowed to call the constructor
  // to ensure the trees are always allocated on heap. We do this so that all
  // the collected pointers are valid throughout the tree's lifetime even if
  // they are moved around.
 private:
  InfostateTree(const std::vector<const State*>& start_states,
                const std::vector<double>& chance_reach_probs,
                std::shared_ptr<Observer> infostate_observer,
                Player acting_player, int max_move_ahead_limit);
  // Friend factories.
  friend std::shared_ptr<InfostateTree> MakeInfostateTree(const Game&, Player,
                                                          int);
  friend std::shared_ptr<InfostateTree> MakeInfostateTree(
      const std::vector<const State*>&, const std::vector<double>&,
      std::shared_ptr<Observer>, Player, int);
  friend std::shared_ptr<InfostateTree> MakeInfostateTree(
      const std::vector<const InfostateNode*>&, int);

 public:
  // -- Root accessors ---------------------------------------------------------
  const InfostateNode& root() const { return *root_; }
  InfostateNode* mutable_root() { return root_.get(); }
  int root_branching_factor() const;

  // -- Tree information -------------------------------------------------------
  Player acting_player() const { return acting_player_; }
  // Zero-based height.
  // (the height of a tree that contains only root node is zero.)
  size_t tree_height() const { return tree_height_; }

  // -- General statistics -----------------------------------------------------
  size_t num_decisions() const { return decision_infostates_.size(); }
  size_t num_sequences() const { return sequences_.size(); }
  size_t num_leaves() const { return nodes_at_depths_.back().size(); }
  // A function overload used for TreeVector templates.
  size_t num_ids(DecisionId) const { return num_decisions(); }
  size_t num_ids(SequenceId) const { return num_sequences(); }
  size_t num_ids(LeafId) const { return num_leaves(); }

  // -- Sequence operations ----------------------------------------------------
  SequenceId empty_sequence() const;
  InfostateNode* observation_infostate(const SequenceId& sequence_id) {
    SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
    return sequences_.at(sequence_id.id());
  }
  const InfostateNode* observation_infostate(
      const SequenceId& sequence_id) const {
    SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
    return sequences_.at(sequence_id.id());
  }
  Range<SequenceId> AllSequenceIds() const {
    return Range<SequenceId>(0, sequences_.size(), this);
  }
  // Returns all DecisionIds which can be found in a subtree of given sequence.
  std::vector<DecisionId> DecisionIdsWithParentSeq(const SequenceId&) const;
  // Returns `None` if the sequence is the empty sequence.
  absl::optional<DecisionId> DecisionIdForSequence(const SequenceId&) const;
  // Returns `None` if the sequence is the empty sequence.
  absl::optional<InfostateNode*> DecisionForSequence(const SequenceId&);
  // Returns whether the sequence ends with the last action the player can make.
  bool IsLeafSequence(const SequenceId&) const;

  // -- Decision operations ----------------------------------------------------
  InfostateNode* decision_infostate(const DecisionId& decision_id) {
    SPIEL_DCHECK_TRUE(decision_id.BelongsToTree(this));
    return decision_infostates_.at(decision_id.id());
  }
  const InfostateNode* decision_infostate(const DecisionId& decision_id) const {
    SPIEL_DCHECK_TRUE(decision_id.BelongsToTree(this));
    return decision_infostates_.at(decision_id.id());
  }
  const std::vector<InfostateNode*>& AllDecisionInfostates() const {
    return decision_infostates_;
  }
  Range<DecisionId> AllDecisionIds() const {
    return Range<DecisionId>(0, decision_infostates_.size(), this);
  }
  DecisionId DecisionIdFromInfostateString(
      const std::string& infostate_string) const;

  // -- Leaf operations --------------------------------------------------------
  const std::vector<InfostateNode*>& leaf_nodes() const {
    return nodes_at_depths_.back();
  }
  InfostateNode* leaf_node(const LeafId& leaf_id) const {
    SPIEL_DCHECK_TRUE(leaf_id.BelongsToTree(this));
    return leaf_nodes().at(leaf_id.id());
  }
  const std::vector<std::vector<InfostateNode*>>& nodes_at_depths() const {
    return nodes_at_depths_;
  }
  const std::vector<InfostateNode*>& nodes_at_depth(size_t depth) const {
    return nodes_at_depths_.at(depth);
  }

  // -- Tree operations --------------------------------------------------------
  // Compute best response and value based on gradient from opponents.
  // This consumes the gradient vector, as it is used to compute the value.
  std::pair<double, SfStrategy> BestResponse(
      TreeplexVector<double>&& gradient) const;
  // Compute best response value based on gradient from opponents over leaves.
  // This consumes the gradient vector, as it is used to compute the value.
  double BestResponseValue(LeafVector<double>&& gradient) const;

  // -- For debugging ----------------------------------------------------------
  std::ostream& operator<<(std::ostream& os) const;

 private:
  const Player acting_player_;
  const std::shared_ptr<Observer> infostate_observer_;
  const std::unique_ptr<InfostateNode> root_;
  /*const*/ size_t tree_height_ = 0;

  // Tree structure collections that index the respective NodeIds.
  std::vector<InfostateNode*> decision_infostates_;
  std::vector<InfostateNode*> sequences_;
  // The last vector corresponds to the leaf nodes.
  std::vector<std::vector<InfostateNode*>> nodes_at_depths_;

  // Utility functions whenever we create a new node for the tree.
  std::unique_ptr<InfostateNode> MakeNode(InfostateNode* parent,
                                          InfostateNodeType type,
                                          const std::string& infostate_string,
                                          double terminal_utility,
                                          double terminal_ch_reach_prob,
                                          size_t depth,
                                          const State* originating_state);
  std::unique_ptr<InfostateNode> MakeRootNode() const;

  // Makes sure that all tree leaves are at the same height.
  // It inserts a linked list of dummy observation nodes with appropriate length
  // to balance all the leaves.
  void RebalanceTree();

  void UpdateLeafNode(InfostateNode* node, const State& state,
                      size_t leaf_depth, double chance_reach_probs);

  // Build the tree.
  void RecursivelyBuildTree(InfostateNode* parent, size_t depth,
                            const State& state, int move_limit,
                            double chance_reach_prob);
  void BuildTerminalNode(InfostateNode* parent, size_t depth,
                         const State& state, double chance_reach_prob);
  void BuildDecisionNode(InfostateNode* parent, size_t depth,
                         const State& state, int move_limit,
                         double chance_reach_prob);
  void BuildObservationNode(InfostateNode* parent, size_t depth,
                            const State& state, int move_limit,
                            double chance_reach_prob);

  void CollectNodesAtDepth(InfostateNode* node, size_t depth);
  void LabelNodesWithIds();
  std::pair<size_t, size_t> CollectStartEndSequenceIds(
      InfostateNode* node, const SequenceId parent_sequence);
};

// Iterate over a vector of unique pointers, but expose only the raw pointers.
template <class T>
class VecWithUniquePtrsIterator {
  int pos_;
  const std::vector<std::unique_ptr<T>>& vec_;

 public:
  explicit VecWithUniquePtrsIterator(const std::vector<std::unique_ptr<T>>& vec,
                                     int pos = 0)
      : pos_(pos), vec_(vec) {}
  VecWithUniquePtrsIterator& operator++() {
    pos_++;
    return *this;
  }
  bool operator==(VecWithUniquePtrsIterator other) const {
    return pos_ == other.pos_;
  }
  bool operator!=(VecWithUniquePtrsIterator other) const {
    return !(*this == other);
  }
  T* operator*() { return vec_[pos_].get(); }
  VecWithUniquePtrsIterator begin() const { return *this; }
  VecWithUniquePtrsIterator end() const {
    return VecWithUniquePtrsIterator(vec_, vec_.size());
  }
};

class InfostateNode final {
  // Note that all of the following members are const or they should be const.
  // However we can't make all of  them const during the node construction
  // because they might be computed only after the whole tree is built.
 private:
  // Reference to the tree this node belongs to. This reference has a valid
  // lifetime, as it is allocated once on the heap and never moved.
  const InfostateTree& tree_;
  // Pointer to the parent node. Null for the root node.
  // This is not const so that we can change it when we rebalance the tree.
  /*const*/ InfostateNode* parent_;
  // Position of this node in the parent's children, i.e. it holds that
  //   parent_->children_.at(incoming_index_).get() == this.
  //
  // For decision nodes this corresponds also to the
  //   State::LegalActions(player_).at(incoming_index_)
  //
  // This is not const so that we can change it when we rebalance the tree.
  /*const*/ int incoming_index_;
  // Type of the node.
  const InfostateNodeType type_;
  // Identifier of the infostate.
  const std::string infostate_string_;
  // Decision identifier of this node.
  // This is not const as the ids are assigned after the tree is built.
  /*const*/ DecisionId decision_id_ = kUndefinedDecisionId;
  // Sequence identifier of this node.
  // The first is the parent sequence of the infostate, while the last
  // two sequence IDs represent the sequence id of the first and last action + 1
  // at the infostate node. Because sequences assigned to an infostate
  // are contiguous, we don't need to store all intermediate sequence IDs.
  // We can thus use a Range iterable to make looping frictionless.
  // This is not const as the ids can be assigned only after the tree is built.
  /*const*/ SequenceId sequence_id_ = kUndefinedSequenceId;
  /*const*/ SequenceId start_sequence_id_ = kUndefinedSequenceId;
  /*const*/ SequenceId end_sequence_id_ = kUndefinedSequenceId;
  // Sequence identifier of this node.
  // This is not const as the ids are assigned after the tree is rebalanced.
  /*const*/ LeafId leaf_id_ = kUndefinedLeafId;
  // Utility of terminal state corresponding to the terminal infostate node.
  // If the node is not terminal, the value is NaN.
  const double terminal_utility_;
  // Cumulative product of chance probabilities leading up to a terminal node.
  // If the node is not terminal, the value is NaN.
  const double terminal_chn_reach_prob_;
  // Depth of the node, i.e. number of edges on the path from the root.
  // Note that depth does not necessarily correspond to the MoveNumber()
  // of corresponding states.
  // This is not const because tree rebalancing can change this value.
  /*const*/ size_t depth_;
  // Children infostate nodes. Notice the node owns its children.
  // This is not const so that we can add children.
  /*const*/ std::vector<std::unique_ptr<InfostateNode>> children_;
  // Store States that correspond to a leaf node.
  // This is not const so that we can add corresponding states.
  /*const*/ std::vector<std::unique_ptr<State>> corresponding_states_;
  // Store chance reach probs for States that correspond to a leaf node.
  // This is not const so that we can add corresponding reaches.
  /*const*/ std::vector<double> corresponding_ch_reaches_;
  // Stored only for decision nodes.
  const std::vector<Action> legal_actions_;
  // Stored only for terminal nodes.
  const std::vector<Action> terminal_history_;

  // Only InfostateTree is allowed to construct nodes.
  InfostateNode(const InfostateTree& tree, InfostateNode* parent,
                int incoming_index, InfostateNodeType type,
                const std::string& infostate_string, double terminal_utility,
                double terminal_ch_reach_prob, size_t depth,
                std::vector<Action> legal_actions,
                std::vector<Action> terminal_history);
  friend class InfostateTree;

 public:
  // -- Node accessors. --------------------------------------------------------
  const InfostateTree& tree() const { return tree_; }
  InfostateNode* parent() const { return parent_; }
  int incoming_index() const { return incoming_index_; }
  const InfostateNodeType& type() const { return type_; }
  size_t depth() const { return depth_; }
  bool is_root_node() const { return !parent_; }
  bool has_infostate_string() const {
    return infostate_string_ != kFillerInfostate &&
           infostate_string_ != kDummyRootNodeInfostate;
  }
  const std::string& infostate_string() const {
    // Avoid working with empty infostate strings.
    SPIEL_DCHECK_TRUE(has_infostate_string());
    return infostate_string_;
  }

  // -- Children accessors. ----------------------------------------------------
  InfostateNode* child_at(int i) const { return children_.at(i).get(); }
  int num_children() const { return children_.size(); }
  VecWithUniquePtrsIterator<InfostateNode> child_iterator() const {
    return VecWithUniquePtrsIterator<InfostateNode>(children_);
  }

  // -- Sequence operations. ---------------------------------------------------
  const SequenceId sequence_id() const {
    SPIEL_CHECK_FALSE(sequence_id_.is_undefined());
    return sequence_id_;
  }
  const SequenceId start_sequence_id() const {
    SPIEL_CHECK_FALSE(start_sequence_id_.is_undefined());
    return start_sequence_id_;
  }
  const SequenceId end_sequence_id() const {
    SPIEL_CHECK_FALSE(end_sequence_id_.is_undefined());
    return end_sequence_id_;
  }
  Range<SequenceId> AllSequenceIds() const {
    return Range<SequenceId>(start_sequence_id_.id(), end_sequence_id_.id(),
                             &tree_);
  }

  // -- Decision operations. ---------------------------------------------------
  const DecisionId decision_id() const {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    SPIEL_CHECK_FALSE(decision_id_.is_undefined());
    return decision_id_;
  }
  const std::vector<Action>& legal_actions() const {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return legal_actions_;
  }

  // -- Leaf operations. -------------------------------------------------------
  bool is_leaf_node() const { return children_.empty(); }
  double terminal_utility() const {
    SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
    return terminal_utility_;
  }
  double terminal_chance_reach_prob() const {
    SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
    return terminal_chn_reach_prob_;
  }
  size_t corresponding_states_size() const {
    return corresponding_states_.size();
  }
  const std::vector<std::unique_ptr<State>>& corresponding_states() const {
    SPIEL_CHECK_TRUE(is_leaf_node());
    return corresponding_states_;
  }
  const std::vector<double>& corresponding_chance_reach_probs() const {
    SPIEL_CHECK_TRUE(is_leaf_node());
    return corresponding_ch_reaches_;
  }
  const std::vector<Action>& TerminalHistory() const {
    SPIEL_DCHECK_EQ(type_, kTerminalInfostateNode);
    return terminal_history_;
  }

  // -- For debugging. ---------------------------------------------------------
  std::ostream& operator<<(std::ostream& os) const;
  // Make subtree certificate (string representation) for easy comparison
  // of (isomorphic) trees.
  std::string MakeCertificate() const;

 private:
  // Make sure that the subtree ends at the requested target depth by inserting
  // dummy observation nodes with one outcome.
  void RebalanceSubtree(int target_depth, int current_depth);

  // Get the unique_ptr for this node. The usage is intended only for tree
  // balance manipulation.
  std::unique_ptr<InfostateNode> Release();

  // Change the parent of this node by inserting it at at index
  // of the new parent. The node at the existing position will be freed.
  // We pass the unique ptr of itself, because calling Release might be
  // undefined: the node we want to swap a parent for can be root of a subtree.
  void SwapParent(std::unique_ptr<InfostateNode> self, InfostateNode* target,
                  int at_index);

  InfostateNode* AddChild(std::unique_ptr<InfostateNode> child);
  InfostateNode* GetChild(const std::string& infostate_string) const;
};

namespace internal {

// An implementation detail - Not to be used directly.
//
// Create a common TreeVector container that can be indexed
// with the respective NodeIds. This is later specialized for the individual
// indexing of the trees.
template <typename T, typename Id>
class TreeVector {
  const InfostateTree* tree_;
  std::vector<T> vec_;

 public:
  explicit TreeVector(const InfostateTree* tree)
      : tree_(tree), vec_(tree_->num_ids(Id(kUndefinedNodeId, tree))) {}
  TreeVector(const InfostateTree* tree, std::vector<T> vec)
      : tree_(tree), vec_(std::move(vec)) {
    SPIEL_CHECK_EQ(tree_->num_ids(Id(kUndefinedNodeId, tree)), vec_.size());
  }
  T& operator[](const Id& id) {
    SPIEL_DCHECK_TRUE(id.BelongsToTree(tree_));
    SPIEL_DCHECK_LE(0, id.id());
    SPIEL_DCHECK_LT(id.id(), vec_.size());
    return vec_[id.id()];
  }
  const T& operator[](const Id& id) const {
    SPIEL_DCHECK_TRUE(id.BelongsToTree(tree_));
    SPIEL_DCHECK_LE(0, id.id());
    SPIEL_DCHECK_LT(id.id(), vec_.size());
    return vec_[id.id()];
  }
  std::ostream& operator<<(std::ostream& os) const {
    return os << vec_ << " (for player " << tree_->acting_player() << ')';
  }
  size_t size() const { return vec_.size(); }
  Range<Id> range() { return Range<Id>(0, vec_.size(), tree_); }
  Range<Id> range(size_t from, size_t to) { return Range<Id>(from, to, tree_); }
  const InfostateTree* tree() const { return tree_; }
};

}  // namespace internal

// Arrays that can be easily indexed by SequenceIds.
// The space of all such arrays forms a treeplex [3].
//
// [3]: Smoothing Techniques for Computing Nash Equilibria of Sequential Games
//      http://www.cs.cmu.edu/~sandholm/proxtreeplex.MathOfOR.pdf
template <typename T>
class TreeplexVector final : public internal::TreeVector<T, SequenceId> {
  using internal::TreeVector<T, SequenceId>::TreeVector;
};

// Arrays that can be easily indexed by LeafIds.
template <typename T>
class LeafVector final : public internal::TreeVector<T, LeafId> {
  using internal::TreeVector<T, LeafId>::TreeVector;
};

// Arrays that can be easily indexed by DecisionIds.
template <typename T>
class DecisionVector final : public internal::TreeVector<T, DecisionId> {
  using internal::TreeVector<T, DecisionId>::TreeVector;
};

// Returns whether the supplied vector is a valid sequence-form strategy:
// The probability flow has to sum up to 1 and each sequence's incoming
// probability must be equal to outgoing probabilities.
bool IsValidSfStrategy(const SfStrategy& strategy);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
