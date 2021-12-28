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

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr const char* kImperfectInfoGoofSpiel2(
    "goofspiel("
    "num_cards=2,"
    "imp_info=True,"
    "points_order=ascending"
    ")");

constexpr const char* kImperfectInfoGoofSpiel3(
    "goofspiel("
    "num_cards=3,"
    "imp_info=True,"
    "points_order=ascending"
    ")");

bool IsNodeBalanced(const InfostateNode& node, int height,
                    int current_depth = 0) {
  if (node.is_leaf_node()) return height == current_depth;

  for (const InfostateNode* child : node.child_iterator()) {
    if (!IsNodeBalanced(*child, height, current_depth + 1)) {
      return false;
    }
  }

  return true;
}

bool RecomputeBalance(const InfostateTree& tree) {
  return IsNodeBalanced(tree.root(), tree.tree_height());
}

std::shared_ptr<InfostateTree> MakeTree(const std::string& game_name,
                                        Player player,
                                        int max_move_limit = 1000) {
  std::shared_ptr<InfostateTree> tree =
      MakeInfostateTree(*LoadGame(game_name), player, max_move_limit);
  SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  return tree;
}

std::shared_ptr<InfostateTree> MakeTree(
    const std::string& game_name, Player player,
    const std::vector<std::vector<Action>>& start_histories,
    const std::vector<double>& start_reaches, int max_move_limit = 1000) {
  const std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<std::unique_ptr<State>> start_states;
  std::vector<const State*> start_state_ptrs;
  for (const std::vector<Action>& history : start_histories) {
    std::unique_ptr<State> rollout = game->NewInitialState();
    for (const Action& a : history) rollout->ApplyAction(a);
    start_states.push_back(std::move(rollout));
    start_state_ptrs.push_back(start_states.back().get());
  }

  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  std::shared_ptr<InfostateTree> tree =
      MakeInfostateTree(start_state_ptrs, start_reaches, infostate_observer,
                        player, max_move_limit);
  SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  return tree;
}

void TestRootCertificates() {
  {
    std::string expected_certificate =
        "(["
        "({}{})"  // Play Heads: HH, HT
        "({}{})"  // Play Tails: TH, TT
        "])";
    for (int i = 0; i < 2; ++i) {
      std::shared_ptr<InfostateTree> tree = MakeTree("matrix_mp", /*player=*/i);
      SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
    }
  }
  {
    std::string expected_certificate =
        "(["
        "({}{})"  // Play 1: draw 1,1  lose 1,2
        "({}{})"  // Play 2: win  2,1  draw 2,2
        "])";
    for (int i = 0; i < 2; ++i) {
      std::shared_ptr<InfostateTree> tree =
          MakeTree(kImperfectInfoGoofSpiel2, /*player=*/i);
      SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
    }
  }
  {  // Full Kuhn test.
    std::shared_ptr<InfostateTree> tree = MakeTree("kuhn_poker", /*player=*/0);
    std::string expected_certificate =
        // Notice all terminals are at the same depth (same indentation).
        "(("  // Root node, 1st is getting a card
        "("   // 2nd is getting card
        "["   // 1st acts
        "(("  // 1st bet, and 2nd acts
        "(({}))"
        "(({}))"
        "(({}))"
        "(({}))"
        "))"
        "(("  // 1st checks, and 2nd acts
              // 2nd checked
        "(({}))"
        "(({}))"
        // 2nd betted
        "[({}"
        "{})"
        "({}"
        "{})]"
        "))"
        "]"
        ")"
        // Just 2 more copies.
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        "))";
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "((("  // Root node, distribute cards.
        "("    // 1st acts
               // 1st betted
        "[(({})({}))(({})({}))]"
        // 1st checked
        "[(({})({}))(({}{}{}{}))]"
        ")"
        // Just 2 more copies.
        "([(({})({}))(({})({}))][(({})({}))(({}{}{}{}))])"
        "([(({})({}))(({})({}))][(({})({}))(({}{}{}{}))])"
        ")))";
    std::shared_ptr<InfostateTree> tree = MakeTree("kuhn_poker", /*player=*/1);
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "(["
        "("  // Play 2
        "[({}{})({}{})]"
        "[({}{})({}{})]"
        "[({}{})({}{})]"
        ")"
        "("  // Play 1
        "[({}{})({}{})]"
        "[({}{}{}{})({}{}{}{})]"
        ")"
        "("  // Play 3
        "[({}{})({}{})]"
        "[({}{}{}{})({}{}{}{})]"
        ")"
        "])";
    for (int i = 0; i < 2; ++i) {
      std::shared_ptr<InfostateTree> tree =
          MakeTree(kImperfectInfoGoofSpiel3, /*player=*/i);
      SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
    }
  }
}

void TestCertificatesFromStartHistories() {
  {
    std::shared_ptr<InfostateTree> tree = MakeTree(
        "kuhn_poker", /*player=*/0, /*start_histories=*/{{0, 1, 0}}, {1 / 6.});
    std::string expected_certificate =
        "(("
        "(({}))"      // 2nd player passes
        "[({})({})]"  // 2nd player bets
        "))";
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        ")";
    std::shared_ptr<InfostateTree> tree =
        MakeTree("kuhn_poker", /*player=*/0,
                 /*start_histories=*/{{0}, {2}}, {1 / 3., 1 / 3.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "([(({}))(({}))][(({}))(({}{}))])"
        "([(({}))(({}))][(({}))(({}{}))])"
        ")";
    std::shared_ptr<InfostateTree> tree =
        MakeTree("kuhn_poker", /*player=*/1,
                 /*start_histories=*/{{1, 0}, {1, 2}}, {1 / 6., 1 / 6.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        "[((((({})))))((((({})))))]"
        ")";
    std::shared_ptr<InfostateTree> tree =
        MakeTree("kuhn_poker", /*player=*/0,
                 /*start_histories=*/{{0}, {2, 1, 0, 1}},
                 /*start_reaches=*/{1 / 3., 1 / 6.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "(((({})))((({}))))"
        "([(({}))(({}))][(({}))(({}{}))])"
        ")";
    std::shared_ptr<InfostateTree> tree = MakeTree(
        "kuhn_poker", /*player=*/1, /*start_histories=*/{{1, 0}, {1, 2, 0, 1}},
        /*start_reaches=*/{1 / 6., 1 / 6.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "[({}{})({}{})]"
        ")";
    std::shared_ptr<InfostateTree> tree =
        MakeTree("kuhn_poker", /*player=*/0,
                 /*start_histories=*/{{0, 1, 0, 1}, {0, 2, 0, 1}},
                 /*start_reaches=*/{1 / 6., 1 / 6.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::string expected_certificate =
        "("
        "({}{})"
        "({}{})"
        ")";
    std::shared_ptr<InfostateTree> tree =
        MakeTree("kuhn_poker", /*player=*/1,
                 /*start_histories=*/{{0, 1, 0, 1}, {0, 2, 0, 1}},
                 /*start_reaches=*/{1 / 6., 1 / 6.});
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
  {
    std::shared_ptr<InfostateTree> tree = MakeTree(
        kImperfectInfoGoofSpiel3, /*player=*/0,
        /*start_histories=*/{{0 /* = 0 0 */}, {1 /* = 1 0 */, 3 /* = 2 2 */}},
        /*start_reaches=*/{1., 1.});
    std::string expected_certificate =
        "("
        "(({}))"
        "[({}{})({}{})]"
        ")";
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);
  }
}

void CheckTreeLeaves(const InfostateTree& tree, int move_limit) {
  for (InfostateNode* leaf_node : tree.leaf_nodes()) {
    SPIEL_CHECK_TRUE(leaf_node->is_leaf_node());
    SPIEL_CHECK_TRUE(leaf_node->has_infostate_string());
    SPIEL_CHECK_FALSE(leaf_node->corresponding_states().empty());

    // Check MoveNumber() for all corresponding states.
    //
    // The conditions are following:
    // - either all states are terminal, and have the same MoveNumber() that
    //   is less or equal to move_limit,
    // - or not all states are terminal and the MoveNumber() == move_limit.

    const int num_states = leaf_node->corresponding_states().size();
    int terminal_cnt = 0;
    int max_move_number = std::numeric_limits<int>::min();
    int min_move_number = std::numeric_limits<int>::max();
    for (const std::unique_ptr<State>& state :
         leaf_node->corresponding_states()) {
      if (state->IsTerminal()) terminal_cnt++;
      max_move_number = std::max(max_move_number, state->MoveNumber());
      min_move_number = std::min(min_move_number, state->MoveNumber());
    }
    SPIEL_CHECK_TRUE(terminal_cnt == 0 || terminal_cnt == num_states);
    SPIEL_CHECK_TRUE(max_move_number == min_move_number);
    if (terminal_cnt == 0) {
      SPIEL_CHECK_EQ(max_move_number, move_limit);
    } else {
      SPIEL_CHECK_LE(max_move_number, move_limit);
    }
  }
}

void CheckContinuation(const InfostateTree& tree) {
  const std::vector<InfostateNode*>& leaves =
      tree.nodes_at_depth(tree.tree_height());
  std::shared_ptr<InfostateTree> continuation = MakeInfostateTree(leaves);

  SPIEL_CHECK_EQ(continuation->root_branching_factor(), leaves.size());
  for (int i = 0; i < leaves.size(); ++i) {
    const InfostateNode* leaf_node = leaves[i];
    const InfostateNode* root_node = continuation->root().child_at(i);
    SPIEL_CHECK_TRUE(leaf_node->is_leaf_node());
    if (leaf_node->type() != kTerminalInfostateNode) {
      SPIEL_CHECK_EQ(leaf_node->type(), root_node->type());
      SPIEL_CHECK_EQ(leaf_node->has_infostate_string(),
                     root_node->has_infostate_string());
      if (leaf_node->has_infostate_string()) {
        SPIEL_CHECK_EQ(leaf_node->infostate_string(),
                       root_node->infostate_string());
      }
    } else {
      // If the leaf node is terminal, the continuation might put this node
      // deeper than in the root due to tree balancing with other leaf
      // non-terminal nodes. Therefore we check whether (the possibly occurring)
      // chain of dummy observations leads to this terminal node.
      InfostateNode* terminal_continuation = continuation->root().child_at(i);
      while (terminal_continuation->type() == kObservationInfostateNode) {
        SPIEL_CHECK_FALSE(terminal_continuation->is_leaf_node());
        SPIEL_CHECK_EQ(terminal_continuation->num_children(), 1);
        terminal_continuation = terminal_continuation->child_at(0);
      }
      SPIEL_CHECK_EQ(terminal_continuation->type(), kTerminalInfostateNode);
      SPIEL_CHECK_EQ(leaf_node->has_infostate_string(),
                     terminal_continuation->has_infostate_string());
      if (leaf_node->has_infostate_string()) {
        SPIEL_CHECK_EQ(leaf_node->infostate_string(),
                       terminal_continuation->infostate_string());
      }
      SPIEL_CHECK_EQ(leaf_node->terminal_utility(),
                     terminal_continuation->terminal_utility());
      SPIEL_CHECK_EQ(leaf_node->terminal_chance_reach_prob(),
                     terminal_continuation->terminal_chance_reach_prob());
      SPIEL_CHECK_EQ(leaf_node->TerminalHistory(),
                     terminal_continuation->TerminalHistory());
    }
  }
}

void BuildAllDepths(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int max_moves = game->MaxMoveNumber();
  for (int move_limit = 0; move_limit < max_moves; ++move_limit) {
    for (int pl = 0; pl < game->NumPlayers(); ++pl) {
      std::shared_ptr<InfostateTree> tree = MakeTree(game_name, pl, move_limit);
      CheckTreeLeaves(*tree, move_limit);
      CheckContinuation(*tree);
    }
  }
}

void TestDepthLimitedTrees() {
  {
    std::string expected_certificate =
        "("  // <dummy>
        "("  // 1st is getting a card
        "("  // 2nd is getting card
        "["  // 1st acts - Node J
             // Depth cutoff.
        "]"
        ")"
        // Repeat the same for the two other cards.
        "([])"  // Node Q
        "([])"  // Node K
        ")"
        ")";  // </dummy>
    std::shared_ptr<InfostateTree> tree = MakeTree("kuhn_poker", 0, 2);
    SPIEL_CHECK_EQ(tree->root().MakeCertificate(), expected_certificate);

    for (InfostateNode* acting : tree->leaf_nodes()) {
      SPIEL_CHECK_TRUE(acting->is_leaf_node());
      SPIEL_CHECK_EQ(acting->type(), kDecisionInfostateNode);
      SPIEL_CHECK_EQ(acting->corresponding_states().size(), 2);
      SPIEL_CHECK_TRUE(acting->has_infostate_string());
    }
  }

  BuildAllDepths("kuhn_poker");
  BuildAllDepths("kuhn_poker(players=3)");
  BuildAllDepths("leduc_poker");
  BuildAllDepths("goofspiel(players=2,num_cards=3,imp_info=True)");
  BuildAllDepths("goofspiel(players=3,num_cards=3,imp_info=True)");
}

void TestDepthLimitedSubgames() {
  {
    std::array<std::string, 4> expected_certificates = {
        "(()()())", "(([][])([][])([][]))",
        "("
        "([(())({}{})][({}{})({}{})])"
        "([(())({}{})][({}{})({}{})])"
        "([(())({}{})][({}{})({}{})])"
        ")",
        "("
        "([(({})({}))(({})({}))][(({})({}))(({}{}{}{}))])"
        "([(({})({}))(({})({}))][(({})({}))(({}{}{}{}))])"
        "([(({})({}))(({})({}))][(({})({}))(({}{}{}{}))])"
        ")"};
    std::array<int, 4> expected_leaf_counts = {3, 6, 21, 30};

    for (int move_limit = 0; move_limit < 4; ++move_limit) {
      std::shared_ptr<InfostateTree> tree =
          MakeTree("kuhn_poker", /*player=*/1,
                   {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}},
                   {1 / 6., 1 / 6., 1 / 6., 1 / 6., 1 / 6., 1 / 6.},
                   /*max_move_limit=*/move_limit);
      SPIEL_CHECK_EQ(tree->root().MakeCertificate(),
                     expected_certificates[move_limit]);
      SPIEL_CHECK_EQ(tree->num_leaves(), expected_leaf_counts[move_limit]);

      for (InfostateNode* leaf : tree->leaf_nodes()) {
        SPIEL_CHECK_EQ(leaf->depth(), tree->tree_height());
      }
    }
  }
}

void TestSequenceIdLabeling() {
  for (int pl = 0; pl < 2; ++pl) {
    std::shared_ptr<InfostateTree> tree = MakeTree("kuhn_poker", /*player=*/pl);

    for (int depth = 0; depth <= tree->tree_height(); ++depth) {
      for (InfostateNode* node : tree->nodes_at_depth(depth)) {
        SPIEL_CHECK_LE(node->start_sequence_id().id(),
                       node->sequence_id().id());
        SPIEL_CHECK_LE(node->end_sequence_id().id(), node->sequence_id().id());
      }
    }

    // Check labeling was done from the deepest nodes.
    size_t depth = -1;  // Some large number.
    for (SequenceId id : tree->AllSequenceIds()) {
      InfostateNode* node = tree->observation_infostate(id);
      SPIEL_CHECK_LE(node->depth(), depth);
      depth = node->depth();
      // Longer sequences (extensions) must have the corresponding
      // infostate nodes placed deeper.
      for (SequenceId extension : node->AllSequenceIds()) {
        InfostateNode* child = tree->observation_infostate(extension);
        SPIEL_CHECK_LT(node->depth(), child->depth());
      }
    }
  }
}

void TestBestResponse() {
  std::shared_ptr<InfostateTree> tree0 = MakeTree("matrix_mp", /*player=*/0);
  std::shared_ptr<InfostateTree> tree1 = MakeTree("matrix_mp", /*player=*/1);
  for (double alpha = 0; alpha < 1.; alpha += 0.1) {
    const double br_value = std::fmax(2 * alpha - 1, -2 * alpha + 1);
    {
      LeafVector<double> grad(tree0.get(),
                              {
                                  1. * alpha,          // Head, Head
                                  -1. * (1. - alpha),  // Head, Tail
                                  -1. * alpha,         // Tail, Head
                                  1. * (1. - alpha),   // Tail, Tail
                              });
      SPIEL_CHECK_FLOAT_EQ(tree0->BestResponseValue(std::move(grad)), br_value);
    }
    {
      LeafVector<double> grad(tree1.get(),
                              {
                                  -1. * alpha,         // Head, Head
                                  1. * (1. - alpha),   // Tail, Head
                                  1. * alpha,          // Head, Tail
                                  -1. * (1. - alpha),  // Tail, Tail
                              });
      SPIEL_CHECK_FLOAT_EQ(tree1->BestResponseValue(std::move(grad)), br_value);
    }
    {
      TreeplexVector<double> grad(tree0.get(),
                                  {-1. + 2. * alpha, 1. - 2. * alpha, 0.});
      std::pair<double, SfStrategy> actual_response =
          tree0->BestResponse(std::move(grad));
      SPIEL_CHECK_FLOAT_EQ(actual_response.first, br_value);
    }
    {
      TreeplexVector<double> grad(tree1.get(),
                                  {1. - 2. * alpha, -1. + 2. * alpha, 0.});
      std::pair<double, SfStrategy> actual_response =
          tree1->BestResponse(std::move(grad));
      SPIEL_CHECK_FLOAT_EQ(actual_response.first, br_value);
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestRootCertificates();
  open_spiel::algorithms::TestCertificatesFromStartHistories();
  open_spiel::algorithms::TestDepthLimitedTrees();
  open_spiel::algorithms::TestDepthLimitedSubgames();
  open_spiel::algorithms::TestSequenceIdLabeling();
  open_spiel::algorithms::TestBestResponse();
}
