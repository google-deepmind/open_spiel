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

#include "open_spiel/algorithms/infostate_tree.h"

#include <algorithm>

#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

std::string iigs2 = "goofspiel("
                      "num_cards=2,"
                      "imp_info=True,"
                      "points_order=ascending"
                    ")";
std::string iigs3 = "goofspiel("
                      "num_cards=3,"
                      "imp_info=True,"
                      "points_order=ascending"
                    ")";

// To make sure we can easily test infostate tree structures, we compute their
// certificate (string representation) that we can easily compare.
std::string ComputeCertificate(const CFRNode& node) {
  if (node.Type() == kTerminalInfostateNode) {
    SPIEL_CHECK_TRUE(node.IsLeafNode());
    return "{}";
  }

  std::vector<std::string> certificates;
  for (CFRNode& child : node.child_iterator()) {
    certificates.push_back(ComputeCertificate(child));
  }
  std::sort(certificates.begin(), certificates.end());

  std::string open, close;
  if (node.Type() == kDecisionInfostateNode) {
    open = "[";
    close = "]";
  } else if (node.Type() == kObservationInfostateNode) {
    open = "(";
    close = ")";
  }

  return absl::StrCat(
      open,
      absl::StrJoin(certificates.begin(), certificates.end(), ""),
      close);
}

std::unique_ptr<CFRTree> MakeTree(const std::string& game_name,
                                  Player player_id,
                                  int max_depth_limit = 1000) {
  return std::make_unique<CFRTree>(*LoadGame(game_name), player_id,
                                   max_depth_limit);
}

std::unique_ptr<CFRTree> MakeTree(
    const std::string& game_name, Player player_id,
    const std::vector<std::vector<Action>>& start_histories,
    const std::vector<double>& start_reaches) {
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

  return std::make_unique<CFRTree>(
      absl::MakeSpan(start_state_ptrs), absl::MakeSpan(start_reaches),
      infostate_observer, player_id);
}

bool IsNodeBalanced(const CFRNode& node, int height, int current_depth = 0) {
  if (node.IsLeafNode()) return height == current_depth;

  for (const CFRNode& child : node.child_iterator())
    if (!IsNodeBalanced(child, height, current_depth + 1))
      return false;

  return true;
}

bool RecomputeBalance(const CFRTree& tree) {
  return IsNodeBalanced(tree.Root(), tree.TreeHeight());
}

void TestRootCertificates() {
  {
    std::string expected_certificate =
      "("  // <dummy>
        "("  // 1st is getting a card
          "("  // 2nd is getting card
            "["  // 1st acts
              "("  // 1st passed
                "("  // 2nd acts
                  "["  // 1st acts (2nd bet)
                      "({}{})({}{})"
                  "]"
                  // 2nd passed too
                  "{}"
                  "{}"
                ")"
              ")"
              "("  // 1st bet
                "("  // 2nd acts
                  "{}{}{}{}"  // 2nd bet too or passed.
                ")"
              ")"
            "]"
          ")"
          // Repeat the same for the two other cards.
          "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
          "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
        ")"
      ")";  // </dummy>
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0);
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "((("  // Dummy, distribute cards.
        "("  // 1st acts
          // 1st passed
          "["
            "(({}{}{}{}))"  // 2nd bets.
            "({}{})"  // 2nd passes.
          "]"
          // 1st bet
          "[({}{})({}{})]"
        ")"
        // Repeat the same for the two other cards.
        "([(({}{}{}{}))({}{})][({}{})({}{})])"
        "([(({}{}{}{}))({}{})][({}{})({}{})])"
      ")))";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 1);
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "(["
        "({}{})"  // Play 1: draw 1,1  lose 1,2
        "({}{})"  // Play 2: win  2,1  draw 2,2
      "])";
    for (int i = 0; i < 2; ++i) {
      std::unique_ptr<CFRTree> tree = MakeTree(iigs2, i);
      SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
      SPIEL_CHECK_TRUE(tree->IsBalanced());
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
    }
  }
  {
    std::string expected_certificate =
      "(["
        "(" // Play 2
          "[({}{})({}{})]"
          "[({}{})({}{})]"
          "[({}{})({}{})]"
        ")"
        "(" // Play 1
          "[({}{})({}{})]"
          "[({}{}{}{})({}{}{}{})]"
        ")"
        "(" // Play 3
          "[({}{})({}{})]"
          "[({}{}{}{})({}{}{}{})]"
        ")"
      "])";
    for (int i = 0; i < 2; ++i) {
      std::unique_ptr<CFRTree> tree = MakeTree(iigs3, i);
      SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
      SPIEL_CHECK_TRUE(tree->IsBalanced());
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
    }
  }
  {
    std::string expected_certificate =
      "(["
        "({}{})"  // Play Heads: HH, HT
        "({}{})"  // Play Tails: TH, TT
      "])";
    for (int i = 0; i < 2; ++i) {
      std::unique_ptr<CFRTree> tree = MakeTree("matrix_mp", i);
      SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
      SPIEL_CHECK_TRUE(tree->IsBalanced());
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
    }
  }
}

void TestCertificatesFromStartHistories() {
  {
    std::string expected_certificate =
      "("
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0,
                                             {{0}, {2}}, {1/3., 1/3.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "("
        "([(({}{}))({})][({})({})])"
        "([(({}{}))({})][({})({})])"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 1,
                                             {{1, 0}, {1, 2}}, {1/6., 1/6.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "("
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
        "[({})({})]"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0,
                                             {{0}, {2, 1, 0, 1}}, {1/3., 1/6.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "("
        "([(({}{}))({})][({})({})])"
        "({}{})"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 1,
                                             {{1, 0}, {1, 2, 0, 1}},
                                             {1/6., 1/6.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "("
        "[({}{})({}{})]"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0,
                                             {{0, 1, 0, 1}, {0, 2, 0, 1}},
                                             {1/6., 1/6.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  }
  {
    std::string expected_certificate =
      "("
        "({}{})"
        "({}{})"
      ")";
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 1,
                                             {{0, 1, 0, 1}, {0, 2, 0, 1}},
                                             {1/6., 1/6.});
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  }
}


void TestTreeRebalancing() {
  {  // Identity test -- no rebalancing is applied.
    std::string expected_certificate =
      "(["
        "({}{})"  // Play Heads: HH, HT
        "({}{})"  // Play Tails: TH, TT
      "])";
    for (int i = 0; i < 2; ++i) {
      std::unique_ptr<CFRTree> tree = MakeTree("matrix_mp", i);
      SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));

      tree->Rebalance();
      SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
    }
  }
  {  // Rebalance test: when 2nd player passes, we add dummy observation nodes.
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0,
                                             {{0, 1, 0}}, {1/6.});
    std::string expected_certificate =
        "(("
        "[({})({})]"  // 2nd player bets
        "{}"          // 2nd player passes
        "))";
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));

    tree->Rebalance();

    // The order is swapped only in the certificate computation, but not in
    // the actual tree.
    std::string expected_rebalanced_certificate =
        "(("
        "(({}))"      // 2nd player passes
        "[({})({})]"  // 2nd player bets
        "))";
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()),
                   expected_rebalanced_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  }
  {  // Rebalance test: simultaneous move games.
    std::unique_ptr<CFRTree> tree = MakeTree(iigs3, 0,
                                             /*start_histories=*/{
                                              {0  /* = 0 0 */},
                                              {1  /* = 1 0 */,
                                               3  /* = 2 2 */}
                                             },
                                             /*start_reaches=*/{1., 1.});

    std::string expected_certificate =
        "("
          "[({}{})({}{})]"
          "{}"
        ")";
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_FALSE(tree->IsBalanced());
    SPIEL_CHECK_FALSE(RecomputeBalance(*tree));

    tree->Rebalance();
    std::string expected_rebalanced_certificate =
      "("
        "(({}))"
        "[({}{})({}{})]"
      ")";
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()),
                   expected_rebalanced_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  }
  {  // Full Kuhn rebalancing test.
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0);
    tree->Rebalance();

    std::string expected_rebalanced_certificate =
      "(("
        "(["
          "(("
          // Notice all terminals are at the same depth (same indentation).
            "(({}))"
            "(({}))"
            "(({}))"
            "(({}))"
          "))"
          "(("
            "(({}))"
            "(({}))"
            "[({}"
              "{})"
             "({}"
              "{})]"
          "))"
        "])"
        // Just 2 more copies.
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
        "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
      "))";
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()),
                   expected_rebalanced_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
  }
}

void CheckTreeLeaves(const CFRTree& tree, int move_limit) {
  for (CFRNode const* leaf_node : tree.leaves_iterator()) {
    SPIEL_CHECK_TRUE(leaf_node->IsLeafNode());
    SPIEL_CHECK_TRUE(leaf_node->HasTensor());
    SPIEL_CHECK_FALSE(leaf_node->CorrespondingStates().empty());

    // Check MoveNumber() for all corresponding states.
    //
    // The conditions are following:
    // - either all states are terminal, and have the same MoveNumber() that
    //   is less or equal to move_limit,
    // - or all states are non-terminal and the MoveNumber() == move_limit.

    const int num_states = leaf_node->CorrespondingStates().size();
    int terminal_cnt = 0;
    int max_move_number = std::numeric_limits<int>::min();
    int min_move_number = std::numeric_limits<int>::max();
    for (const std::unique_ptr<State>
          & state : leaf_node->CorrespondingStates()) {
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

void BuildAllDepths(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int max_moves = game->MaxMoveNumber();

  for (int move_limit = 0; move_limit < max_moves; ++move_limit) {
    for (int pl = 0; pl < game->NumPlayers(); ++pl) {
      std::unique_ptr<CFRTree> tree = MakeTree(game_name, pl, move_limit);
      SPIEL_CHECK_EQ(tree->IsBalanced(), RecomputeBalance(*tree));
      CheckTreeLeaves(*tree, move_limit);
      tree->Rebalance();
      SPIEL_CHECK_TRUE(RecomputeBalance(*tree));
      CheckTreeLeaves(*tree, move_limit);
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
          "([])" // Node Q
          "([])" // Node K
        ")"
      ")";  // </dummy>
    std::unique_ptr<CFRTree> tree = MakeTree("kuhn_poker", 0, 2);
    SPIEL_CHECK_EQ(ComputeCertificate(tree->Root()), expected_certificate);
    SPIEL_CHECK_TRUE(tree->IsBalanced());
    SPIEL_CHECK_TRUE(RecomputeBalance(*tree));

    for (CFRNode const* acting : tree->leaves_iterator()) {
      SPIEL_CHECK_TRUE(acting->IsLeafNode());
      SPIEL_CHECK_EQ(acting->Type(), kDecisionInfostateNode);
      SPIEL_CHECK_EQ(acting->CorrespondingStates().size(), 2);
      SPIEL_CHECK_TRUE(acting->HasTensor());
    }
  }

  // Check that arbitrary depth-limited trees always have tensors,
  // and the corresponding states have correct MoveNumber().
  // This must hold even after rebalancing the trees.
  BuildAllDepths("kuhn_poker");
  BuildAllDepths("kuhn_poker(players=3)");
  BuildAllDepths("leduc_poker");
  BuildAllDepths("goofspiel(players=2,num_cards=3,imp_info=True)");
  BuildAllDepths("goofspiel(players=3,num_cards=3,imp_info=True)");
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestRootCertificates();
  open_spiel::algorithms::TestCertificatesFromStartHistories();
  open_spiel::algorithms::TestTreeRebalancing();
  open_spiel::algorithms::TestDepthLimitedTrees();
}
