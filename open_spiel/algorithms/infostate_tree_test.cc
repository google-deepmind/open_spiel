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


// To make sure we can easily test infostate tree structures, we compute their
// certificate (string representation) that we can easily compare.
std::string ComputeCertificate(CFRNode& node) {
  if (node.Type() == kDecisionNode) {
    SPIEL_CHECK_GT(node->legal_actions.size(), 0);
    SPIEL_CHECK_GT(node->cumulative_regrets.size(), 0);
    SPIEL_CHECK_GT(node->cumulative_policy.size(), 0);
    SPIEL_CHECK_GT(node->current_policy.size(), 0);
  } else {
    SPIEL_CHECK_EQ(node->legal_actions.size(), 0);
    SPIEL_CHECK_EQ(node->cumulative_regrets.size(), 0);
    SPIEL_CHECK_EQ(node->cumulative_policy.size(), 0);
    SPIEL_CHECK_EQ(node->current_policy.size(), 0);
  }

  if (node.Type() == kTerminalNode) {
    SPIEL_CHECK_EQ(node.NumChildren(), 0);
    return "{}";
  }

  std::vector<std::string> certificates;
  for (CFRNode& child : node) {
    certificates.push_back(ComputeCertificate(child));
  }
  std::sort(certificates.begin(), certificates.end());

  std::string open, close;
  if (node.Type() == kDecisionNode) {
    open = "[";
    close = "]";
  } else if (node.Type() == kObservationNode) {
    open = "(";
    close = ")";
  }

  return absl::StrCat(
      open,
      absl::StrJoin(certificates.begin(), certificates.end(), ""),
      close);
}

std::string CertificateFromStartHistories(
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

  CFRTree tree(absl::MakeSpan(start_state_ptrs), absl::MakeSpan(start_reaches),
               infostate_observer, player_id);
  return ComputeCertificate(*tree.Root());
}

std::string RootCertificate(const std::string& game_name, Player player_id) {
  CFRTree tree(*LoadGame(game_name), player_id);
  return ComputeCertificate(*tree.Root());
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
    SPIEL_CHECK_EQ(RootCertificate("kuhn_poker", 0), expected_certificate);
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
    SPIEL_CHECK_EQ(RootCertificate("kuhn_poker", 1), expected_certificate);
  }
  {
    std::string expected_certificate =
      "(["
        "({})"  // draw 1,1
        "({})"  // lose 1,2
        "({})"  // win  2,1
        "({})"  // draw 2,2
      "])";
    std::string iigs2 = "goofspiel("
                          "num_cards=2,"
                          "imp_info=True,"
                          "points_order=ascending"
                        ")";
    SPIEL_CHECK_EQ(RootCertificate(iigs2, 0), expected_certificate);
    SPIEL_CHECK_EQ(RootCertificate(iigs2, 1), expected_certificate);
  }
  {
    std::string expected_certificate =
      "(["
        "({})"  // HH
        "({})"  // HT
        "({})"  // TH
        "({})"  // TT
      "])";
    SPIEL_CHECK_EQ(RootCertificate("matrix_mp", 0), expected_certificate);
    SPIEL_CHECK_EQ(RootCertificate("matrix_mp", 1), expected_certificate);
  }
}

void TestCertificatesFromStartHistories() {
  {
    std::string expected_certificate =
      "("
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
      ")";
    SPIEL_CHECK_EQ(
        CertificateFromStartHistories("kuhn_poker", 0,
            {{0}, {2}}, {1/3., 1/3.}),
        expected_certificate);
  }
  {
    std::string expected_certificate =
      "("
        "([(({}{}))({})][({})({})])"
        "([(({}{}))({})][({})({})])"
      ")";
    SPIEL_CHECK_EQ(
        CertificateFromStartHistories("kuhn_poker", 1,
            {{1, 0}, {1, 2}}, {1/6., 1/6.}),
        expected_certificate);
  }
  {
    std::string expected_certificate =
      "("
        "([(([({}{})({}{})]{}{}))(({}{}{}{}))])"
        "[({})({})]"
      ")";
    SPIEL_CHECK_EQ(
        CertificateFromStartHistories("kuhn_poker", 0,
            {{0}, {2, 1, 0, 1}}, {1/3., 1/6.}),
        expected_certificate);
  }
  {
    std::string expected_certificate =
      "("
        "([(({}{}))({})][({})({})])"
        "({}{})"
      ")";
    SPIEL_CHECK_EQ(
        CertificateFromStartHistories("kuhn_poker", 1,
            {{1, 0}, {1, 2, 0, 1}}, {1/6., 1/6.}),
        expected_certificate);
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestRootCertificates();
  open_spiel::algorithms::TestCertificatesFromStartHistories();
}
