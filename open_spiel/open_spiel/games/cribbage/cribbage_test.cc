// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/cribbage/cribbage.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

constexpr int kSeed = 2871611;

namespace open_spiel {
namespace cribbage {
namespace {

namespace testing = open_spiel::testing;

void CardToStringTest() {
  std::cout << "CardToStringTest" << std::endl;
  std::vector<std::string> card_strings;
  card_strings.reserve(52);
  std::string suit_names(kSuitNames);
  std::string ranks(kRanks);

  for (int i = 0; i < 52; ++i) {
    Card card = GetCard(i);
    std::string card_string = card.to_string();
    size_t rank_pos = ranks.find(card_string[0]);
    SPIEL_CHECK_TRUE(rank_pos != std::string::npos);
    size_t suit_pos = suit_names.find(card_string[1]);
    SPIEL_CHECK_TRUE(suit_pos != std::string::npos);
    auto iter =
        std::find(card_strings.begin(), card_strings.end(), card_string);
    SPIEL_CHECK_TRUE(iter == card_strings.end());
    card_strings.push_back(card_string);
  }
}

void BasicLoadTest() {
  std::cout << "BasicLoadTest" << std::endl;
  std::shared_ptr<const Game> game = LoadGame("cribbage");
  std::unique_ptr<State> state = game->NewInitialState();
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);

  game = LoadGame("cribbage(players=3)");
  state = game->NewInitialState();
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_EQ(game->NumPlayers(), 3);

  game = LoadGame("cribbage(players=4)");
  state = game->NewInitialState();
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_EQ(game->NumPlayers(), 4);
}

void BasicOneTurnPlaythrough() {
  std::cout << "BasicOneTurnPlaythroughTest" << std::endl;
  std::mt19937 rng(kSeed);
  std::shared_ptr<const Game> game = LoadGame("cribbage");
  std::unique_ptr<State> state = game->NewInitialState();
  CribbageState* crib_state = static_cast<CribbageState*>(state.get());

  // Deal.
  while (state->IsChanceNode()) {
    std::cout << state->ToString() << std::endl;
    double z = absl::Uniform(rng, 0.0, 1.0);
    Action outcome = SampleAction(state->ChanceOutcomes(), z).first;
    std::cout << "Sampled outcome: "
              << state->ActionToString(kChancePlayerId, outcome) << std::endl;
    state->ApplyAction(outcome);
  }

  // Card choices.
  for (int p = 0; p < game->NumPlayers(); ++p) {
    std::cout << state->ToString() << std::endl;
    std::vector<Action> legal_actions = state->LegalActions();
    int idx = absl::Uniform<int>(rng, 0, legal_actions.size());
    Action action = legal_actions[idx];
    std::cout << "Sampled action: "
              << state->ActionToString(state->CurrentPlayer(), action)
              << std::endl;
    state->ApplyAction(action);
  }

  // Starter.
  std::cout << state->ToString() << std::endl;
  double z = absl::Uniform(rng, 0.0, 1.0);
  Action outcome = SampleAction(state->ChanceOutcomes(), z).first;
  std::cout << "Sampled outcome: "
            << state->ActionToString(kChancePlayerId, outcome) << std::endl;
  state->ApplyAction(outcome);
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  // Play phase.
  while (crib_state->round() < 1) {
    std::cout << state->ToString() << std::endl;
    std::vector<Action> legal_actions = state->LegalActions();
    int idx = absl::Uniform<int>(rng, 0, legal_actions.size());
    Action action = legal_actions[idx];
    std::cout << "Sampled action: "
              << state->ActionToString(state->CurrentPlayer(), action)
              << std::endl;
    state->ApplyAction(action);
  }

  std::cout << state->ToString() << std::endl;
}

void AssertApproxEqual(const std::vector<double>& values1,
                       const std::vector<double>& values2) {
  for (int i = 0; i < values1.size(); ++i) {
    SPIEL_CHECK_TRUE(Near(values1[i], values2[i]));
  }
}

void WikipediaExampleTwoPlayers() {
  // https://en.wikipedia.org/wiki/Rules_of_cribbage
  std::shared_ptr<const Game> game = LoadGame("cribbage");
  std::unique_ptr<State> state = game->NewInitialState();
  CribbageState* crib_state = static_cast<CribbageState*>(state.get());

  // Deal.
  //   Player 0 (dealer) Alice: 5S 4S 2S 6H | 7H 8H
  //   Player 1          Bob:   6D JH 4H 7C | 2D 8D
  //   Starter: 3C
  const std::vector<std::string> cards = {"5S", "4S", "2S", "6H", "7H", "8H",
                                          "6D", "JH", "4H", "7C", "2D", "8D"};
  for (const std::string& cstr : cards) {
    Card card = GetCardByString(cstr);
    state->ApplyAction(card.id);
  }

  std::cout << state->ToString() << std::endl;

  // Card choices. Alice, then Bob.
  state->ApplyAction(ToAction(GetCardByString("7H"), GetCardByString("8H")));
  state->ApplyAction(ToAction(GetCardByString("2D"), GetCardByString("8D")));

  // Starter.
  state->ApplyAction(GetCardByString("3D").id);

  // Play phase.
  std::cout << state->ToString() << std::endl;

  // Bob plays JH
  state->ApplyAction(GetCardByString("JH").id);
  AssertApproxEqual(crib_state->scores(), {0.0, 0.0});
  // Alice plays 5S
  state->ApplyAction(GetCardByString("5S").id);
  AssertApproxEqual(crib_state->scores(), {2.0, 0.0});
  // Bob plays 7C
  state->ApplyAction(GetCardByString("7C").id);
  AssertApproxEqual(crib_state->scores(), {2.0, 0.0});
  // Alice plays 6H
  state->ApplyAction(GetCardByString("6H").id);
  AssertApproxEqual(crib_state->scores(), {5.0, 0.0});
  // Bob passes
  state->ApplyAction(kPassAction);
  // Alice plays 2S
  state->ApplyAction(GetCardByString("2S").id);
  AssertApproxEqual(crib_state->scores(), {5.0, 0.0});
  // Bob passes, Alice passes.
  state->ApplyAction(kPassAction);
  state->ApplyAction(kPassAction);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0});

  // New play round. Bob starts with 6D.
  state->ApplyAction(GetCardByString("6D").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0});
  // Alice plays 4S.
  state->ApplyAction(GetCardByString("4S").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0});
  // Bob plays 4H.
  state->ApplyAction(GetCardByString("4H").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 2.0});
  // Alice passes, Bob passes.
  state->ApplyAction(kPassAction);
  state->ApplyAction(kPassAction);
  // Points are now {6, 3}.
  //   Alice counts her hand: 15 two, 15 four (using the starter) and a run of
  //                          5 = 9.
  //   Bob counts his hand: nothing.
  //   Alice counts the crib: 15 two, 15 four, and a pair = 6.
  std::cout << state->ToString() << std::endl;

  AssertApproxEqual(crib_state->scores(), {21.0, 3.0});
}

void WikipediaExampleThreePlayers() {
  // https://en.wikipedia.org/wiki/Rules_of_cribbage
  std::shared_ptr<const Game> game = LoadGame("cribbage(players=3)");
  std::unique_ptr<State> state = game->NewInitialState();
  CribbageState* crib_state = static_cast<CribbageState*>(state.get());

  // Deal.
  //   Player 0 (dealer) Claire: 7S KD 9D 8H | 7H
  //   Player 1          David:  TS 5S 4S 7C | 2D
  //   Player 2          Eve:    7D 3D TH 5C | 3S
  //   Crib: TC
  //   Starter: 3C
  const std::vector<std::string> cards = {
      "7S", "KD", "9D", "8H", "7H", "TS", "5S", "4S",
      "7C", "2D", "7D", "3D", "TH", "5C", "3S", "TC",
  };
  for (const std::string& cstr : cards) {
    Card card = GetCardByString(cstr);
    state->ApplyAction(card.id);
  }

  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  // Card choices.
  state->ApplyAction(GetCardByString("7H").id);
  state->ApplyAction(GetCardByString("2D").id);
  state->ApplyAction(GetCardByString("3S").id);

  // Starter.
  std::cout << state->ToString() << std::endl;
  state->ApplyAction(GetCardByString("3C").id);

  // David plays 7C.
  state->ApplyAction(GetCardByString("7C").id);
  AssertApproxEqual(crib_state->scores(), {0.0, 0.0, 0.0});
  // Eve plays 7D.
  state->ApplyAction(GetCardByString("7D").id);
  AssertApproxEqual(crib_state->scores(), {0.0, 0.0, 2.0});
  // Claire plays 7S.
  state->ApplyAction(GetCardByString("7S").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 2.0});
  // David plays 5S.
  state->ApplyAction(GetCardByString("5S").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 2.0});
  // Eve plays 31.
  state->ApplyAction(GetCardByString("5C").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 6.0});

  // Claire plays 8H.
  state->ApplyAction(GetCardByString("8H").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 6.0});
  // David plays TS.
  state->ApplyAction(GetCardByString("TS").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 6.0});
  // Eve plays TH.
  state->ApplyAction(GetCardByString("TH").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 8.0});
  // Claire passes, David passess, Eve plays 3D
  state->ApplyAction(kPassAction);
  state->ApplyAction(kPassAction);
  state->ApplyAction(GetCardByString("3D").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 10.0});

  // Claire plays KD
  state->ApplyAction(GetCardByString("KD").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 10.0});
  // David plays 4S
  state->ApplyAction(GetCardByString("4S").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 10.0});
  // Eve passes
  state->ApplyAction(kPassAction);
  // Claire plays 9D
  state->ApplyAction(GetCardByString("9D").id);
  AssertApproxEqual(crib_state->scores(), {6.0, 0.0, 10.0});
  // David passes, Eve passes again. Claire passes.
  // Claire gets 1 point, then hands scored.
  state->ApplyAction(kPassAction);
  state->ApplyAction(kPassAction);
  state->ApplyAction(kPassAction);

  // Claire scores 15 two and 7-8-9 for four = 5.
  // Claire's crib scores 15 two, 15 four, 15 six, and pair = 8.
  // David scores 15 two, 15 four, and 3-4-5 for three is 7.
  // Eve scores 15 two, 15 four, 15 six, and a pair for 8.
  AssertApproxEqual(crib_state->scores(), {20.0, 7.0, 18.0});
  std::cout << state->ToString() << std::endl;
}

void HandScoringTests() {
  // Suit order: CDHS
  std::vector<Card> hand;
  hand = GetHandFromStrings({"QC", "TD", "7H", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 4);
  hand = GetHandFromStrings({"QC", "QD", "7H", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 6);
  hand = GetHandFromStrings({"QC", "QD", "QH", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 12);
  hand = GetHandFromStrings({"QC", "QD", "QH", "5S", "QS"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 20);
  hand = GetHandFromStrings({"5C", "QC", "5D", "5H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 28);  // 8 for 15s w/ Q, 12 4-of-K, 8 more 15s
  hand = GetHandFromStrings({"QC", "JD", "7H", "9H"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("5D")), 5);  // 4 15s + jack
  hand = GetHandFromStrings({"QC", "JD", "7H", "9H"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("5S")), 4);  // 4 15s
  // Flushes. 5-card flush, then a 4-card flush, then no flush.
  hand = GetHandFromStrings({"QC", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2C")), 5);
  hand = GetHandFromStrings({"QC", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2D")), 4);
  hand = GetHandFromStrings({"QD", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2C")), 0);
  // 5-card flush and run of 5 + nobs = 11.
  hand = GetHandFromStrings({"9C", "TC", "JC", "QC"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("KC")), 11);
  // Examples of runs from the rule book.
  hand = GetHandFromStrings({"5C", "6C", "7C", "8D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("8S")), 14);
  // 3 runs of 3 (9) + 3-of-a-kind (6) + three 15s (8) = 21.
  hand = GetHandFromStrings({"4C", "4D", "4S", "5D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("6S")), 21);
  // 4 runs of 3 (12) + 2 pairs (4) + 2 15s (4) = 20.
  hand = GetHandFromStrings({"6C", "6D", "7S", "7D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("8S")), 20);
}

void BasicCribbageTests() {
  testing::RandomSimTest(*LoadGame("cribbage"), 10);
  testing::RandomSimTest(*LoadGame("cribbage(players=3)"), 10);
  testing::RandomSimTest(*LoadGame("cribbage(players=4)"), 10);
}

}  // namespace
}  // namespace cribbage
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::cribbage::CardToStringTest();
  open_spiel::cribbage::BasicLoadTest();
  open_spiel::cribbage::BasicCribbageTests();
  open_spiel::cribbage::BasicOneTurnPlaythrough();
  open_spiel::cribbage::HandScoringTests();
  open_spiel::cribbage::WikipediaExampleTwoPlayers();
  open_spiel::cribbage::WikipediaExampleThreePlayers();
  open_spiel::cribbage::BasicCribbageTests();
}
