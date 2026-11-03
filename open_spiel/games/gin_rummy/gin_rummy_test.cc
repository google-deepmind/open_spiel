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

#include "open_spiel/games/gin_rummy/gin_rummy.h"
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace gin_rummy {
namespace {

namespace testing = open_spiel::testing;

void BasicGameTests() {
  testing::LoadGameTest("gin_rummy");
  testing::RandomSimTest(*LoadGame("gin_rummy"), 10);
}

void MeldTests() {
  GinRummyUtils utils = GinRummyUtils(kDefaultNumRanks, kDefaultNumSuits,
                                      kDefaultHandSize);
  // There are 185 melds of length between 3 and 5 cards. All melds of
  // length greater than 5 can be expressed as two or more smaller melds.
  std::vector<int> full_deck;
  for (int i = 0; i < utils.num_cards; ++i) full_deck.push_back(i);
  std::vector<std::vector<int>> all_melds = utils.AllMelds(full_deck);
  SPIEL_CHECK_EQ(all_melds.size(), kNumMeldActions);

  // Some simple meld tests
  std::vector<std::string> cards;
  cards = {"As", "2s", "3s"};
  SPIEL_CHECK_TRUE(utils.IsSuitMeld(utils.CardStringsToCardInts(cards)));
  SPIEL_CHECK_FALSE(utils.IsRankMeld(utils.CardStringsToCardInts(cards)));
  cards = {"As", "Ac", "Ad"};
  SPIEL_CHECK_TRUE(utils.IsRankMeld(utils.CardStringsToCardInts(cards)));
  SPIEL_CHECK_FALSE(utils.IsSuitMeld(utils.CardStringsToCardInts(cards)));
  cards = {"As", "Ac", "Ad", "2s"};
  SPIEL_CHECK_FALSE(utils.IsRankMeld(utils.CardStringsToCardInts(cards)));
  SPIEL_CHECK_FALSE(utils.IsSuitMeld(utils.CardStringsToCardInts(cards)));

  // No "around the corner" melds
  cards = {"As", "2s", "3s", "Ks"};
  SPIEL_CHECK_FALSE(utils.IsRankMeld(utils.CardStringsToCardInts(cards)));
  SPIEL_CHECK_FALSE(utils.IsSuitMeld(utils.CardStringsToCardInts(cards)));

  // These cards are represented internally as consecutive ints
  // but are not a meld.
  cards = {"Js", "Qs", "Ks", "Ac"};
  SPIEL_CHECK_FALSE(utils.IsRankMeld(utils.CardStringsToCardInts(cards)));
  SPIEL_CHECK_FALSE(utils.IsSuitMeld(utils.CardStringsToCardInts(cards)));

  // Check that the meld_to_int and int_to_meld maps work correctly.
  int meld_id;
  cards = {"Ks", "Kc", "Kd", "Kh"};
  meld_id = utils.meld_to_int.at(utils.CardStringsToCardInts(cards));
  SPIEL_CHECK_EQ(meld_id, 64);
  SPIEL_CHECK_EQ(utils.meld_to_int.at(utils.int_to_meld.at(64)), 64);
  cards = {"As", "2s", "3s"};
  meld_id = utils.meld_to_int.at(utils.CardStringsToCardInts(cards));
  SPIEL_CHECK_EQ(meld_id, 65);
  SPIEL_CHECK_EQ(utils.meld_to_int.at(utils.int_to_meld.at(65)), 65);
  cards = {"As", "2s", "3s", "4s"};
  meld_id = utils.meld_to_int.at(utils.CardStringsToCardInts(cards));
  SPIEL_CHECK_EQ(meld_id, 109);
  SPIEL_CHECK_EQ(utils.meld_to_int.at(utils.int_to_meld.at(109)), 109);
  cards = {"As", "2s", "3s", "4s", "5s"};
  meld_id = utils.meld_to_int.at(utils.CardStringsToCardInts(cards));
  SPIEL_CHECK_EQ(meld_id, 149);
  SPIEL_CHECK_EQ(utils.meld_to_int.at(utils.int_to_meld.at(149)), 149);
  cards = {"9h", "Th", "Jh", "Qh", "Kh"};
  meld_id = utils.meld_to_int.at(utils.CardStringsToCardInts(cards));
  SPIEL_CHECK_EQ(meld_id, 184);
  SPIEL_CHECK_EQ(utils.meld_to_int.at(utils.int_to_meld.at(184)), 184);

  // Should find five rank melds and one suit meld.
  // +--------------------------+
  // |As2s3s                    |
  // |Ac                        |
  // |Ad                        |
  // |Ah                        |
  // +--------------------------+
  cards = {"As", "Ac", "Ad", "Ah", "2s", "3s"};
  std::vector<int> card_ints = utils.CardStringsToCardInts(cards);
  all_melds = utils.AllMelds(card_ints);
  SPIEL_CHECK_EQ(all_melds.size(), 6);

  // More complicated example with 14 possible melds.
  // +--------------------------+
  // |      4s5s6s              |
  // |      4c5c6c              |
  // |      4d5d6d              |
  // |      4h5h                |
  // +--------------------------+
  cards = {"4s", "4c", "4d", "4h", "5s", "5c", "5d", "5h", "6s", "6c", "6d"};
  card_ints = utils.CardStringsToCardInts(cards);
  all_melds = utils.AllMelds(card_ints);
  SPIEL_CHECK_EQ(all_melds.size(), 14);

  // +--------------------------+
  // |    3s4s5s6s              |
  // |  2c3c4c5c                |
  // |      4d5d                |
  // |      4h                  |
  // +--------------------------+
  // Should find the best meld group 4s4d4h, 5s5c5d, 2c3c4c with 3 deadwood.
  cards = {"4s", "4c", "4d", "4h", "5s", "5c", "5d", "6s", "2c", "3s", "3c"};
  card_ints = utils.CardStringsToCardInts(cards);
  std::vector<std::vector<int>> meld_group = utils.BestMeldGroup(card_ints);
  std::cout << meld_group << std::endl;
  for (auto meld : meld_group)
    std::cout << utils.CardIntsToCardStrings(meld) << std::endl;
  int deadwood = utils.MinDeadwood(card_ints);
  SPIEL_CHECK_EQ(deadwood, 3);
}

// Cards laid off on a run extend it, so further cards may be laid off on the
// same end, whatever the order of the previous layoffs.
void ChainedLayoffTest() {
  GinRummyUtils utils(kDefaultNumRanks, kDefaultNumSuits, kDefaultHandSize);
  auto layoffs = [&utils](const std::vector<std::vector<std::string>>& melds,
                          const std::vector<std::string>& previous) {
    std::vector<int> meld_ids;
    for (const auto& meld : melds) {
      meld_ids.push_back(utils.MeldToInt(utils.CardStringsToCardInts(meld)));
    }
    return utils.CardIntsToCardStrings(
        utils.AllLayoffs(meld_ids, utils.CardStringsToCardInts(previous)));
  };
  using Cards = std::vector<std::string>;
  SPIEL_CHECK_EQ(layoffs({{"9c", "Tc", "Jc"}}, {}), (Cards{"8c", "Qc"}));
  SPIEL_CHECK_EQ(layoffs({{"9c", "Tc", "Jc"}}, {"8c"}), (Cards{"7c", "Qc"}));
  SPIEL_CHECK_EQ(layoffs({{"9c", "Tc", "Jc"}}, {"8c", "7c"}),
                 (Cards{"6c", "Qc"}));
  SPIEL_CHECK_EQ(layoffs({{"9c", "Tc", "Jc"}}, {"7c", "8c"}),
                 (Cards{"6c", "Qc"}));
  SPIEL_CHECK_EQ(layoffs({{"9c", "Tc", "Jc"}}, {"8c", "Qc", "7c", "Kc"}),
                 (Cards{"6c"}));
  SPIEL_CHECK_EQ(layoffs({{"As", "2s", "3s"}}, {"4s", "5s"}), (Cards{"6s"}));
  SPIEL_CHECK_EQ(layoffs({{"Jh", "Qh", "Kh"}}, {"Th", "9h"}), (Cards{"8h"}));
  // A card laid off on a set does not extend a run it does not touch.
  SPIEL_CHECK_EQ(layoffs({{"2h", "3h", "4h"}, {"7s", "7c", "7d"}}, {"7h"}),
                 (Cards{"Ah", "5h", "7h"}));
}

// The deadwood of a hand that must discard is minimized over the discard. The
// best meld group of these eleven cards (2s-2d-2h, 4s-4c-4d and the run Ac-3c)
// leaves Ad and Ah, one of which is discarded for 1 deadwood; discarding 3c
// instead leaves gin.
void MandatoryDiscardDeadwoodTest() {
  GinRummyUtils utils(kDefaultNumRanks, kDefaultNumSuits, kDefaultHandSize);
  VecInt hand = utils.CardStringsToCardInts(
      {"2s", "4s", "Ac", "2c", "3c", "4c", "Ad", "2d", "4d", "Ah", "2h"});
  SPIEL_CHECK_EQ(utils.MinDeadwood(hand), 0);
  SPIEL_CHECK_EQ(utils.LegalDiscards(hand, 0),
                 utils.CardStringsToCardInts({"3c"}));
  absl::c_reverse(hand);
  SPIEL_CHECK_EQ(utils.MinDeadwood(hand), 0);
  // The discard is taken whenever the hand holds hand_size + 1 cards.
  GinRummyUtils small(kDefaultNumRanks, kDefaultNumSuits, /*hand_size=*/3);
  SPIEL_CHECK_EQ(small.MinDeadwood(utils.CardStringsToCardInts(
                     {"As", "Ac", "Ad", "Ks"})),
                 0);
  SPIEL_CHECK_EQ(small.MinDeadwood(utils.CardStringsToCardInts(
                     {"As", "5s", "9s"})),
                 15);
}

void CheckLegalActionsAfter(const std::string& game_string,
                            const std::vector<Action>& history,
                            const std::vector<Action>& expected) {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action : history) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), action));
    state->ApplyAction(action);
  }
  SPIEL_CHECK_EQ(state->LegalActions(), expected);
}

// A knock is legal when the hand left after the best discard is within the
// knock card, even if the best eleven-card meld group is not.
void KnockAfterBestDiscardGameplayTest() {
  // Knock card 3. Player 0 holds 2s, 2c-5c, 3d-5d, 2h, 3h and 5h. Discarding
  // 5h leaves 2s-2c-2h, 3c-5c, 3d-5d and 3 deadwood, so Knock (55) is legal.
  CheckLegalActionsAfter(
      "gin_rummy(oklahoma=true)",
      {28, 29, 17, 14, 24, 1,  25, 40, 43, 15, 44, 30, 48, 11, 16,
       23, 39, 31, 34, 6,  41, 52, 25, 53, 32, 11, 53, 46, 24, 53,
       9,  23, 53, 35, 35, 52, 16, 52, 46, 53, 18, 30, 52},
      {1, 14, 15, 16, 17, 28, 29, 30, 40, 41, 43, 55});
  // Knock card 0 (an ace upcard). Player 1 holds 2s-4s, Ac-3c, Ad, 2d and
  // Ah-3h. Discarding 4s leaves three aces, four twos and three threes: gin.
  CheckLegalActionsAfter(
      "gin_rummy(oklahoma=true)",
      {43, 23, 24, 28, 8,  44, 19, 14, 18, 50, 39, 15, 3,  5,  46, 26,
       27, 45, 36, 51, 13, 54, 52, 51, 53, 41, 14, 52, 36, 53, 11, 23,
       53, 1,  46, 53, 33, 33, 53, 2,  45, 52, 8,  53, 37, 37, 52, 41,
       52, 5,  53, 47, 47, 53, 9,  9,  53, 42, 37, 53, 40},
      {1, 2, 3, 13, 14, 15, 26, 27, 39, 40, 41, 55});
}

// The knocker lays the run 4s-6s; the defender lays off 7s, then 8s, and may
// then lay off 9s (action 8).
void ChainedLayoffGameplayTest() {
  CheckLegalActionsAfter(
      "gin_rummy(oklahoma=true)",
      {34, 51, 17, 31, 6,  7,  42, 23, 36, 50, 15, 38, 4,   27,  48, 3,
       19, 39, 5,  49, 37, 54, 54, 53, 12, 51, 53, 1,  38,  53,  8,  12,
       53, 30, 30, 53, 47, 50, 52, 55, 19, 163, 124, 54, 6, 7},
      {8, 47, 54});
}

// An extremely rare situation, but one that does arise in actual gameplay.
// Tests both layoff and undercut functionality.
void GameplayTest1() {
  GameParameters params;
  // Modify undercut bonus game parameter as an additional test.
  params["undercut_bonus"] = GameParameter(20);
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  initial_actions = {11, 4,  5, 6,  21, 22, 23, 12, 25, 38, 1,  14,
                     27, 40, 7, 20, 33, 8,  19, 13, 36, 52, 55, 11};
  for (auto action : initial_actions) state->ApplyAction(action);
  std::cout << state->ToString() << std::endl;
  // Player turn: 0
  // Phase: Knock
  //
  // Player1: Deadwood=49
  // +--------------------------+
  // |  2s          8s9s        |
  // |Ac2c        7c8c          |
  // |  2d          8d          |
  // |  2h                      |
  // +--------------------------+
  //
  // Stock size: 31  Upcard: XX
  // Discard pile: Qs
  //
  // Player0: Deadwood=87
  // +--------------------------+
  // |        5s6s7s          Ks|
  // |                9cTcJc  Kc|
  // |                    Jd  Kd|
  // |                          |
  // +--------------------------+
  //
  // Player0 has knocked, and after laying melds will have the Jd left for
  // 10 points. Player 1 has two melds (2's and 8's) with 17 points remaining.
  // Laying the hand this way gives Player0 a win of 7 points. But there's a
  // better play! Player1 is not compelled to lay his 8's as a meld. Instead,
  // Player1 can lay off the 8s9s on the 5s6s7s, and the 7c8c on the 9cTcJc.
  // This leaves Player1 with only the 8d and Ac as deadwood, for a total of
  // 9 points, less than the 10 points Player0 knocked with. By breaking the
  // meld of 8's Player1 wins an undercut!

  // Player0 lays melds.
  state->ApplyAction(119);  // KsKcKd
  state->ApplyAction(125);  // 5s6s7s
  state->ApplyAction(140);  // 9cTcJc
  state->ApplyAction(54);
  // Player1 layoffs.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 7));
  state->ApplyAction(7);  // Lay off 8s
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 8));
  state->ApplyAction(8);  // Lay off 9s
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 20));
  state->ApplyAction(20);  // Lay off 8c
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 19));
  state->ApplyAction(19);  // Lay off 7c
  state->ApplyAction(54);  // Finished layoffs
  state->ApplyAction(65);  // Lay meld of 2's
  state->ApplyAction(54);
  // Player1 wins the difference in deadwood (10 - 9 = 1) and the undercut
  // bonus which was set to 20, for a total of 21 points.
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], -21);
  SPIEL_CHECK_EQ(returns[1], 21);
}

void GameplayTest2() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  initial_actions = {1,  4,  5,  6,  17, 18, 19, 30, 31, 32, 2,  3,
                     16, 29, 43, 44, 45, 7,  20, 33, 0,  52, 55, 1};
  for (auto action : initial_actions) state->ApplyAction(action);
  std::cout << state->ToString() << std::endl;
  // Player turn: 0
  // Phase: Knock
  //
  // Player1: Deadwood=57
  // +--------------------------+
  // |    3s4s      8s          |
  // |      4c      8c          |
  // |      4d      8d          |
  // |        5h6h7h            |
  // +--------------------------+
  //
  // Stock size: 31  Upcard: XX
  // Discard pile:
  //
  // Player0: Deadwood=57
  // +--------------------------+
  // |As      5s6s7s            |
  // |        5c6c7c            |
  // |        5d6d7d            |
  // |                          |
  // +--------------------------+
  //
  // Player0 has knocked. There are 6 different melds in Player0's hand.
  // Because the melds overlap, the first meld layed dictates the remaining
  // melds than can be layed.
  // In situations where there is a choice between laying rank melds or suit
  // melds, it is often advantageous to lay the hand as rank melds, which offer
  // fewer layoffs. Indeed, here Player0 must lay the hand as three rank melds
  // to avoid the undercut.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 6);
  state->ApplyAction(79);  // Lay the 5s5c5d
  legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2);
  state->ApplyAction(84);  // Lay the 6s6c6d
  legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  state->ApplyAction(89);  // Lay the 7s7c7d
  state->ApplyAction(54);
  // Player1 can lay off the 5h6h7h, but there's no need, as it's already
  // a meld.
  legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 4);  // 3 layoffs & kPassAction
  state->ApplyAction(54);
  state->ApplyAction(74);
  state->ApplyAction(94);
  state->ApplyAction(158);
  state->ApplyAction(54);
  // Player0 has 1 deadwood and Player1 has 3, so Player0 scores 2 points.
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 2);
  SPIEL_CHECK_EQ(returns[1], -2);
}

// Potentially tricky corner case.
void GameplayTest3() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  initial_actions = {10, 11, 12, 22, 35, 48, 13, 26, 1, 40, 9,  8,
                     3,  16, 29, 42, 4,  17, 30, 43, 0, 52, 55, 1};
  for (auto action : initial_actions) state->ApplyAction(action);
  std::cout << state->ToString() << std::endl;
  // Player turn: 0
  // Phase: Knock
  //
  // Player1: Deadwood=55
  // +--------------------------+
  // |      4s5s      9sTs      |
  // |      4c5c                |
  // |      4d5d                |
  // |      4h5h                |
  // +--------------------------+
  //
  // Stock size: 31  Upcard: XX
  // Discard pile: 2s
  //
  // Player0: Deadwood=65
  // +--------------------------+
  // |As                  JsQsKs|
  // |Ac                Tc      |
  // |Ad                Td      |
  // |  2h              Th      |
  // +--------------------------+
  //
  // Player0 has knocked. Player1 will have the opportunity to lay off the Ts.
  // We want to make sure that after laying off the Ts, Player1 will then
  // be able to lay off the 9s as well. If the Ts only gets counted as
  // a layoff on the rank meld of three tens, then the 9s would not lay off.

  // Player0 lays melds
  state->ApplyAction(59);
  state->ApplyAction(101);
  state->ApplyAction(131);
  state->ApplyAction(54);
  // Player1 lays off the Ts
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 9));
  state->ApplyAction(9);
  // Assert Player1 can lay off the 9s
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, 8));
  state->ApplyAction(8);
  // Player1 completes the hand and wins an undercut
  state->ApplyAction(54);
  state->ApplyAction(75);
  state->ApplyAction(80);
  state->ApplyAction(54);
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], -27);
  SPIEL_CHECK_EQ(returns[1], 27);
}

// Deals the knocker's and defender's hands and the upcard, then has the
// knocker draw the upcard, knock and discard. The knocker may sit in either
// seat; if the knocker is player 1, player 0 passes on the first upcard.
std::unique_ptr<State> DealAndKnock(const Game& game,
                                    const GinRummyUtils& utils, Player knocker,
                                    const std::vector<int>& knocker_cards,
                                    const std::vector<int>& defender_cards,
                                    const std::string& upcard,
                                    const std::string& discard) {
  std::unique_ptr<State> state = game.NewInitialState();
  const std::vector<int>& seat0 = knocker == 0 ? knocker_cards : defender_cards;
  const std::vector<int>& seat1 = knocker == 0 ? defender_cards : knocker_cards;
  for (int card : seat0) state->ApplyAction(card);
  for (int card : seat1) state->ApplyAction(card);
  state->ApplyAction(utils.CardInt(upcard));
  if (knocker == 1) state->ApplyAction(kPassAction);
  state->ApplyAction(kDrawUpcardAction);
  state->ApplyAction(kKnockAction);
  state->ApplyAction(utils.CardInt(discard));
  return state;
}

// When the knocker's deadwood equals the defender's, the defender wins an
// undercut: the deadwood difference (zero) plus the undercut bonus.
void EqualDeadwoodUndercutTest() {
  GinRummyUtils utils = GinRummyUtils(kDefaultNumRanks, kDefaultNumSuits,
                                      kDefaultHandSize);
  auto meld_action = [&utils](const std::vector<std::string>& cards) {
    return kMeldActionBase +
           utils.MeldToInt(utils.CardStringsToCardInts(cards));
  };
  // After drawing the As and discarding the 2s, the knocker holds three rank
  // melds and the As. The defender holds three runs and the Ah, and has
  // nothing to lay off on the knocker's rank melds. Both end on 1 deadwood.
  std::vector<int> knocker_cards = utils.CardStringsToCardInts(
      {"2s", "5s", "6s", "7s", "5c", "6c", "7c", "5d", "6d", "7d"});
  std::vector<int> defender_cards = utils.CardStringsToCardInts(
      {"8s", "9s", "Ts", "8c", "9c", "Tc", "8d", "9d", "Td", "Ah"});
  for (Player knocker : {0, 1}) {
    Player defender = 1 - knocker;
    for (int undercut_bonus : {kDefaultUndercutBonus, 0}) {
      std::shared_ptr<const Game> game = LoadGame(
          "gin_rummy", {{"undercut_bonus", GameParameter(undercut_bonus)}});
      std::unique_ptr<State> state = DealAndKnock(
          *game, utils, knocker, knocker_cards, defender_cards, "As", "2s");
      state->ApplyAction(meld_action({"5s", "5c", "5d"}));
      state->ApplyAction(meld_action({"6s", "6c", "6d"}));
      state->ApplyAction(meld_action({"7s", "7c", "7d"}));
      state->ApplyAction(kPassAction);
      state->ApplyAction(kPassAction);  // No layoffs.
      state->ApplyAction(meld_action({"8s", "9s", "Ts"}));
      state->ApplyAction(meld_action({"8c", "9c", "Tc"}));
      state->ApplyAction(meld_action({"8d", "9d", "Td"}));
      state->ApplyAction(kPassAction);
      SPIEL_CHECK_TRUE(state->IsTerminal());
      const auto& gin_state = down_cast<const GinRummyState&>(*state);
      SPIEL_CHECK_TRUE(gin_state.Knocked()[knocker]);
      SPIEL_CHECK_EQ(gin_state.Deadwood()[knocker], 1);
      SPIEL_CHECK_EQ(gin_state.Deadwood()[defender], 1);
      std::vector<double> returns = state->Returns();
      SPIEL_CHECK_EQ(returns[defender], undercut_bonus);
      SPIEL_CHECK_EQ(returns[knocker], -undercut_bonus);
    }
  }
}

// A gin hand cannot be undercut, even when the defender also has no deadwood
// and the gin bonus is 0.
void GinCannotBeUndercutTest() {
  GinRummyUtils utils = GinRummyUtils(kDefaultNumRanks, kDefaultNumSuits,
                                      kDefaultHandSize);
  auto meld_action = [&utils](const std::vector<std::string>& cards) {
    return kMeldActionBase +
           utils.MeldToInt(utils.CardStringsToCardInts(cards));
  };
  // After drawing the 8s and discarding the 2s, the knocker has gin. The
  // defender's hand is also fully melded.
  std::vector<int> knocker_cards = utils.CardStringsToCardInts(
      {"2s", "5s", "6s", "7s", "5c", "6c", "7c", "5d", "6d", "7d"});
  std::vector<int> defender_cards = utils.CardStringsToCardInts(
      {"Ts", "Js", "Qs", "Ks", "Tc", "Jc", "Qc", "Td", "Jd", "Qd"});
  for (Player knocker : {0, 1}) {
    Player defender = 1 - knocker;
    for (int gin_bonus : {kDefaultGinBonus, 0}) {
      std::shared_ptr<const Game> game =
          LoadGame("gin_rummy", {{"gin_bonus", GameParameter(gin_bonus)}});
      std::unique_ptr<State> state = DealAndKnock(
          *game, utils, knocker, knocker_cards, defender_cards, "8s", "2s");
      state->ApplyAction(meld_action({"5s", "6s", "7s", "8s"}));
      state->ApplyAction(meld_action({"5c", "6c", "7c"}));
      state->ApplyAction(meld_action({"5d", "6d", "7d"}));
      state->ApplyAction(kPassAction);
      // No layoffs against gin; the defender lays melds directly.
      state->ApplyAction(meld_action({"Ts", "Js", "Qs", "Ks"}));
      state->ApplyAction(meld_action({"Tc", "Jc", "Qc"}));
      state->ApplyAction(meld_action({"Td", "Jd", "Qd"}));
      state->ApplyAction(kPassAction);
      SPIEL_CHECK_TRUE(state->IsTerminal());
      const auto& gin_state = down_cast<const GinRummyState&>(*state);
      SPIEL_CHECK_EQ(gin_state.Deadwood()[knocker], 0);
      SPIEL_CHECK_EQ(gin_state.Deadwood()[defender], 0);
      std::vector<double> returns = state->Returns();
      SPIEL_CHECK_EQ(returns[knocker], gin_bonus);
      SPIEL_CHECK_EQ(returns[defender], -gin_bonus);
    }
  }
}

// Tests action on the 50th card, and tests that layoffs are not allowed when
// the knocking player has gin.
void WallTest() {
  GinRummyUtils utils = GinRummyUtils(kDefaultNumRanks, kDefaultNumSuits,
                                      kDefaultHandSize);
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> legal_actions;
  std::vector<Action> initial_actions;
  initial_actions = {8,  9,  10, 11, 12, 13, 14, 15, 48, 49, 0,  1,  2,  3,
                     4,  5,  6,  7,  50, 51, 16, 54, 54, 53, 17, 17, 53, 18,
                     18, 53, 19, 19, 53, 20, 20, 53, 21, 21, 53, 22, 22, 53,
                     23, 23, 53, 24, 24, 53, 25, 25, 53, 26, 26, 53, 27, 27,
                     53, 28, 28, 53, 29, 29, 53, 30, 30, 53, 31, 31, 53, 32,
                     32, 53, 33, 33, 53, 34, 34, 53, 35, 35, 53, 36, 36, 53,
                     37, 37, 53, 38, 38, 53, 39, 39, 53, 40, 40, 53, 41, 41,
                     53, 42, 42, 53, 43, 43, 53, 44, 44, 53, 46, 49};
  for (auto action : initial_actions) state->ApplyAction(action);
  std::cout << state->ToString() << std::endl;
  // Player turn: 1
  // Phase: Wall
  //
  // Player1: Deadwood=20
  // +--------------------------+
  // |As2s3s4s5s6s7s8s          |
  // |                          |
  // |                          |
  // |                      QhKh|
  // +--------------------------+
  //
  // Stock size: 2  Upcard: Jh
  // Discard pile: 4c5c6c7c8c9cTcJcQcKcAd2d3d4d5d6d7d8d9dTdJdQdKdAh2h3h4h5h6h
  //
  // Player0: Deadwood=18
  // +--------------------------+
  // |                9sTsJsQsKs|
  // |Ac2c3c                    |
  // |                          |
  // |              8h  Th      |
  // +--------------------------+
  //
  // We've reached the wall (i.e. only two cards are left in the stock).
  // Player1 is not allowed to draw from the stock, and instead must either pass
  // (which ends the game) or knock (if legal). In this case, Player1 can gin.
  // First let's make sure the game ends if Player1 passes.
  state->ApplyAction(54);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Now let's reset the state as depicted above and knock instead.
  state = game->NewInitialState();
  for (auto action : initial_actions) state->ApplyAction(action);
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, kKnockAction));
  // Player1 knocks and lays melds.
  state->ApplyAction(55);
  state->ApplyAction(0);
  state->ApplyAction(126);
  state->ApplyAction(164);
  state->ApplyAction(166);
  state->ApplyAction(54);
  // Player1 made gin, so Player0 cannot layoff the Th on JhQhKh
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_linear_search(legal_actions, utils.CardInt("Th")));
  // Player0 lays melds.
  state->ApplyAction(213);
  state->ApplyAction(132);
  state->ApplyAction(54);
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(legal_actions.size(), 0);
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[1], 43);  // 25 point gin bonus + 18 deadwood
  SPIEL_CHECK_EQ(returns[0], -43);
}

// The rules of gin rummy do not explicitly prevent infinite action sequences.
// Both players can keep drawing the upcard and discarding indefinitely. This
// is poor strategy and never occurs in actual play, but we need a way of
// ensuring the game is finite. In doing so, we don't want to prematurely
// declare a draw and prevent legitimate lines of play. Our solution is to cap
// the number of times the upcard can be drawn at 50. This is well above
// anything observed in actual play, and corresponds nicely to the 50 cards in
// play each hand.
void MaxGameLengthTest() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  // Deal hands
  initial_actions = {0,  1,  2,  13, 14, 15, 26, 27, 28, 39, 9,
                     10, 11, 12, 23, 24, 25, 36, 37, 38, 40};
  // Loop of drawing and discarding.
  for (int i = 0; i < 16; ++i) {
    initial_actions.push_back(52);
    initial_actions.push_back(0);
    initial_actions.push_back(52);
    initial_actions.push_back(12);
    initial_actions.push_back(52);
    initial_actions.push_back(1);
  }
  initial_actions.push_back(52);
  initial_actions.push_back(0);
  initial_actions.push_back(52);
  initial_actions.push_back(12);
  // 51st time an upcard is drawn ends the game in a draw.
  initial_actions.push_back(52);
  for (auto action : initial_actions) {
    state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 0);
  SPIEL_CHECK_EQ(returns[1], 0);
}

// Tests Oklahoma variation, where the value of the initial upcard determines
// the knock card. Oklahoma is standard in competitive play. It increases the
// skill level as correct strategy changes in response to the knock card.
void OklahomaTest() {
  GameParameters params;
  params["oklahoma"] = GameParameter(true);
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  initial_actions = {35, 37, 10, 11, 41, 14, 15, 16, 48, 49, 0, 1,
                     2,  3,  4,  5,  6,  7,  8,  51, 13, 54, 52};
  for (auto action : initial_actions) {
    state->ApplyAction(action);
  }
  std::cout << state->ToString() << std::endl;
  // Player turn: 1
  // Phase: Discard
  //
  // Player1: Deadwood=3
  // +--------------------------+
  // |As2s3s4s5s6s7s8s          |
  // |Ac                        |
  // |                          |
  // |  2h                    Kh|
  // +--------------------------+
  //
  // Stock size: 31  Upcard: XX
  // Discard pile:
  //
  // Player0: Deadwood=20
  // +--------------------------+
  // |                9sTsJsQsKs|
  // |  2c3c4c                  |
  // |                          |
  // |                  ThJh    |
  // +--------------------------+
  //
  // The initial upcard was the Ac, which was passed by Player0 and taken by
  // Player1. Player1 has 1 deadwood, but since we're playing Oklahoma that's
  // not low enough for a knock. In this case, the upcard was an ace, so both
  // players must play for gin.

  // Assert Player1 cannot knock.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_linear_search(legal_actions, kKnockAction));
  // Play continues.
  state->ApplyAction(51);
  state->ApplyAction(53);
  state->ApplyAction(26);
  state->ApplyAction(26);
  state->ApplyAction(52);
  std::cout << state->ToString() << std::endl;
  // Player turn: 1
  // Phase: Discard
  //
  // Player1: Deadwood=0
  // +--------------------------+
  // |As2s3s4s5s6s7s8s9s        |
  // |Ac                        |
  // |Ad                        |
  // |                          |
  // +--------------------------+
  //
  // Stock size: 30  Upcard: XX
  // Discard pile: Kh
  //
  // Player0: Deadwood=63
  // +--------------------------+
  // |                    JsQs  |
  // |  2c3c4c                  |
  // |                  Td  Qd  |
  // |    3h            ThJh    |
  // +--------------------------+
  //
  // Player1 can now knock with gin.
  legal_actions = state->LegalActions();
  SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, kKnockAction));
  state->ApplyAction(55);
  state->ApplyAction(8);
  state->ApplyAction(59);
  state->ApplyAction(122);
  state->ApplyAction(169);
  state->ApplyAction(54);
  state->ApplyAction(133);
  state->ApplyAction(54);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 0);
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[1], 88);  // 25 point gin bonus + 63 deadwood
  SPIEL_CHECK_EQ(returns[0], -88);
}

// Basic Observer functionality test.
void ObserverTest() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);

  std::shared_ptr<Observer> observer = game->MakeObserver(kDefaultObsType,
                                                          params);
  Observation observation = Observation(*game, observer);

  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  initial_actions = {1,  4,  5,  6,  17, 18, 19, 30, 31, 32, 2,  3,
                     16, 29, 43, 44, 45, 7,  20, 33, 0,  52, 55, 1};
  for (auto action : initial_actions) state->ApplyAction(action);
  std::cout << state->ToString() << std::endl;

  for (Player player = 0; player < kNumPlayers; ++player) {
    std::cout << observation.StringFrom(*state, player) << std::endl;
    observation.SetFrom(*state, player);
    std::cout << observation.Tensor() << std::endl;
    SPIEL_CHECK_EQ(observation.Tensor(), state->ObservationTensor(player));
    std::cout << state->InformationStateString(player) << std::endl;
  }
}

void KnownCardsTest() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  GinRummyState* gr_state = static_cast<GinRummyState*>(state.get());
  std::vector<Action> initial_actions;
  // Deal hands & first upcard
  initial_actions = {0,  1,  2,  13, 14, 15, 26, 27, 28, 39,
                     9, 10, 11, 12, 23, 24, 25, 36, 37, 38,
                     40};
  for (auto action : initial_actions) {
    gr_state->ApplyAction(action);
  }
  gr_state->ApplyAction(52);  // Player 0 picks upcard 40
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[0], 40));
  SPIEL_CHECK_FALSE(absl::c_linear_search(gr_state->KnownCards()[1], 0));
  state->ApplyAction(0);   // Player 0 discards 0
  state->ApplyAction(52);  // Player 1 picks upcard 0
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[0], 40));
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[1], 0));
  state->ApplyAction(12);  // Player 1 discards 12
  state->ApplyAction(52);  // Player 0 picks upcard 12
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[0], 40));
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[0], 12));
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[1], 0));
  state->ApplyAction(40);  // Player 0 discards 40
  SPIEL_CHECK_FALSE(absl::c_linear_search(gr_state->KnownCards()[0], 40));
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[0], 12));
  SPIEL_CHECK_TRUE(absl::c_linear_search(gr_state->KnownCards()[1], 0));
}

void ResampleFromInfostateTest() {
  int num_sims = 10;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy");
  std::mt19937 rng;
  UniformProbabilitySampler sampler;
  for (int i = 0; i < num_sims; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> other_state =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->ObservationString(p),
                         other_state->ObservationString(p));
          SPIEL_CHECK_EQ(state->ObservationTensor(p),
                         other_state->ObservationTensor(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), other_state->CurrentPlayer());
        }
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(rng)];
      state->ApplyAction(action);
    }
  }
}

void ResampleFromInfostateKnownCardsTest() {
  GameParameters params;
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> initial_actions;
  // Deal hands and first upcard
  initial_actions = {0,  1,  2,  13, 14, 15, 26, 27, 28, 39, 9,
                     10, 11, 12, 23, 24, 25, 36, 37, 38, 40};
  for (auto action : initial_actions) {
    state->ApplyAction(action);
  }
  // Player 0 passes on first upcard.
  state->ApplyAction(kPassAction);
  // Player 1 takes upcard (card 40).
  state->ApplyAction(kDrawUpcardAction);
  // Now card 40 is known to be in player 1's hand.
  // Resample from player 0's perspective.
  UniformProbabilitySampler sampler;
  std::unique_ptr<State> resampled_state =
      state->ResampleFromInfostate(0, sampler);
  GinRummyState* resampled_gr_state =
      static_cast<GinRummyState*>(resampled_state.get());

  // Check that card 40 is in player 1's hand in the resampled state.
  SPIEL_CHECK_TRUE(absl::c_linear_search(resampled_gr_state->Hands()[1], 40));
}

// TODO(jhtschultz) Add more extensive testing of parameterized deck size.
void DeckSizeTests() {
  const int kNumRanks = 10;
  const int kNumSuits = 3;
  const int kHandSize = 7;
  GinRummyUtils utils = GinRummyUtils(kNumRanks, kNumSuits, kHandSize);
  std::vector<int> full_deck;
  for (int i = 0; i < 30; ++i) full_deck.push_back(i);
  std::vector<std::vector<int>> all_melds = utils.AllMelds(full_deck);
  SPIEL_CHECK_EQ(all_melds.size(), 73);  // 73 melds in a 10x3 deck.
  // Check string representation of hand.
  SPIEL_CHECK_EQ(utils.HandToString(full_deck),
                 "+--------------------+\n"
                 "|As2s3s4s5s6s7s8s9sTs|\n"
                 "|Ac2c3c4c5c6c7c8c9cTc|\n"
                 "|Ad2d3d4d5d6d7d8d9dTd|\n"
                 "+--------------------+\n");
  // Random sims with 10x3 deck size.
  GameParameters params;
  params["num_ranks"] = GameParameter(10);
  params["num_suits"] = GameParameter(3);
  params["hand_size"] = GameParameter(7);
  std::shared_ptr<const Game> game = LoadGame("gin_rummy", params);
  testing::RandomSimTest(*game, 10);
}

void TestStructs() {
  auto game = LoadGame("gin_rummy");
  auto state = game->NewInitialState();
  // Apply some actions to get a non-initial state.
  std::vector<Action> initial_actions = {
      11, 4,  5, 6,  21, 22, 23, 12, 25, 38,  // P0 hand
      1,  14, 27, 40, 7,  20, 33, 8,  19, 13,  // P1 hand
      36                                      // upcard
  };
  for (auto action : initial_actions) state->ApplyAction(action);

  GinRummyState* gr_state = static_cast<GinRummyState*>(state.get());
  auto state_struct = gr_state->ToStruct();

  // Test state/state_struct -> json string.
  SPIEL_CHECK_EQ(state_struct->ToJson(), gr_state->ToJson());

  // Check some fields in the struct
  GinRummyStateStruct* gr_struct =
      static_cast<GinRummyStateStruct*>(state_struct.get());
  SPIEL_CHECK_EQ(gr_struct->phase, "FirstUpcard");
  SPIEL_CHECK_EQ(gr_struct->current_player, "Player_0");
  SPIEL_CHECK_EQ(gr_struct->stock_size, 31);
  SPIEL_CHECK_EQ(gr_struct->upcard.value(), "Jd");
  SPIEL_CHECK_EQ(gr_struct->hands[0].size(), 10);
  SPIEL_CHECK_EQ(gr_struct->hands[1].size(), 10);

  // Test json string -> state_struct
  std::string state_json = gr_state->ToJson();
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 GinRummyStateStruct(state_json).ToJson());

  auto obs_struct = gr_state->ToObservationStruct(0);

  // Check fields in the observation struct for player 0.
  GinRummyObservationStruct* gr_obs_struct =
      static_cast<GinRummyObservationStruct*>(obs_struct.get());
  SPIEL_CHECK_EQ(gr_obs_struct->phase, gr_struct->phase);
  SPIEL_CHECK_EQ(gr_obs_struct->current_player, gr_struct->current_player);
  SPIEL_CHECK_EQ(gr_obs_struct->stock_size, gr_struct->stock_size);
  SPIEL_CHECK_EQ(gr_obs_struct->upcard, gr_struct->upcard);
  SPIEL_CHECK_EQ(gr_obs_struct->hands[0], gr_struct->hands[0]);
  SPIEL_CHECK_EQ(gr_obs_struct->deadwood[0], gr_struct->deadwood[0]);
  SPIEL_CHECK_EQ(gr_obs_struct->hands[1].size(), 10);
  SPIEL_CHECK_EQ(gr_obs_struct->hands[1][0], "XX");
  SPIEL_CHECK_EQ(gr_obs_struct->deadwood[1], -1);
  SPIEL_CHECK_EQ(gr_obs_struct->observing_player, 0);

  // Test json string -> obs_struct
  std::string obs_json = obs_struct->ToJson();
  SPIEL_CHECK_EQ(nlohmann::json::parse(obs_json).dump(),
                 GinRummyObservationStruct(obs_json).ToJson());
}

// The observation struct must not reveal a hand whose owner has not knocked.
// During the Layoff phase the observer may be the knocker and the opponent the
// defender, whose remaining cards are still private.
void ObservationStructHidesDefenderHandTest() {
  std::shared_ptr<const Game> game = LoadGame("gin_rummy");
  std::unique_ptr<State> state = game->NewInitialState();
  // Same deal as GameplayTest3: Player0 knocks, Player1 defends.
  std::vector<Action> initial_actions = {
      10, 11, 12, 22, 35, 48, 13, 26, 1, 40, 9,  8,
      3,  16, 29, 42, 4,  17, 30, 43, 0, 52, 55, 1};
  for (auto action : initial_actions) state->ApplyAction(action);
  // Player0 lays its melds and passes, moving the hand into the Layoff phase.
  for (Action action : {59, 101, 131, 54}) state->ApplyAction(action);

  GinRummyState* gr_state = static_cast<GinRummyState*>(state.get());
  auto knocker_obs = gr_state->ToObservationStruct(0);
  auto* knocker_view =
      static_cast<GinRummyObservationStruct*>(knocker_obs.get());
  SPIEL_CHECK_EQ(knocker_view->phase, "Layoff");
  SPIEL_CHECK_TRUE(knocker_view->knocked[0]);
  SPIEL_CHECK_FALSE(knocker_view->knocked[1]);

  // The defender has not knocked, so the knocker must not see its cards.
  for (const std::string& card : knocker_view->hands[1]) {
    SPIEL_CHECK_EQ(card, "XX");
  }
  SPIEL_CHECK_EQ(knocker_view->deadwood[1], -1);

  // The knocker laid its hand face up, so the defender still sees it, and
  // each player always sees its own cards.
  auto defender_obs = gr_state->ToObservationStruct(1);
  auto* defender_view =
      static_cast<GinRummyObservationStruct*>(defender_obs.get());
  SPIEL_CHECK_FALSE(defender_view->hands[0].empty());
  for (const std::string& card : defender_view->hands[0]) {
    SPIEL_CHECK_NE(card, "XX");
  }
  for (const std::string& card : defender_view->hands[1]) {
    SPIEL_CHECK_NE(card, "XX");
  }
}

}  // namespace
}  // namespace gin_rummy
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::gin_rummy::BasicGameTests();
  open_spiel::gin_rummy::MeldTests();
  open_spiel::gin_rummy::ChainedLayoffTest();
  open_spiel::gin_rummy::MandatoryDiscardDeadwoodTest();
  open_spiel::gin_rummy::KnockAfterBestDiscardGameplayTest();
  open_spiel::gin_rummy::ChainedLayoffGameplayTest();
  open_spiel::gin_rummy::GameplayTest1();
  open_spiel::gin_rummy::GameplayTest2();
  open_spiel::gin_rummy::GameplayTest3();
  open_spiel::gin_rummy::EqualDeadwoodUndercutTest();
  open_spiel::gin_rummy::GinCannotBeUndercutTest();
  open_spiel::gin_rummy::MaxGameLengthTest();
  open_spiel::gin_rummy::WallTest();
  open_spiel::gin_rummy::OklahomaTest();
  open_spiel::gin_rummy::ObserverTest();
  open_spiel::gin_rummy::KnownCardsTest();
  open_spiel::gin_rummy::DeckSizeTests();
  open_spiel::gin_rummy::ResampleFromInfostateTest();
  open_spiel::gin_rummy::ResampleFromInfostateKnownCardsTest();
  open_spiel::gin_rummy::TestStructs();
  open_spiel::gin_rummy::ObservationStructHidesDefenderHandTest();
  std::cout << "Gin rummy tests passed!" << std::endl;
}
