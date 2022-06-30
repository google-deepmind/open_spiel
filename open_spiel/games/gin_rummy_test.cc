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

#include "open_spiel/games/gin_rummy.h"

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
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

// An extremely rare situation, but one that does arise in actual gameplay.
// Tests both layoff and undercut functionality.
void GameplayTest1() {
  GameParameters params;
  // Modify undercut bonus game parameter as an additional test.
  params["undercut_bonus"] = GameParameter(20);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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

// Tests action on the 50th card, and tests that layoffs are not allowed when
// the knocking player has gin.
void WallTest() {
  GinRummyUtils utils = GinRummyUtils(kDefaultNumRanks, kDefaultNumSuits,
                                      kDefaultHandSize);
  GameParameters params;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);

  std::shared_ptr<Observer> observer = game->MakeObserver(kDefaultObsType,
                                                          params);
  Observation observation = Observation(*game, observer);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("gin_rummy", params);
  testing::RandomSimTest(*game, 10);
}

}  // namespace
}  // namespace gin_rummy
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::gin_rummy::BasicGameTests();
  open_spiel::gin_rummy::MeldTests();
  open_spiel::gin_rummy::GameplayTest1();
  open_spiel::gin_rummy::GameplayTest2();
  open_spiel::gin_rummy::GameplayTest3();
  open_spiel::gin_rummy::MaxGameLengthTest();
  open_spiel::gin_rummy::WallTest();
  open_spiel::gin_rummy::OklahomaTest();
  open_spiel::gin_rummy::ObserverTest();
  open_spiel::gin_rummy::DeckSizeTests();
  std::cout << "Gin rummy tests passed!" << std::endl;
}
