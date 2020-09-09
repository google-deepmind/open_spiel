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

#include "open_spiel/games/tarok.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tarok {

static constexpr int kDealCardsAction = 0;
static constexpr int kBidTwoAction = 3;
static constexpr int kBidOneAction = 4;
static constexpr int kBidSoloTwoAction = 6;
static constexpr int kBidBeggarAction = 8;
static constexpr int kBidSoloWithoutAction = 9;
static constexpr int kBidOpenBeggarAction = 10;
static constexpr int kBidColourValatAction = 11;
static constexpr int kBidValatWithoutAction = 12;

static inline const std::array<Card, 54> card_deck = InitializeCardDeck();

// helper methods
std::shared_ptr<const TarokGame> NewTarokGame(const GameParameters& params) {
  return std::static_pointer_cast<const TarokGame>(LoadGame("tarok", params));
}

std::unique_ptr<TarokState> StateAfterActions(
    const GameParameters& params, const std::vector<Action>& actions) {
  auto state = NewTarokGame(params)->NewInitialTarokState();
  for (auto const& action : actions) {
    state->ApplyAction(action);
  }
  return state;
}

bool AllActionsInOtherActions(const std::vector<Action>& actions,
                              const std::vector<Action>& other_actions) {
  for (auto const& action : actions) {
    if (std::find(other_actions.begin(), other_actions.end(), action) ==
        other_actions.end()) {
      return false;
    }
  }
  return true;
}

Action CardLongNameToAction(const std::string& long_name) {
  for (int i = 0; i < card_deck.size(); i++) {
    if (card_deck.at(i).long_name == long_name) return i;
  }
  SpielFatalError("Invalid long_name!");
  return -1;
}

std::vector<Action> CardLongNamesToActions(
    const std::vector<std::string>& long_names) {
  std::vector<Action> actions;
  actions.reserve(long_names.size());
  for (auto const long_name : long_names) {
    actions.push_back(CardLongNameToAction(long_name));
  }
  return actions;
}

template <typename T>
bool AllEq(const std::vector<T>& xs0, const std::vector<T>& xs1) {
  if (xs0.size() != xs1.size()) return false;
  for (int i = 0; i < xs0.size(); i++) {
    if (xs0.at(i) != xs1.at(i)) return false;
  }
  return true;
}

// testing
void BasicGameTests() {
  testing::LoadGameTest("tarok");
  testing::ChanceOutcomesTest(*LoadGame("tarok"));
  testing::RandomSimTest(*LoadGame("tarok"), 100);
}

// cards tests
void CardDeckShufflingSeedTest() {
  auto game = NewTarokGame(GameParameters({{"rng_seed", GameParameter(0)}}));

  // subsequent shuffles within the same game should be different
  auto state1 = game->NewInitialTarokState();
  state1->ApplyAction(0);
  auto state2 = game->NewInitialTarokState();
  state2->ApplyAction(0);
  SPIEL_CHECK_NE(state1->PlayerCards(0), state2->PlayerCards(0));

  game = NewTarokGame(GameParameters({{"rng_seed", GameParameter(0)}}));
  // shuffles should be the same when recreating a game with the same seed
  auto state3 = game->NewInitialTarokState();
  state3->ApplyAction(0);
  auto state4 = game->NewInitialTarokState();
  state4->ApplyAction(0);
  SPIEL_CHECK_EQ(state1->PlayerCards(0), state3->PlayerCards(0));
  SPIEL_CHECK_EQ(state2->PlayerCards(0), state4->PlayerCards(0));
}

void DealtCardsSizeTest() {
  auto [talon, players_cards] = DealCards(3, 42);
  SPIEL_CHECK_EQ(talon.size(), 6);
  for (auto const& player_cards : players_cards) {
    SPIEL_CHECK_EQ(player_cards.size(), 16);
  }
}

void DealtCardsContentTest() {
  auto [talon, players_cards] = DealCards(3, 42);
  // flatten and sort all the dealt cards
  std::vector<int> all_dealt_cards(talon.begin(), talon.end());
  for (auto const& player_cards : players_cards) {
    all_dealt_cards.insert(all_dealt_cards.end(), player_cards.begin(),
                           player_cards.end());
  }
  std::sort(all_dealt_cards.begin(), all_dealt_cards.end());

  // check the actual content
  for (int i = 0; i < 54; i++) {
    SPIEL_CHECK_EQ(all_dealt_cards.at(i), i);
  }
}

void PlayersCardsSortedTest() {
  auto [talon, players_cards] = DealCards(3, 42);
  for (auto const& player_cards : players_cards) {
    SPIEL_CHECK_TRUE(std::is_sorted(player_cards.begin(), player_cards.end()));
  }
}

void CountCardsTest() {
  std::vector<Action> all_card_actions(54);
  std::iota(all_card_actions.begin(), all_card_actions.end(), 0);
  SPIEL_CHECK_EQ(CardPoints(all_card_actions, card_deck), 70);
  SPIEL_CHECK_EQ(CardPoints({}, card_deck), 0);
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions({"II"}), card_deck), 0);
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions({"II", "III"}), card_deck),
                 1);
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions({"Mond"}), card_deck), 4);

  std::vector<std::string> cards{"Mond", "Jack of Diamonds"};
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions(cards), card_deck), 6);

  cards = {"XIV", "Mond", "Jack of Diamonds"};
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions(cards), card_deck), 6);

  cards = {"XIV", "Mond", "Jack of Diamonds", "Queen of Diamonds"};
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions(cards), card_deck), 9);

  cards = {"II", "Jack of Clubs", "Queen of Clubs", "Mond", "King of Clubs"};
  SPIEL_CHECK_EQ(CardPoints(CardLongNamesToActions(cards), card_deck), 14);
}

void CardDealingPhaseTest() {
  auto game = NewTarokGame(GameParameters());
  auto state = game->NewInitialTarokState();
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kCardDealing);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_TRUE(state->TalonSets().empty());
  for (int i = 0; i < game->NumPlayers(); i++) {
    SPIEL_CHECK_TRUE(state->PlayerCards(i).empty());
  }

  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0}));
  SPIEL_CHECK_TRUE(AllEq(state->ChanceOutcomes(), {{0, 1.0}}));

  // deal the cards
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_NE(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  // talon sets are only visible in the talon exchange phase
  SPIEL_CHECK_TRUE(state->TalonSets().empty());
  SPIEL_CHECK_EQ(state->Talon().size(), 6);
  for (int i = 0; i < game->NumPlayers(); i++) {
    SPIEL_CHECK_FALSE(state->PlayerCards(i).empty());
  }
  SPIEL_CHECK_TRUE(state->ChanceOutcomes().empty());
}

// bidding phase tests
void BiddingPhase3PlayersTest1() {
  // scenario: all players pass
  auto game = NewTarokGame(GameParameters());
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidKlopAction, kBidThreeAction, kBidTwoAction, kBidOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidKlopAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
}

void BiddingPhase3PlayersTest2() {
  // scenario: forehand passes, player 1 eventually bids beggar, player 2 bids
  // beggar
  auto game = NewTarokGame(GameParameters());
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidTwoAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kBeggar);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void BiddingPhase3PlayersTest3() {
  // scenario: forehand passes, player 1 bids beggar, player 2 bids solo without
  auto game = NewTarokGame(GameParameters());
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidSoloWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(),
                         {kBidSoloWithoutAction, kBidOpenBeggarAction,
                          kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidSoloWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kSoloWithout);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
}

void BiddingPhase3PlayersTest4() {
  // scenario: forehand bids valat without, others are forced to pass, todo: we
  // could check this case in DoApplyActionInBidding and simply finish the
  // bidding phase early
  auto game = NewTarokGame(GameParameters());
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidTwoAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidOneAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidValatWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {kBidPassAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {kBidPassAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {kBidValatWithoutAction}));
  state->ApplyAction(kBidValatWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kValatWithout);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
}

void BiddingPhase4PlayersTest1() {
  // scenario: all players pass
  auto game = NewTarokGame(GameParameters({{"players", GameParameter(4)}}));
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidKlopAction, kBidThreeAction, kBidTwoAction, kBidOneAction,
             kBidSoloThreeAction, kBidSoloTwoAction, kBidSoloOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidKlopAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
}

void BiddingPhase4PlayersTest2() {
  // scenario: forehand bids one, player 2 bids one, others pass
  auto game = NewTarokGame(GameParameters({{"players", GameParameter(4)}}));
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidOneAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidSoloThreeAction, kBidSoloTwoAction,
       kBidSoloOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidOneAction, kBidSoloThreeAction, kBidSoloTwoAction,
       kBidSoloOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidOneAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidSoloThreeAction, kBidSoloTwoAction,
       kBidSoloOneAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidOneAction, kBidSoloThreeAction, kBidSoloTwoAction, kBidSoloOneAction,
       kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
       kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidOneAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kKingCalling);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
}

void BiddingPhase4PlayersTest3() {
  // scenario: player 1 bids solo three, player 3 eventually bids solo one,
  // others pass
  auto game = NewTarokGame(GameParameters({{"players", GameParameter(4)}}));
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidSoloThreeAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloTwoAction, kBidSoloOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloTwoAction, kBidSoloOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidSoloTwoAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloTwoAction, kBidSoloOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloTwoAction, kBidSoloOneAction,
             kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidSoloOneAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kSoloOne);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
}

void BiddingPhase4PlayersTest4() {
  // scenario: player 2 bids beggar, others pass
  auto game = NewTarokGame(GameParameters({{"players", GameParameter(4)}}));
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(
      state->LegalActions(),
      {kBidPassAction, kBidBeggarAction, kBidSoloWithoutAction,
       kBidOpenBeggarAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidBeggarAction, kBidSoloWithoutAction, kBidOpenBeggarAction,
             kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kBeggar);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
}

void BiddingPhase4PlayersTest5() {
  // scenario: forehand passes, player 1 bids open beggar, player 2 bids colour
  // valat without, player 3 bids valat without
  auto game = NewTarokGame(GameParameters({{"players", GameParameter(4)}}));
  auto state = game->NewInitialTarokState();
  state->ApplyAction(kDealCardsAction);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kBidding);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kNotSelected);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidTwoAction, kBidOneAction, kBidSoloThreeAction,
             kBidSoloTwoAction, kBidSoloOneAction, kBidBeggarAction,
             kBidSoloWithoutAction, kBidOpenBeggarAction, kBidColourValatAction,
             kBidValatWithoutAction}));
  state->ApplyAction(kBidOpenBeggarAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {kBidPassAction, kBidColourValatAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidColourValatAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(), {kBidPassAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidValatWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(), {kBidPassAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(), {kBidPassAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(), {kBidPassAction, kBidValatWithoutAction}));
  state->ApplyAction(kBidPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {kBidValatWithoutAction}));
  state->ApplyAction(kBidValatWithoutAction);

  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kValatWithout);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
}

// talon exchange phase tests

}  // namespace tarok
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tarok::BasicGameTests();
  // cards tests
  open_spiel::tarok::CardDeckShufflingSeedTest();
  open_spiel::tarok::DealtCardsSizeTest();
  open_spiel::tarok::DealtCardsContentTest();
  open_spiel::tarok::PlayersCardsSortedTest();
  open_spiel::tarok::CountCardsTest();
  open_spiel::tarok::CardDealingPhaseTest();
  // bidding phase tests
  open_spiel::tarok::BiddingPhase3PlayersTest1();
  open_spiel::tarok::BiddingPhase3PlayersTest2();
  open_spiel::tarok::BiddingPhase3PlayersTest3();
  open_spiel::tarok::BiddingPhase3PlayersTest4();
  open_spiel::tarok::BiddingPhase4PlayersTest1();
  open_spiel::tarok::BiddingPhase4PlayersTest2();
  open_spiel::tarok::BiddingPhase4PlayersTest3();
  open_spiel::tarok::BiddingPhase4PlayersTest4();
  open_spiel::tarok::BiddingPhase4PlayersTest5();
  // talon exchange phase tests
}
