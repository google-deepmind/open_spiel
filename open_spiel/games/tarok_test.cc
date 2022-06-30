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

#include "open_spiel/games/tarok.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tarok {

constexpr int kDealCardsAction = 0;
constexpr int kBidTwoAction = 3;
constexpr int kBidOneAction = 4;
constexpr int kBidSoloTwoAction = 6;
constexpr int kBidBeggarAction = 8;
constexpr int kBidSoloWithoutAction = 9;
constexpr int kBidOpenBeggarAction = 10;
constexpr int kBidColourValatAction = 11;
constexpr int kBidValatWithoutAction = 12;

const std::array<Card, 54> card_deck = InitializeCardDeck();

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

void DealtCardsSizeTest(int num_players) {
  auto [talon, players_cards] = DealCards(num_players, 42);
  SPIEL_CHECK_EQ(talon.size(), 6);
  int num_cards_per_player = 48 / num_players;
  for (auto const& player_cards : players_cards) {
    SPIEL_CHECK_EQ(player_cards.size(), num_cards_per_player);
  }
}

void DealtCardsContentTest(int num_players) {
  // 3 players
  auto [talon, players_cards] = DealCards(num_players, 42);
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
void TalonExchangePhaseTest1() {
  // 3 talon exchanges, select the first set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidThreeAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kThree);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 3);
  }

  // select the first set
  state->ApplyAction(0);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 1);
  SPIEL_CHECK_EQ(talon_initial.at(1), talon_end.at(0));
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(0), state->PlayerCards(state->CurrentPlayer())));

  // discard the first three cards
  auto legal_actions = state->LegalActions();
  for (int i = 0; i < 3; i++) {
    state->ApplyAction(legal_actions.at(i));
    SPIEL_CHECK_FALSE(AllActionsInOtherActions(
        {legal_actions.at(i)}, state->PlayerCards(state->CurrentPlayer())));
  }
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest2() {
  // 3 talon exchanges, select the second set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidThreeAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kThree);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 3);
  }

  // select the second set
  state->ApplyAction(1);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 1);
  SPIEL_CHECK_EQ(talon_initial.at(0), talon_end.at(0));
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(1), state->PlayerCards(state->CurrentPlayer())));

  // discard the first three cards
  auto legal_actions = state->LegalActions();
  for (int i = 0; i < 3; i++) {
    state->ApplyAction(legal_actions.at(i));
    SPIEL_CHECK_FALSE(AllActionsInOtherActions(
        {legal_actions.at(i)}, state->PlayerCards(state->CurrentPlayer())));
  }
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest3() {
  // 2 talon exchanges, select the middle set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidTwoAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kTwo);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 3);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1, 2}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 2);
  }

  // select the middle set
  state->ApplyAction(1);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 2);
  SPIEL_CHECK_EQ(talon_initial.at(0), talon_end.at(0));
  SPIEL_CHECK_EQ(talon_initial.at(2), talon_end.at(1));
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(1), state->PlayerCards(state->CurrentPlayer())));

  // discard the first two cards
  auto legal_actions = state->LegalActions();
  for (int i = 0; i < 2; i++) {
    state->ApplyAction(legal_actions.at(i));
    SPIEL_CHECK_FALSE(AllActionsInOtherActions(
        {legal_actions.at(i)}, state->PlayerCards(state->CurrentPlayer())));
  }
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest4() {
  // 1 talon exchange, select the first set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 6);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1, 2, 3, 4, 5}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 1);
  }

  // select the first set
  state->ApplyAction(0);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 5);
  for (int i = 1; i < 6; i++) {
    SPIEL_CHECK_EQ(talon_initial.at(i), talon_end.at(i - 1));
  }
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(0), state->PlayerCards(state->CurrentPlayer())));

  // discard the last card
  auto legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions.at(legal_actions.size() - 1));
  SPIEL_CHECK_FALSE(
      AllActionsInOtherActions({legal_actions.at(legal_actions.size() - 1)},
                               state->PlayerCards(state->CurrentPlayer())));
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest5() {
  // 1 talon exchange, select the fourth set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 6);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1, 2, 3, 4, 5}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 1);
  }

  // select the fourth set
  state->ApplyAction(3);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 5);
  for (int i = 0; i < 5; i++) {
    if (i < 3)
      SPIEL_CHECK_EQ(talon_initial.at(i), talon_end.at(i));
    else
      SPIEL_CHECK_EQ(talon_initial.at(i + 1), talon_end.at(i));
  }
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(3), state->PlayerCards(state->CurrentPlayer())));

  // discard the second card
  auto legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions.at(1));
  SPIEL_CHECK_FALSE(AllActionsInOtherActions(
      {legal_actions.at(1)}, state->PlayerCards(state->CurrentPlayer())));
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest6() {
  // 1 talon exchange, select the last set
  auto state = StateAfterActions(
      GameParameters(),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);
  auto talon_initial = state->TalonSets();
  SPIEL_CHECK_EQ(talon_initial.size(), 6);
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 1, 2, 3, 4, 5}));
  for (auto const& talon_set : talon_initial) {
    SPIEL_CHECK_EQ(talon_set.size(), 1);
  }

  // select the last set
  state->ApplyAction(5);
  auto talon_end = state->TalonSets();
  SPIEL_CHECK_EQ(talon_end.size(), 5);
  for (int i = 0; i < 5; i++) {
    SPIEL_CHECK_EQ(talon_initial.at(i), talon_end.at(i));
  }
  SPIEL_CHECK_TRUE(AllActionsInOtherActions(
      talon_initial.at(5), state->PlayerCards(state->CurrentPlayer())));

  // discard the first card
  auto legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions.at(0));
  SPIEL_CHECK_FALSE(AllActionsInOtherActions(
      {legal_actions.at(0)}, state->PlayerCards(state->CurrentPlayer())));
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

void TalonExchangePhaseTest7() {
  // check that taroks and kings cannot be exchanged
  auto state =
      StateAfterActions(GameParameters({{"rng_seed", GameParameter(42)}}),
                        {kDealCardsAction, kBidPassAction, kBidOneAction,
                         kBidPassAction, kBidOneAction, 1});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  // check taroks and kings are not in legal actions
  for (auto const& action : state->LegalActions()) {
    const Card& card = card_deck.at(action);
    SPIEL_CHECK_TRUE(card.suit != CardSuit::kTaroks);
    SPIEL_CHECK_NE(card.points, 5);
  }
}

void TalonExchangePhaseTest8() {
  // check that tarok can be exchanged if player has no other choice
  auto state =
      StateAfterActions(GameParameters({{"players", GameParameter(4)},
                                        {"rng_seed", GameParameter(141750)}}),
                        {kDealCardsAction, kBidPassAction, kBidPassAction,
                         kBidPassAction, kBidSoloTwoAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kSoloTwo);

  // select first set from talon
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);

  // first the player must exchange non-tarok or non-king card
  // check taroks and kings are not in legal actions
  for (auto const& action : state->LegalActions()) {
    const Card& card = card_deck.at(action);
    SPIEL_CHECK_TRUE(card.suit != CardSuit::kTaroks);
    SPIEL_CHECK_NE(card.points, 5);
  }
  state->ApplyAction(state->LegalActions().at(0));
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTalonExchange);

  // at this point the player has only taroks and kings in his hand but still
  // needs to exchange one card
  // check only taroks (no trula or kings) are in legal actions
  for (auto const& action : state->LegalActions()) {
    const Card& card = card_deck.at(action);
    SPIEL_CHECK_TRUE(card.suit == CardSuit::kTaroks);
    SPIEL_CHECK_NE(card.points, 5);
  }
  state->ApplyAction(state->LegalActions().at(0));
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
}

// tricks playing phase tests
static inline const GameParameters kTricksPlayingGameParams = GameParameters(
    {{"players", GameParameter(3)}, {"rng_seed", GameParameter(634317)}});

// the above "rng_seed" yields:
//
// player 0 cards:
// ('II', 1), ('IIII', 3), ('V', 4), ('VIII', 7), ('XI', 10), ('XIX', 18),
// ('Mond', 20), ('Jack of Hearts', 26), ('Knight of Hearts', 27), ('4 of
// Diamonds', 30), ('8 of Spades', 39), ('Jack of Spades', 42), ('King of
// Spades', 45), ('10 of Clubs', 49), ('Jack of Clubs', 50), ('Knight of Clubs',
// 51)
//
// player 1 cards:
// ('III', 2), ('VII', 6), ('XII', 11), ('XIII', 12), ('XIV', 13), ('XX', 19),
// ('Skis', 21), ('1 of Hearts', 25), ('3 of Diamonds', 31), ('Knight of
// Diamonds', 35), ('Queen of Diamonds', 36), ('King of Diamonds', 37), ('7 of
// Spades', 38), ('Knight of Spades', 43), ('8 of Clubs', 47), ('Queen of
// Clubs', 52)
//
// player 2 cards:
// ('Pagat', 0), ('VI', 5), ('IX', 8), ('X', 9), ('XV', 14), ('XVI', 15),
// ('XVII', 16), ('XVIII', 17), ('4 of Hearts', 22), ('2 of Diamonds', 32), ('1
// of Diamonds', 33), ('Jack of Diamonds', 34), ('9 of Spades', 40), ('10 of
// Spades', 41), ('9 of Clubs', 48), ('King of Clubs', 53)

void TricksPlayingPhaseTest1() {
  // check forced pagat in klop
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(20);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {20}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {21}));
  state->ApplyAction(21);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {20, 21}));
  // pagat is forced
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0}));
  state->ApplyAction(0);
  // pagat won the trick
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

void TricksPlayingPhaseTest2() {
  // check pagat not a legal action in klop when following and all taroks lower
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(18);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {18}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {19, 21}));
  state->ApplyAction(21);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {18, 21}));
  // pagat not available but all other taroks available
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {5, 8, 9, 14, 15, 16, 17}));
  state->ApplyAction(17);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

void TricksPlayingPhaseTest3() {
  // check pagat not a legal action in klop when opening
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(4);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {4}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {6, 11, 12, 13, 19, 21}));
  state->ApplyAction(6);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {4, 6}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {8, 9, 14, 15, 16, 17}));
  state->ApplyAction(8);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {5, 9, 14, 15, 16, 17, 22, 32,
                                                 33, 34, 40, 41, 48, 53}));
}

void TricksPlayingPhaseTest4() {
  // check legal non-tarok cards in klop
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(42);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {42}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {43}));
  state->ApplyAction(43);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {42, 43}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {40, 41}));
  state->ApplyAction(41);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

void TricksPlayingPhaseTest5() {
  // check scenarios where no card has to be beaten in klop
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(30);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {30}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {31, 35, 36, 37}));
  state->ApplyAction(37);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {30, 37}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {32, 33, 34}));
  state->ApplyAction(34);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {2, 6, 11, 12, 13, 19, 21, 25,
                                                 31, 35, 36, 38, 43, 47, 52}));
  state->ApplyAction(52);
  state->ApplyAction(53);
  state->ApplyAction(51);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(),
                         {5, 8, 9, 14, 15, 16, 17, 22, 32, 33, 40, 41, 48}));
  state->ApplyAction(32);

  // can't follow suit, i.e. forced to play tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {32}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20}));
  state->ApplyAction(1);

  // doesn't have to beat the opening card due to the second card being tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {32, 1}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {31, 35, 36}));
  state->ApplyAction(36);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

void TricksPlayingPhaseTest6() {
  // check taroks don't win in colour valat
  auto state = StateAfterActions(
      kTricksPlayingGameParams,
      {kDealCardsAction, kBidColourValatAction, kBidPassAction, kBidPassAction,
       kBidColourValatAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(),
                 ContractName::kColourValatWithout);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(
      AllEq(state->LegalActions(),
            {2, 6, 11, 12, 13, 19, 21, 25, 31, 35, 36, 37, 38, 43, 47, 52}));
  state->ApplyAction(35);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {35}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {32, 33, 34}));
  state->ApplyAction(32);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {35, 32}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {30}));
  state->ApplyAction(30);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {2, 6, 11, 12, 13, 19, 21, 25,
                                                 31, 36, 37, 38, 43, 47, 52}));
  state->ApplyAction(37);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {37}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {33, 34}));
  state->ApplyAction(33);

  // can't follow suit, i.e. forced to play tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {37, 33}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20}));
  state->ApplyAction(1);

  // tarok didn't win the trick
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

void TricksPlayingPhaseTest7() {
  // check positive contracts scenarios
  auto state =
      StateAfterActions(kTricksPlayingGameParams,
                        {kDealCardsAction, kBidPassAction, kBidTwoAction,
                         kBidPassAction, kBidTwoAction, 0, 40, 41});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kTwo);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20, 26, 27,
                                                 30, 39, 42, 45, 49, 50, 51}));
  state->ApplyAction(30);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {30}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {31, 35, 36, 37}));
  state->ApplyAction(31);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {30, 31}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {32, 33, 34}));
  state->ApplyAction(32);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {0, 5, 8, 9, 14, 15, 16, 17, 22,
                                                 24, 28, 33, 34, 48, 53}));
  state->ApplyAction(33);

  // can't follow suit, i.e. forced to play tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {33}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {1, 3, 4, 7, 10, 18, 20}));
  state->ApplyAction(18);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->TrickCards(), {33, 18}));
  SPIEL_CHECK_TRUE(AllEq(state->LegalActions(), {35, 36, 37}));
  state->ApplyAction(37);

  // tarok won the trick
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->TrickCards().empty());
}

// captured mond tests
void CapturedMondTest1() {
  // mond captured by skis
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(634317)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction, 0, 49});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play low tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("VI"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {-20, 0, 0}));
}

void CapturedMondTest2() {
  // mond captured by pagat (emperor trick)
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(634317)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction, 0, 49});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play pagat
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("Pagat"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {-20, 0, 0}));
}

void CapturedMondTest3() {
  // mond taken from talon
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(497200)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction, 3, 49});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {0, 0, 0}));
}

void CapturedMondTest4() {
  // mond left in talon
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(497200)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOneAction, 0, 49});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {-20, 0, 0}));
}

void CapturedMondTest5() {
  // mond left in talon but won with a called king
  auto state = StateAfterActions(
      GameParameters(
          {{"players", GameParameter(4)}, {"rng_seed", GameParameter(297029)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidPassAction,
       kBidOneAction, kKingOfSpadesAction, 2, 49});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  // play the called king and win the trick
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {-20, 0, 0, 0}));
  state->ApplyAction(CardLongNameToAction("King of Spades"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Queen of Spades"));
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("8 of Spades"));
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
  state->ApplyAction(CardLongNameToAction("7 of Spades"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {0, 0, 0, 0}));
}

void CapturedMondTest6() {
  // mond captured by ally should also be penalized
  auto state =
      StateAfterActions(GameParameters({{"rng_seed", GameParameter(634317)}}),
                        {kDealCardsAction, kBidPassAction, kBidOneAction,
                         kBidPassAction, kBidOneAction, 0, 22});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOne);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play low tarok
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("VI"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {-20, 0, 0}));
}

void CapturedMondTest7() {
  // mond captured in klop should not be penalized
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(634317)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidKlopAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kKlop);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play pagat
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("Pagat"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {0, 0, 0}));
}

void CapturedMondTest8() {
  // mond captured in bagger should not be penalized
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(634317)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidBeggarAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kBeggar);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play pagat
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("Pagat"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {0, 0, 0}));
}

void CapturedMondTest9() {
  // mond captured in open bagger should not be penalized
  auto state = StateAfterActions(
      GameParameters({{"rng_seed", GameParameter(634317)}}),
      {kDealCardsAction, kBidPassAction, kBidPassAction, kBidOpenBeggarAction});
  SPIEL_CHECK_EQ(state->CurrentGamePhase(), GamePhase::kTricksPlaying);
  SPIEL_CHECK_EQ(state->SelectedContractName(), ContractName::kOpenBeggar);

  // play mond
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(CardLongNameToAction("Mond"));
  // play skis
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(CardLongNameToAction("Skis"));
  // play pagat
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(CardLongNameToAction("Pagat"));

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_TRUE(AllEq(state->CapturedMondPenalties(), {0, 0, 0}));
}

}  // namespace tarok
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tarok::BasicGameTests();
  // cards tests
  open_spiel::tarok::CardDeckShufflingSeedTest();
  open_spiel::tarok::DealtCardsSizeTest(3);
  open_spiel::tarok::DealtCardsSizeTest(4);
  open_spiel::tarok::DealtCardsContentTest(3);
  open_spiel::tarok::DealtCardsContentTest(4);
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
  open_spiel::tarok::TalonExchangePhaseTest1();
  open_spiel::tarok::TalonExchangePhaseTest2();
  open_spiel::tarok::TalonExchangePhaseTest3();
  open_spiel::tarok::TalonExchangePhaseTest4();
  open_spiel::tarok::TalonExchangePhaseTest5();
  open_spiel::tarok::TalonExchangePhaseTest6();
  open_spiel::tarok::TalonExchangePhaseTest7();
  open_spiel::tarok::TalonExchangePhaseTest8();
  // tricks playing phase tests
  open_spiel::tarok::TricksPlayingPhaseTest1();
  open_spiel::tarok::TricksPlayingPhaseTest2();
  open_spiel::tarok::TricksPlayingPhaseTest3();
  open_spiel::tarok::TricksPlayingPhaseTest4();
  open_spiel::tarok::TricksPlayingPhaseTest5();
  open_spiel::tarok::TricksPlayingPhaseTest6();
  open_spiel::tarok::TricksPlayingPhaseTest7();
  // captured mond tests
  open_spiel::tarok::CapturedMondTest1();
  open_spiel::tarok::CapturedMondTest2();
  open_spiel::tarok::CapturedMondTest3();
  open_spiel::tarok::CapturedMondTest4();
  open_spiel::tarok::CapturedMondTest5();
  open_spiel::tarok::CapturedMondTest6();
  open_spiel::tarok::CapturedMondTest7();
  open_spiel::tarok::CapturedMondTest8();
  open_spiel::tarok::CapturedMondTest9();
}
