// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/seep/seep.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace seep {
namespace {

namespace testing = open_spiel::testing;

bool ContainsAction(const std::vector<Action>& actions, Action target) {
  return std::find(actions.begin(), actions.end(), target) != actions.end();
}

void CardAndActionEncodingTests() {
  int total_deck_points = 0;
  for (int c = 0; c < kNumCards; ++c) {
    total_deck_points += CardPointValue(c);
    std::string str = CardString(c);
    SPIEL_CHECK_EQ(CardFromString(str), c);
    SPIEL_CHECK_GE(CardCaptureValue(c), 1);
    SPIEL_CHECK_LE(CardCaptureValue(c), 13);
  }
  // Total points across the 52-card Seep deck must equal 100.
  SPIEL_CHECK_EQ(total_deck_points, kDeckTotalPoints);

  // Specific card point checks: Spades = capture value, non-Spade Aces = 1,
  // 10 of Diamonds = 6, others = 0.
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("AS")), 1);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("KS")), 13);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("AC")), 1);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("AD")), 1);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("AH")), 1);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("TD")), 6);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("TC")), 0);
  SPIEL_CHECK_EQ(CardPointValue(CardFromString("KH")), 0);

  // Invalid CardFromString inputs must return kInvalidAction.
  SPIEL_CHECK_EQ(CardFromString(""), kInvalidAction);
  SPIEL_CHECK_EQ(CardFromString("10S"), kInvalidAction);
  SPIEL_CHECK_EQ(CardFromString("1S"), kInvalidAction);
  SPIEL_CHECK_EQ(CardFromString("9X"), kInvalidAction);
  SPIEL_CHECK_EQ(CardFromString("as"), kInvalidAction);

  // Bid action encoding round-trip.
  for (int bid = kMinBidValue; bid <= kMaxBidValue; ++bid) {
    Action a = MakeBidAction(bid);
    SPIEL_CHECK_GE(a, 0);
    SPIEL_CHECK_LT(a, kPlayActionBase);
    SPIEL_CHECK_EQ(BidValueFromAction(a), bid);
  }

  // Play action encoding round-trip.
  for (int c = 0; c < kNumCards; ++c) {
    for (int i = 0; i < kNumIntents; ++i) {
      Intent intent = static_cast<Intent>(i);
      Action a = MakePlayAction(c, intent);
      SPIEL_CHECK_GE(a, kPlayActionBase);
      SPIEL_CHECK_LT(a, kNumDistinctActions);
      auto [decoded_card, decoded_intent] = DecodePlayAction(a);
      SPIEL_CHECK_EQ(decoded_card, c);
      SPIEL_CHECK_EQ(decoded_intent, intent);
    }
  }
}

void InitialDealAndBiddingTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* seep_state = static_cast<SeepState*>(state.get());

  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kDealInitial);
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Deal 4 low cards to Player 0: 3C, 4D, 6H, 8S (all < 9).
  state->ApplyAction(CardFromString("3C"));
  state->ApplyAction(CardFromString("4D"));
  state->ApplyAction(CardFromString("6H"));
  SPIEL_CHECK_EQ(seep_state->Hand(0).size(), 3);
  SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
  state->ApplyAction(CardFromString("8S"));

  // Player 0's 4 hand cards had no card >= 9, so all 4 cards are returned to
  // the deck, reshuffled, and dealt again!
  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kDealInitial);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_EQ(seep_state->NumRedeals(), 1);
  SPIEL_CHECK_EQ(seep_state->NumCardsDealt(), 0);
  SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
  SPIEL_CHECK_TRUE(seep_state->Hand(0).empty());
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumCards);

  // Now deal 4 cards to Player 0 including 9S (>= 9).
  state->ApplyAction(CardFromString("3C"));
  state->ApplyAction(CardFromString("4D"));
  state->ApplyAction(CardFromString("6H"));
  state->ApplyAction(CardFromString("9S"));
  SPIEL_CHECK_EQ(seep_state->Hand(0).size(), 4);
  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kBid);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // Crucially, during Phase::kBid, no floor cards have been dealt or revealed
  // yet: FloorPiles() and ToObservationStruct(0)->floor_piles must be empty!
  SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
  std::unique_ptr<ObservationStruct> bid_obs = state->ToObservationStruct(0);
  auto* seep_bid_obs = static_cast<SeepObservationStruct*>(bid_obs.get());
  SPIEL_CHECK_TRUE(seep_bid_obs->floor_piles.empty());
  SPIEL_CHECK_EQ(seep_bid_obs->hands[0].size(), 4);

  // Player 0 only holds 9S among >= 9 cards, so only bidding 9 is legal.
  std::vector<Action> bid_actions = state->LegalActions();
  SPIEL_CHECK_EQ(bid_actions.size(), 1);
  SPIEL_CHECK_EQ(bid_actions[0], MakeBidAction(9));

  // Apply the bid of 9. Only AFTER the bid is made does the game enter
  // Phase::kDealFloor to deal the 4 face-up floor cards.
  state->ApplyAction(MakeBidAction(9));
  SPIEL_CHECK_EQ(seep_state->BidValue(), 9);
  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kDealFloor);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());

  // Deal the 4 floor cards (2C, 5D, JH, KC).
  state->ApplyAction(CardFromString("2C"));
  state->ApplyAction(CardFromString("5D"));
  state->ApplyAction(CardFromString("JH"));
  state->ApplyAction(CardFromString("KC"));
  SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 4);
  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kPlay);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(seep_state->Hand(0).size(), 4);

  // Player 0 plays their opening bid move from their 4-card hand.
  std::vector<Action> turn0_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(turn0_actions.empty());
  state->ApplyAction(turn0_actions[0]);
  SPIEL_CHECK_EQ(seep_state->Hand(0).size(), 3);
  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kDealRemainder);
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Deal the remaining 44 cards.
  while (state->IsChanceNode()) {
    std::vector<Action> deal_actions = state->LegalActions();
    SPIEL_CHECK_FALSE(deal_actions.empty());
    state->ApplyAction(deal_actions[0]);
  }

  SPIEL_CHECK_EQ(seep_state->CurrentPhase(), Phase::kPlay);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(seep_state->Hand(0).size(), kCardsPerPlayer - 1);
  for (Player p = 1; p < kNumPlayers; ++p) {
    SPIEL_CHECK_EQ(seep_state->Hand(p).size(), kCardsPerPlayer);
  }
}

void FirstTurnBidConstraintTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* seep_state = static_cast<SeepState*>(state.get());

  // Set up Turn 0 where Player 0 bid 10, holds {TD, TS, 4C, 7H}, and floor has
  // {6C, 2D}.
  std::array<std::vector<int>, kNumPlayers> hands;
  hands[0] = {CardFromString("4C"), CardFromString("7H"), CardFromString("TD"),
              CardFromString("TS")};
  hands[1] = {CardFromString("AC")};
  hands[2] = {CardFromString("AD")};
  hands[3] = {CardFromString("AH")};
  std::vector<House> floor = {
      House{6, false, {CardFromString("6C")}},
      House{2, false, {CardFromString("2D")}},
  };
  seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                 /*num_cards_played=*/0, hands, floor);

  std::vector<Action> legal = state->LegalActions();
  // Player 0 can build a house of 10 using 4C (+ 6C on floor) since Player 0
  // holds TD/TS in hand.
  SPIEL_CHECK_TRUE(ContainsAction(
      legal, MakePlayAction(CardFromString("4C"), Intent::kBuild10)));
  // Player 0 can also throw TD or TS (value 10 == bid_value).
  SPIEL_CHECK_TRUE(ContainsAction(
      legal, MakePlayAction(CardFromString("TD"), Intent::kThrow)));
  // Player 0 CANNOT throw 7H or build a house of 9 on Turn 0 because bid is 10.
  SPIEL_CHECK_FALSE(ContainsAction(
      legal, MakePlayAction(CardFromString("7H"), Intent::kThrow)));
  SPIEL_CHECK_FALSE(ContainsAction(
      legal, MakePlayAction(CardFromString("7H"), Intent::kBuild9)));
}

void HouseBuildingRaisingAndCementingTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* seep_state = static_cast<SeepState*>(state.get());

  // Floor has loose 5C and loose 6D.
  // Player 0 holds {4D, 9S} -> builds uncemented (kacha) house of 9 with 4D+5C.
  // Player 1 holds {9H, 9C, 3S, QD} -> can either raise the uncemented 9 house
  // to 12 using 3S, OR cement the 9 house using 9H (or 3S+6D).
  std::array<std::vector<int>, kNumPlayers> hands;
  hands[0] = {CardFromString("4D"), CardFromString("9S")};
  hands[1] = {CardFromString("3S"), CardFromString("9C"), CardFromString("9H"),
              CardFromString("QD")};
  hands[2] = {CardFromString("AC")};
  hands[3] = {CardFromString("AD")};
  std::vector<House> floor = {
      House{5, false, {CardFromString("5C")}},
      House{6, false, {CardFromString("6D")}},
  };
  seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                 /*num_cards_played=*/0, hands, floor);

  // Player 0 builds house of 9 with 4D.
  Action build9 = MakePlayAction(CardFromString("4D"), Intent::kBuild9);
  SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), build9));
  state->ApplyAction(build9);

  // Verify floor now has an uncemented house of 9 ({4D, 5C}) and loose 6D.
  SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 2);
  bool found_house9 = false;
  for (const House& h : seep_state->FloorPiles()) {
    if (h.value == 9) {
      found_house9 = true;
      SPIEL_CHECK_FALSE(h.is_cemented);
      SPIEL_CHECK_EQ(h.cards.size(), 2);
    }
  }
  SPIEL_CHECK_TRUE(found_house9);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  // Branch A: Clone state and test Player 1 raising the uncemented house of 9
  // to 12 using 3S (since Player 1 holds QD = 12).
  {
    std::unique_ptr<State> clone = state->Clone();
    auto* clone_seep = static_cast<SeepState*>(clone.get());
    Action raise_to_12 = MakePlayAction(CardFromString("3S"), Intent::kBuild12);
    SPIEL_CHECK_TRUE(ContainsAction(clone->LegalActions(), raise_to_12));
    clone->ApplyAction(raise_to_12);

    bool found_house12 = false;
    for (const House& h : clone_seep->FloorPiles()) {
      if (h.value == 12) {
        found_house12 = true;
        SPIEL_CHECK_FALSE(h.is_cemented);
        SPIEL_CHECK_EQ(h.cards.size(), 3);  // 4D, 5C, 3S
      }
    }
    SPIEL_CHECK_TRUE(found_house12);
  }

  // Branch B: Player 1 cements the house of 9 by playing 9C (while holding 9H).
  Action cement9 = MakePlayAction(CardFromString("9C"), Intent::kBuild9);
  SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), cement9));
  state->ApplyAction(cement9);

  // Verify the house of 9 is now cemented (is_cemented == true).
  for (const House& h : seep_state->FloorPiles()) {
    if (h.value == 9) {
      SPIEL_CHECK_TRUE(h.is_cemented);
      SPIEL_CHECK_EQ(h.cards.size(), 3);  // 4D, 5C, 9C
    }
  }

  // Verify a cemented house of 9 CANNOT be raised to 12 even if a player holds
  // 3 and Queen (12).
  hands[2] = {CardFromString("3C"), CardFromString("QH")};
  seep_state->SetCustomPlayState(
      /*current_player=*/2, /*bid_value=*/9,
      /*num_cards_played=*/2,
      {seep_state->Hand(0), seep_state->Hand(1), hands[2], seep_state->Hand(3)},
      seep_state->FloorPiles());
  Action illegal_raise = MakePlayAction(CardFromString("3C"), Intent::kBuild12);
  SPIEL_CHECK_FALSE(ContainsAction(state->LegalActions(), illegal_raise));

  // Verify throwing a card of value 9 is illegal when a house of 9 is on the
  // floor. Partner P2 with a single 9D CAN re-cement P0's 9 house (9D:Build9),
  // whereas P0 with only a single 9S in hand CANNOT play 9S:Build9.
  hands[2] = {CardFromString("9D")};
  seep_state->SetCustomPlayState(
      /*current_player=*/2, /*bid_value=*/9,
      /*num_cards_played=*/2,
      {seep_state->Hand(0), seep_state->Hand(1), hands[2], seep_state->Hand(3)},
      seep_state->FloorPiles());
  SPIEL_CHECK_FALSE(
      ContainsAction(state->LegalActions(),
                     MakePlayAction(CardFromString("9D"), Intent::kThrow)));
  SPIEL_CHECK_TRUE(
      ContainsAction(state->LegalActions(),
                     MakePlayAction(CardFromString("9D"), Intent::kBuild9)));
  seep_state->SetCustomPlayState(
      /*current_player=*/0, /*bid_value=*/9,
      /*num_cards_played=*/2,
      {seep_state->Hand(0), seep_state->Hand(1), hands[2], seep_state->Hand(3)},
      seep_state->FloorPiles());
  SPIEL_CHECK_FALSE(
      ContainsAction(state->LegalActions(),
                     MakePlayAction(CardFromString("9S"), Intent::kBuild9)));

  // Verify cementing an uncemented house of 9 via card + subset (3S + 6D -> 9)
  // and single-move multi-group cementing (3S + 6H and 4C + 5D -> 9P).
  hands[0] = {CardFromString("3S"), CardFromString("9H")};
  std::vector<House> multi_floor = {
      House{4, false, {CardFromString("4C")}},
      House{5, false, {CardFromString("5D")}},
      House{6, false, {CardFromString("6H")}},
  };
  seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                 /*num_cards_played=*/4, hands, multi_floor);
  Action multi_cement9 = MakePlayAction(CardFromString("3S"), Intent::kBuild9);
  SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), multi_cement9));
  state->ApplyAction(multi_cement9);
  SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 1);
  SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].value, 9);
  SPIEL_CHECK_TRUE(seep_state->FloorPiles()[0].is_cemented);
  SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].cards.size(), 4);
}

void CaptureTieBreakingAndDuplicateFloorTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* seep_state = static_cast<SeepState*>(state.get());

  // Floor has overlapping subsets summing to 9: {3S, 6S} (9 pts) vs {3S, 6D}
  // (3 pts), plus two loose 10s (TS = 10 pts, TD = 6 pts) to verify
  // ObservationTensor section 6 with duplicate floor values.
  std::array<std::vector<int>, kNumPlayers> hands;
  hands[0] = {CardFromString("9C")};
  std::vector<House> floor = {
      House{3, false, {CardFromString("3S")}},
      House{6, false, {CardFromString("6D")}},
      House{6, false, {CardFromString("6S")}},
      House{10, false, {CardFromString("TD")}},
      House{10, false, {CardFromString("TS")}},
  };
  seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                 /*num_cards_played=*/4, hands, floor);

  // Check ObservationTensor section 6 for value 10 (index 1 in 9..13):
  // offset = 4 + 6 + 5 + 52 + 52 = 119; h_idx = 1 -> 119 + 3 = 122.
  std::vector<float> tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(tensor));
  SPIEL_CHECK_EQ(tensor[122], 1.0f);         // exists
  SPIEL_CHECK_EQ(tensor[123], 0.0f);         // is_cemented == false
  SPIEL_CHECK_FLOAT_EQ(tensor[124], 0.16f);  // (10 + 6) / 100

  // Apply 9C:Capture and verify {3S, 6S} (9 pts) is captured over {3S, 6D}.
  state->ApplyAction(MakePlayAction(CardFromString("9C"), Intent::kCapture));
  SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0], 9);
  SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 3);  // 6D, TD, TS remain
}

void CompulsoryCaptureAndKachhaVsPakkaHouseRulesTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");

  // 1. Loose 5 and 3 on floor, player plays 8 -> CANNOT throw 8 loose; MUST
  // capture 5 + 3.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[0] = {CardFromString("8S")};
    std::vector<House> floor = {
        House{3, false, {CardFromString("3D")}},
        House{5, false, {CardFromString("5S")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/4, hands, floor);
    std::vector<Action> legal = state->LegalActions();
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("8S"), Intent::kThrow)));
    Action capture8 = MakePlayAction(CardFromString("8S"), Intent::kCapture);
    SPIEL_CHECK_TRUE(ContainsAction(legal, capture8));
    state->ApplyAction(capture8);
    SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0], 13);  // 5S + 8S
  }

  // 2. Kachha (uncemented) house of 11 + loose 2 on floor, player plays K (13)
  // -> CANNOT throw K loose; MUST pick up kachha 11 + 2.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[0] = {CardFromString("KS")};
    std::vector<House> floor = {
        House{11, false, {CardFromString("5C"), CardFromString("6S")}},
        House{2, false, {CardFromString("2S")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/6, hands, floor);
    std::vector<Action> legal = state->LegalActions();
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("KS"), Intent::kThrow)));
    Action capture_k = MakePlayAction(CardFromString("KS"), Intent::kCapture);
    SPIEL_CHECK_TRUE(ContainsAction(legal, capture_k));
    state->ApplyAction(capture_k);
    SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0], 21);  // 6S + 2S + KS
    SPIEL_CHECK_EQ(seep_state->TeamSeeps()[0], 1);
  }

  // 3. Pakka (cemented) house of J (11) + loose 2 on floor, player plays K (13)
  // -> K CANNOT pick up pakka 11 + 2 (nor build 13 with them); K must be
  // thrown loose.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[0] = {CardFromString("KS"), CardFromString("KH")};
    std::vector<House> floor = {
        House{
            11,
            true,
            {CardFromString("5C"), CardFromString("6S"), CardFromString("JC")}},
        House{2, false, {CardFromString("2S")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/6, hands, floor);
    std::vector<Action> legal = state->LegalActions();
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("KS"), Intent::kCapture)));
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("KS"), Intent::kBuild13)));
    Action throw_k = MakePlayAction(CardFromString("KS"), Intent::kThrow);
    SPIEL_CHECK_TRUE(ContainsAction(legal, throw_k));
    state->ApplyAction(throw_k);
    SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 3);  // 2S, 11P, KS
  }

  // 4. Cannot break a kachha house using a loose floor card (kachha 9 + loose 2
  // + hand Ace -> 12 is illegal, whereas kachha 9 + hand 3 -> 12 is legal), and
  // cannot use a single 9 in hand to build 12 with loose 3 while a 9 house is
  // on the floor.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[0] = {CardFromString("AC"), CardFromString("3S"),
                CardFromString("9C"), CardFromString("QD")};
    std::vector<House> floor = {
        House{9, false, {CardFromString("4C"), CardFromString("5D")}},
        House{2, false, {CardFromString("2H")}},
        House{3, false, {CardFromString("3H")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/6, hands, floor);
    std::vector<Action> legal = state->LegalActions();
    // AC (1) + kachha 9 + loose 2 -> 12 is illegal.
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("AC"), Intent::kBuild12)));
    // 9C (only 9 in hand) + loose 3 -> 12 is illegal while 9 house is on floor.
    SPIEL_CHECK_FALSE(ContainsAction(
        legal, MakePlayAction(CardFromString("9C"), Intent::kBuild12)));
    // 3S (3) + kachha 9 -> 12 is legal!
    SPIEL_CHECK_TRUE(ContainsAction(
        legal, MakePlayAction(CardFromString("3S"), Intent::kBuild12)));
  }

  // 5. Two Q's in hand + kachha 10 and loose 2 on floor:
  //    - With two Q's (QC, QS), player can play QC:Build12 to combine kachha 10
  //      + 2 + QC into a pakka 12P house while retaining QS to pick it up
  //      later (or play QC:Capture; QC:Throw is illegal).
  //    - With only ONE Q (QC), QC:Build12 is illegal (no second Q to retain)
  //      and QC:Throw is illegal; only QC:Capture is legal.
  //    - If the 10 is a pakka 10P + loose 2, neither QC:Build12 nor QC:Capture
  //      is legal even with two Q's; only QC:Throw is legal.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::vector<House> kachha10_and_2 = {
        House{10, false, {CardFromString("4C"), CardFromString("6S")}},
        House{2, false, {CardFromString("2D")}},
    };

    // Case 5a: Only one Q in hand -> cannot Build12, cannot Throw, must
    // Capture.
    std::array<std::vector<int>, kNumPlayers> one_q_hands;
    one_q_hands[0] = {CardFromString("QC")};
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/4, one_q_hands,
                                   kachha10_and_2);
    std::vector<Action> legal_one_q = state->LegalActions();
    SPIEL_CHECK_FALSE(ContainsAction(
        legal_one_q, MakePlayAction(CardFromString("QC"), Intent::kBuild12)));
    SPIEL_CHECK_FALSE(ContainsAction(
        legal_one_q, MakePlayAction(CardFromString("QC"), Intent::kThrow)));
    SPIEL_CHECK_TRUE(ContainsAction(
        legal_one_q, MakePlayAction(CardFromString("QC"), Intent::kCapture)));

    // Case 5b: Two Q's in hand (QC, QS) -> QC:Build12 creates pakka 12P house
    // of all 4 cards (4C, 6S, 2D, QC), and QS remains in hand to pick it up!
    std::array<std::vector<int>, kNumPlayers> two_q_hands;
    two_q_hands[0] = {CardFromString("QC"), CardFromString("QS")};
    two_q_hands[1] = {CardFromString("AC")};
    two_q_hands[2] = {CardFromString("AD")};
    two_q_hands[3] = {CardFromString("AH")};
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/4, two_q_hands,
                                   kachha10_and_2);
    std::vector<Action> legal_two_q = state->LegalActions();
    Action cement_12 = MakePlayAction(CardFromString("QC"), Intent::kBuild12);
    SPIEL_CHECK_TRUE(ContainsAction(legal_two_q, cement_12));
    SPIEL_CHECK_FALSE(ContainsAction(
        legal_two_q, MakePlayAction(CardFromString("QC"), Intent::kThrow)));
    state->ApplyAction(cement_12);

    SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 1);
    SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].value, 12);
    SPIEL_CHECK_TRUE(seep_state->FloorPiles()[0].is_cemented);
    SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].cards.size(), 4);
    // Player 0 still holds QS in hand to capture the pakka 12P house later.
    SPIEL_CHECK_EQ(seep_state->Hand(0), std::vector<int>{CardFromString("QS")});

    // Advance P1, P2, P3 throwing Aces, then P0 captures 12P with QS.
    state->ApplyAction(MakePlayAction(CardFromString("AC"), Intent::kThrow));
    state->ApplyAction(MakePlayAction(CardFromString("AD"), Intent::kCapture));
    state->ApplyAction(MakePlayAction(CardFromString("AH"), Intent::kThrow));
    Action capture_pakka_12 =
        MakePlayAction(CardFromString("QS"), Intent::kCapture);
    SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), capture_pakka_12));
    state->ApplyAction(capture_pakka_12);
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0],
                   20);  // 6S + QS (12) + AC+AD

    // Case 5c: Pakka 10P + loose 2 on floor with two Q's in hand -> cannot
    // Build12 or Capture; can only Throw QC.
    std::vector<House> pakka10_and_2 = {
        House{
            10,
            true,
            {CardFromString("4C"), CardFromString("6S"), CardFromString("TC")}},
        House{2, false, {CardFromString("2D")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/4, two_q_hands,
                                   pakka10_and_2);
    std::vector<Action> legal_pakka10 = state->LegalActions();
    SPIEL_CHECK_FALSE(ContainsAction(
        legal_pakka10, MakePlayAction(CardFromString("QC"), Intent::kBuild12)));
    SPIEL_CHECK_FALSE(ContainsAction(
        legal_pakka10, MakePlayAction(CardFromString("QC"), Intent::kCapture)));
    SPIEL_CHECK_TRUE(ContainsAction(
        legal_pakka10, MakePlayAction(CardFromString("QC"), Intent::kThrow)));
  }

  // 6. Automatic cementing when raising a kachha house or building a house
  //    whose target value matches another loose card or sum on the floor.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    // Floor has kachha 9 (4C+5D) plus loose 5H and 7S (which sum to 12).
    // Player 0 holds 3S and QD (only ONE Q needed since 3S is played).
    // Playing 3S:Build12 raises kachha 9 -> 12 AND automatically pulls in
    // 5H + 7S to cement into a pakka 12P house of 5 cards!
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[0] = {CardFromString("3S"), CardFromString("QD")};
    std::vector<House> floor = {
        House{9, false, {CardFromString("4C"), CardFromString("5D")}},
        House{5, false, {CardFromString("5H")}},
        House{7, false, {CardFromString("7S")}},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                   /*num_cards_played=*/4, hands, floor);
    Action raise_and_cement12 =
        MakePlayAction(CardFromString("3S"), Intent::kBuild12);
    SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), raise_and_cement12));
    state->ApplyAction(raise_and_cement12);
    SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 1);
    SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].value, 12);
    SPIEL_CHECK_TRUE(seep_state->FloorPiles()[0].is_cemented);
    SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].cards.size(), 5);
  }

  // 7. House-owner card binding lifecycle:
  //    - P0 builds a 12 house (10 + 2 = 12). With 1 Q in hand + K + loose A
  //      on floor, P0 cannot use Q + A -> 13 (must keep Q for own 12 house),
  //      even after an opponent cements that 12 house into 12P!
  //    - With 2 Q's in hand, P0 CAN use one Q + A -> 13 while keeping the
  //      second Q for the 12 house.
  //    - Once an opponent or partner picks up P0's 12 house and later builds
  //      a new 5 + 7 = 12 house (kachha or pakka), P0 is NOT bound to that
  //      new 12 house and CAN use a single Q either to pick it up or with
  //      loose A on floor to build 13!
  //    - If P0 built a kachha 10 house and loose 3 is on floor, playing K
  //      (13) from hand must pick up P0's own kachha 10 + 3 (cannot throw K).
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());

    // P0 builds 12[2D, TC] (owners_mask = 1 << 0). Floor also has loose AC.
    std::vector<House> p0_kachha12_floor = {
        House{1, false, {CardFromString("AC")}},
        House{12, false, {CardFromString("2D"), CardFromString("TC")}, 1 << 0},
    };
    // (a) Single Q in hand (QD) + KS: QD:Build13 is forbidden; QD:Capture is
    //     legal.
    std::array<std::vector<int>, kNumPlayers> hands_1q;
    hands_1q[0] = {CardFromString("QD"), CardFromString("KS")};
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                   /*num_cards_played=*/4, hands_1q,
                                   p0_kachha12_floor);
    SPIEL_CHECK_FALSE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("QD"), Intent::kBuild13)));
    SPIEL_CHECK_TRUE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("QD"), Intent::kCapture)));

    // (b) Two Q's in hand (QD, QS) + KS: QD:Build13 IS legal!
    std::array<std::vector<int>, kNumPlayers> hands_2q;
    hands_2q[0] = {CardFromString("QD"), CardFromString("QS"),
                   CardFromString("KS")};
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                   /*num_cards_played=*/4, hands_2q,
                                   p0_kachha12_floor);
    SPIEL_CHECK_TRUE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("QD"), Intent::kBuild13)));

    // (c) Opponent P1 cements P0's 12 house with QC:Build12 -> owners_mask
    //     becomes (1 << 0) | (1 << 1). P0 with 1 Q is STILL bound to 12P!
    hands_1q[1] = {CardFromString("QC"), CardFromString("QH")};
    seep_state->SetCustomPlayState(/*current_player=*/1, /*bid_value=*/10,
                                   /*num_cards_played=*/5, hands_1q,
                                   p0_kachha12_floor);
    state->ApplyAction(MakePlayAction(CardFromString("QC"), Intent::kBuild12));
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                   /*num_cards_played=*/8, hands_1q,
                                   seep_state->FloorPiles());
    SPIEL_CHECK_FALSE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("QD"), Intent::kBuild13)));
    SPIEL_CHECK_TRUE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("QD"), Intent::kCapture)));

    // (d) After P0's 12 house is picked up, Partner P2 or Opponent P1 builds
    //     a new 5 + 7 = 12 house (kachha or pakka, owners_mask = 1 << 2).
    //     Now P0 with only 1 Q (QD) CAN both Capture 12 AND Build13 with AC!
    for (bool is_pakka : {false, true}) {
      std::vector<House> other_12_floor = {
          House{1, false, {CardFromString("AC")}},
          House{12,
                is_pakka,
                {CardFromString("5C"), CardFromString("7D")},
                1 << 2},
      };
      seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                     /*num_cards_played=*/10, hands_1q,
                                     other_12_floor);
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("QD"), Intent::kCapture)));
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("QD"), Intent::kBuild13)));
    }

    // (e) P0 owns a kachha 10 house + loose 3 on floor; if P0 plays K (13),
    //     P0 must pick up 10 + 3 (cannot throw K loose).
    std::array<std::vector<int>, kNumPlayers> hands_k;
    hands_k[0] = {CardFromString("TC"), CardFromString("KS")};
    std::vector<House> p0_kachha10_and_3 = {
        House{3, false, {CardFromString("3D")}},
        House{10, false, {CardFromString("4C"), CardFromString("6S")}, 1 << 0},
    };
    seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                   /*num_cards_played=*/8, hands_k,
                                   p0_kachha10_and_3);
    SPIEL_CHECK_FALSE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("KS"), Intent::kThrow)));
    SPIEL_CHECK_TRUE(
        ContainsAction(state->LegalActions(),
                       MakePlayAction(CardFromString("KS"), Intent::kCapture)));

    // (f) Partner Exception on kachha or pakka 12 house owned by P0:
    //     - Partner P2 can play 7 + 5 -> 12 onto P0's 12 house (kachha or
    //       pakka) even with NO Q in P2's hand!
    //     - Partner P2 with a single Q (QC) can EITHER Capture P0's 12 house
    //       OR Build12 (cement/re-cement it by placing QC on the 12 pile)
    //       without needing a second Q!
    //     - Opponent P1 cannot do either without holding a Q in hand after
    //       playing.
    for (bool is_pakka : {false, true}) {
      std::vector<House> p0_house_and_5 = {
          House{5, false, {CardFromString("5C")}},
          House{12,
                is_pakka,
                {CardFromString("2D"), CardFromString("TC")},
                1 << 0},
      };
      std::array<std::vector<int>, kNumPlayers> partner_hands;
      partner_hands[0] = {CardFromString("QD")};
      partner_hands[1] = {CardFromString("7H"), CardFromString("QC")};
      partner_hands[2] = {CardFromString("7D")};  // Partner P2 has NO Q!

      // Partner P2 can play 7D:Build12 (7D + 5C -> 12P) without holding a Q.
      seep_state->SetCustomPlayState(/*current_player=*/2, /*bid_value=*/10,
                                     /*num_cards_played=*/6, partner_hands,
                                     p0_house_and_5);
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("7D"), Intent::kBuild12)));

      // Partner P2 with a single Q (QC) can both Capture AND Build12.
      partner_hands[2] = {CardFromString("QC")};
      seep_state->SetCustomPlayState(/*current_player=*/2, /*bid_value=*/10,
                                     /*num_cards_played=*/6, partner_hands,
                                     p0_house_and_5);
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("QC"), Intent::kCapture)));
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("QC"), Intent::kBuild12)));

      // Opponent P1 with a single Q (QC) cannot play QC:Build12 (needs 2 Qs).
      seep_state->SetCustomPlayState(/*current_player=*/1, /*bid_value=*/10,
                                     /*num_cards_played=*/5, partner_hands,
                                     p0_house_and_5);
      SPIEL_CHECK_FALSE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("QC"), Intent::kBuild12)));
    }

    // (g) Cementing partner's 13 house with KS:Build13 when a 10 house and
    //     loose 3S are also on the floor:
    //     - If 10 house is kachha, KS:Build13 MUST merge 13[4C,9D] +
    //       10[2H,8C] + 3S + KS into a 6-card 13P house (no choice to leave
    //       10 + 3 behind).
    //     - If 10 house is pakka (10P), KS:Build13 CANNOT merge 10P + 3S;
    //       only 13[4C,9D] + KS becomes 13P, while 3S and 10P remain on
    //       floor.
    {
      std::array<std::vector<int>, kNumPlayers> k_hands;
      k_hands[0] = {CardFromString("KC"), CardFromString("KS")};

      // Kachha 10 case: mandatory merge of 13 + (10 + 3) + KS.
      std::vector<House> kachha10_floor = {
          House{3, false, {CardFromString("3S")}},
          House{
              10, false, {CardFromString("2H"), CardFromString("8C")}, 1 << 1},
          House{
              13, false, {CardFromString("4C"), CardFromString("9D")}, 1 << 2},
      };
      seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                     /*num_cards_played=*/16, k_hands,
                                     kachha10_floor);
      SPIEL_CHECK_TRUE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("KS"), Intent::kBuild13, 0)));
      SPIEL_CHECK_FALSE(ContainsAction(
          state->LegalActions(),
          MakePlayAction(CardFromString("KS"), Intent::kBuild13, 1)));
      state->ApplyAction(
          MakePlayAction(CardFromString("KS"), Intent::kBuild13, 0));
      SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 1);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].value, 13);
      SPIEL_CHECK_TRUE(seep_state->FloorPiles()[0].is_cemented);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].cards.size(), 6);

      // Pakka 10P case: 10P and 3S cannot be merged; only 13[4C,9D] + KS
      // becomes 13P.
      std::vector<House> pakka10_floor = {
          House{3, false, {CardFromString("3S")}},
          House{10,
                true,
                {CardFromString("2H"), CardFromString("8C"),
                 CardFromString("TC")},
                1 << 1},
          House{
              13, false, {CardFromString("4C"), CardFromString("9D")}, 1 << 2},
      };
      seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/10,
                                     /*num_cards_played=*/16, k_hands,
                                     pakka10_floor);
      state->ApplyAction(
          MakePlayAction(CardFromString("KS"), Intent::kBuild13, 0));
      SPIEL_CHECK_EQ(seep_state->FloorPiles().size(), 3);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[0].value, 3);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[1].value, 10);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[2].value, 13);
      SPIEL_CHECK_TRUE(seep_state->FloorPiles()[2].is_cemented);
      SPIEL_CHECK_EQ(seep_state->FloorPiles()[2].cards.size(), 3);
    }
  }
}

void CaptureAndSeepBonusTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* seep_state = static_cast<SeepState*>(state.get());

  // Floor has a house of 9 ({4S, 5S}) and two loose cards {3S, 6S} that also
  // sum to 9. Player 0 plays 9S to capture ALL of them, clearing the floor for
  // a Seep (+50 bonus)!
  std::array<std::vector<int>, kNumPlayers> hands;
  hands[0] = {CardFromString("9S"), CardFromString("AC")};
  hands[1] = {CardFromString("AD")};
  hands[2] = {CardFromString("AH")};
  hands[3] = {CardFromString("2C")};
  std::vector<House> floor = {
      House{9, false, {CardFromString("4S"), CardFromString("5S")}},
      House{3, false, {CardFromString("3S")}},
      House{6, false, {CardFromString("6S")}},
  };
  seep_state->SetCustomPlayState(/*current_player=*/0, /*bid_value=*/9,
                                 /*num_cards_played=*/10, hands, floor);

  Action capture_all = MakePlayAction(CardFromString("9S"), Intent::kCapture);
  SPIEL_CHECK_TRUE(ContainsAction(state->LegalActions(), capture_all));
  state->ApplyAction(capture_all);

  // Floor is now empty -> Team 0 gets 1 Seep (+50) and captures 4S+5S+3S+6S+9S
  // = 4+5+3+6+9 = 27 card points, total = 77 points.
  SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
  SPIEL_CHECK_EQ(seep_state->TeamSeeps()[0], 1);
  SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0], 27);
  SPIEL_CHECK_EQ(seep_state->TeamTotalScores()[0], 77);
  SPIEL_CHECK_EQ(seep_state->LastCapturePlayer(), 0);
}

void LastCardCaptureNoSeepAndFloorSweepTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");

  // Case 1: Clearing the floor on the 48th (final) card does NOT award +50
  // Seep bonus.
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[3] = {CardFromString("KS")};
    std::vector<House> floor = {House{13, false, {CardFromString("KH")}}};
    seep_state->SetCustomPlayState(/*current_player=*/3, /*bid_value=*/9,
                                   /*num_cards_played=*/kTotalPlayTurns - 1,
                                   hands, floor);

    Action final_capture =
        MakePlayAction(CardFromString("KS"), Intent::kCapture);
    state->ApplyAction(final_capture);

    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_EQ(seep_state->TeamSeeps()[1], 0);  // No Seep on final turn!
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[1], 13);  // KS = 13 pts
    SPIEL_CHECK_EQ(state->Returns(),
                   (std::vector<double>{-13.0, 13.0, -13.0, 13.0}));
  }

  // Case 2: Uncaptured floor cards at the end of the 48th turn go to the
  // partnership of LastCapturePlayer().
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto* seep_state = static_cast<SeepState*>(state.get());
    std::array<std::vector<int>, kNumPlayers> hands;
    hands[3] = {CardFromString("2C")};  // 0 pts
    std::vector<House> floor = {
        House{10, false, {CardFromString("TD")}},  // 6 pts on floor
    };
    // Player 2 (Team 0) made the last capture earlier.
    seep_state->SetCustomPlayState(
        /*current_player=*/3, /*bid_value=*/9,
        /*num_cards_played=*/kTotalPlayTurns - 1, hands, floor,
        /*team_captured_cards=*/{}, /*team_seeps=*/{0, 0},
        /*last_capture_player=*/2);

    Action final_throw = MakePlayAction(CardFromString("2C"), Intent::kThrow);
    state->ApplyAction(final_throw);

    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_TRUE(seep_state->FloorPiles().empty());
    // Team 0 sweeps TD (6 points) because Player 2 made the last capture.
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[0], 6);
    SPIEL_CHECK_EQ(seep_state->TeamCardPoints()[1], 0);
    SPIEL_CHECK_EQ(state->Returns(),
                   (std::vector<double>{6.0, -6.0, 6.0, -6.0}));
  }
}

void ObservationStructAndSerializationTests() {
  std::shared_ptr<const Game> game = LoadGame("seep");
  std::unique_ptr<State> state = game->NewInitialState();

  // Advance 10 moves into the game.
  for (int i = 0; i < 10; ++i) {
    std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_FALSE(actions.empty());
    state->ApplyAction(actions[0]);
  }

  // Serialization round-trip check.
  std::unique_ptr<State> deserialized =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized->ToString());

  // Observation tensor and string checks for all 4 players.
  std::vector<float> tensor(game->ObservationTensorSize());
  for (Player p = 0; p < kNumPlayers; ++p) {
    state->ObservationTensor(p, absl::MakeSpan(tensor));
    SPIEL_CHECK_EQ(tensor[p], 1.0f);  // Observing player one-hot.
    SPIEL_CHECK_FALSE(state->ObservationString(p).empty());
    SPIEL_CHECK_FALSE(state->InformationStateString(p).empty());

    // ObservationStruct must hide other players' private hands.
    std::unique_ptr<ObservationStruct> obs_struct =
        state->ToObservationStruct(p);
    auto* seep_obs = static_cast<SeepObservationStruct*>(obs_struct.get());
    SPIEL_CHECK_EQ(seep_obs->observing_player, p);
    for (Player other = 0; other < kNumPlayers; ++other) {
      if (other != p) {
        SPIEL_CHECK_TRUE(seep_obs->hands[other].empty());
      }
    }
  }
}

void BasicSeepTests() {
  testing::LoadGameTest("seep");
  testing::ChanceOutcomesTest(*LoadGame("seep"));
  testing::RandomSimTest(*LoadGame("seep"), 50);
}

}  // namespace
}  // namespace seep
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  open_spiel::seep::CardAndActionEncodingTests();
  open_spiel::seep::InitialDealAndBiddingTests();
  open_spiel::seep::FirstTurnBidConstraintTests();
  open_spiel::seep::HouseBuildingRaisingAndCementingTests();
  open_spiel::seep::CaptureTieBreakingAndDuplicateFloorTensorTests();
  open_spiel::seep::CompulsoryCaptureAndKachhaVsPakkaHouseRulesTests();
  open_spiel::seep::CaptureAndSeepBonusTests();
  open_spiel::seep::LastCardCaptureNoSeepAndFloorSweepTests();
  open_spiel::seep::ObservationStructAndSerializationTests();
  open_spiel::seep::BasicSeepTests();
}
