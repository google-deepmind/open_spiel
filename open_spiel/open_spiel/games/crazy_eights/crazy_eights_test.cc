// Copyright 2023 DeepMind Technologies Limited
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

#include <array>
#include <vector>

#include "open_spiel/games/crazy_eights/crazy_eights.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crazy_eights {
namespace {

void BasicGameTests() {
  testing::LoadGameTest("crazy_eights");
  for (int players = 2; players <= 6; ++players) {
    for (bool b : {false, true}) {
      testing::RandomSimTest(
          *LoadGame("crazy_eights", {{"players", GameParameter(players)},
                                     {"use_special_cards", GameParameter(b)}}),
          5);
    }
  }
}

void SpecialCardTests() {
  std::shared_ptr<const Game> game =
      LoadGame("crazy_eights", {{"players", GameParameter(4)},
                                {"use_special_cards", GameParameter(true)}});

  std::unique_ptr<State> state = game->NewInitialState();
  // 0 is the dealer
  CrazyEightsState* ce_state = static_cast<CrazyEightsState*>(state.get());
  ce_state->ApplyAction(kDecideDealerActionBase);
  // Player0 has (S2)(H8)(DQ)(SK)(SA)
  // Player1 has (C2)(C3)(S8)(HQ)(CA)
  // Player2 has (D2)(C8)(C9)(SQ)(DA)
  // Player3 has (H2)(D8)(CQ)(CK)(HA)
  std::vector<int> dealt_cards = {0,  1,  2,  3,  4,  24, 25, 26, 27, 28,
                                  40, 41, 42, 43, 44, 47, 48, 49, 50, 51};

  for (auto card : dealt_cards) state->ApplyAction(card);

  // The first card is D3
  ce_state->ApplyAction(5);

  // Player 1 plays C3
  ce_state->ApplyAction(4);

  // Player 2 plays C8
  ce_state->ApplyAction(24);

  // Check the current actions are color nomination
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 2);
  std::vector<Action> legal_actions = ce_state->LegalActions();
  SPIEL_CHECK_EQ(static_cast<int>(legal_actions.size()), kNumSuits);

  for (int i = 0; i < kNumSuits; ++i) {
    SPIEL_CHECK_GE(legal_actions[i], kNominateSuitActionBase);
    SPIEL_CHECK_LT(legal_actions[i], kNominateSuitActionBase + kNumSuits);
  }

  // The next suit is H
  ce_state->ApplyAction(kNominateSuitActionBase + 2);

  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 3);
  // Player 3 plays HA
  state->ApplyAction(50);
  // Reverse direction to player 2
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 2);
  // Player 2 plays DA
  state->ApplyAction(49);
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 3);
  // Reverse direction to player 3
  // Player 3 plays D8
  state->ApplyAction(25);
  // Player 3 nominates D
  state->ApplyAction(kNominateSuitActionBase + 1);

  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 0);
  // Player 0 plays DQ
  state->ApplyAction(41);

  // Player 1 is skipped, next is player 2
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 2);

  // Player 2 plays D2!
  ce_state->ApplyAction(1);
  // Player 3 only has two actions: H2 or start drawing
  legal_actions = ce_state->LegalActions();
  SPIEL_CHECK_EQ(static_cast<int>(legal_actions.size()), 2);
  SPIEL_CHECK_EQ(legal_actions[0], 2);
  SPIEL_CHECK_EQ(legal_actions[1], kDraw);
  // Let's stack the twos!
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 0);

  // Keep stacking
  ce_state->ApplyAction(3);
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 1);

  // Keep stacking
  ce_state->ApplyAction(0);
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 2);
  legal_actions = ce_state->LegalActions();
  SPIEL_CHECK_EQ(static_cast<int>(legal_actions.size()), 1);
  // Player 2 has to draw 8 cards

  ce_state->ApplyAction(kDraw);
  std::vector<int> draw_cards = {6, 7, 8, 9, 10, 11, 12, 13};
  for (auto card : draw_cards) ce_state->ApplyAction(card);
  // Then it is player 3's turn
  SPIEL_CHECK_EQ(ce_state->CurrentPlayer(), 3);

  std::array<int, kNumCards> deck = ce_state->GetDealerDeck();
  SPIEL_CHECK_EQ(static_cast<int>(deck.size()), kNumCards);
  std::array<int, kNumCards> deck_exp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0};
  SPIEL_CHECK_EQ(deck, deck_exp);
}

}  // namespace
}  // namespace crazy_eights
}  // namespace open_spiel

int main() {
  open_spiel::crazy_eights::BasicGameTests();
  open_spiel::crazy_eights::SpecialCardTests();
}
