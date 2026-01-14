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

#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/games/leduc_poker/repeated_leduc_poker.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/leduc_poker/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace leduc_poker {
namespace {

namespace testing = open_spiel::testing;

void PlaythroughTest() {
  std::cout << "Running playthrough test..." << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("repeated_leduc_poker(num_hands=2,players=2)");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* repeated_state = down_cast<RepeatedLeducPokerState*>(state.get());

  // Hand 1 starts.
  SPIEL_CHECK_EQ(repeated_state->HandNumber(), 0);
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Deal private cards. P0 gets Queen, P1 gets King.
  // In 2-player Leduc, deck is J,Q,K of 2 suits. J=0,1 Q=2,3 K=4,5.
  // P0 gets Q1 (2), P1 gets K1 (4).
  state->ApplyAction(2);  // Deal Q to P0
  state->ApplyAction(4);  // Deal K to P1

  SPIEL_CHECK_EQ(repeated_state->GetLeducState()->private_card(0), 2);
  SPIEL_CHECK_EQ(repeated_state->GetLeducState()->private_card(1), 4);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // Round 1 betting
  // P0 (Q) calls.
  state->ApplyAction(ActionType::kCall);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  // P1 (K) raises.
  state->ApplyAction(ActionType::kRaise);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // P0 (Q) calls.
  state->ApplyAction(ActionType::kCall);

  // Round 2 starts, deal public card.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  // Public card is a Q (card 3, Q2).
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(repeated_state->GetLeducState()->public_card(), 3);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // P0 has a pair of Queens, P1 has a King. P0 should be confident.
  // Round 2 betting
  // P0 (QQ) raises.
  state->ApplyAction(ActionType::kRaise);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  // P1 (K) calls.
  state->ApplyAction(ActionType::kCall);

  // Hand 1 is over.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);  // between_hands state, P0 to act

  // P0 won. Pot was 1+1 ante + 2*1 first round raise + 2*2 second round raise =
  // 1+1+2+4=8. No, the pot calculation is more complex. Let's check returns.
  // P0: 100 - (1 ante + 2 call_r1 + 4 call_r2) + 14 pot = 107. Return 7.
  // P1: 100 - (1 ante + 2 raise_r1 + 4 call_r2) = 93. Return -7.
  const LeducState* leduc_state = repeated_state->GetLeducState();
  SPIEL_CHECK_FLOAT_EQ(leduc_state->Returns()[0], 7);
  SPIEL_CHECK_FLOAT_EQ(leduc_state->Returns()[1], -7);

  // Rewards are issued one player at a time in the between-hands state.
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 7);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[1], 0);

  // "Continue" actions
  state->ApplyAction(kContinueAction);  // P0
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 0);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[1], -7);

  state->ApplyAction(kContinueAction);  // P1

  // Hand 2 starts.
  SPIEL_CHECK_EQ(repeated_state->HandNumber(), 1);
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Deal J, K.
  state->ApplyAction(0);
  state->ApplyAction(4);

  // P0 calls, P1 calls.
  state->ApplyAction(ActionType::kCall);
  state->ApplyAction(ActionType::kCall);

  // Public card is J.
  state->ApplyAction(1);

  // P0 has pair of Jacks. P1 has a king.
  // P0 bets, P1 folds.
  state->ApplyAction(ActionType::kRaise);
  state->ApplyAction(ActionType::kFold);

  // Hand 2 is over, but the game is not.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // Check rewards for hand 2.
  leduc_state = repeated_state->GetLeducState();
  SPIEL_CHECK_FLOAT_EQ(leduc_state->Returns()[0], 1);
  SPIEL_CHECK_FLOAT_EQ(leduc_state->Returns()[1], -1);

  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 1);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[1], 0);

  // "Continue" actions to end the game.
  state->ApplyAction(kContinueAction);  // P0
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 0);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[1], -1);
  state->ApplyAction(kContinueAction);  // P1

  // Game is over now, because it was the last hand.
  SPIEL_CHECK_TRUE(state->IsTerminal());

  // Hand 1: P0 won 7.
  // Hand 2: P0 calls(1), P1 calls(1). Pot=2. P0 raises(4), P1 folds. P0 wins
  // pot. P0 put in 1+4=5, P1 put in 1. Pot was 2+4=6. P0 wins 6. Net gain 1.
  // P1 loses 1.
  // Total returns: P0: 7 + 1 = 8. P1: -7 - 1 = -8.
  auto returns = state->Returns();
  SPIEL_CHECK_FLOAT_EQ(returns[0], 8);
  SPIEL_CHECK_FLOAT_EQ(returns[1], -8);
}

void RepeatedLeducRandomSimTest() {
  testing::RandomSimTest(
      *LoadGame("repeated_leduc_poker", {{"num_hands", GameParameter(20)},
                                         {"players", GameParameter(2)}}),
      10);
  testing::RandomSimTest(
      *LoadGame("repeated_leduc_poker", {{"num_hands", GameParameter(10)},
                                         {"players", GameParameter(3)}}),
      10);
}

}  // namespace
}  // namespace leduc_poker
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::leduc_poker::PlaythroughTest();
  open_spiel::leduc_poker::RepeatedLeducRandomSimTest();
}
