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

#include "open_spiel/games/blackjack.h"

#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace blackjack {
namespace {

namespace testing = open_spiel::testing;

void NoBustPlayerWinTest() {
  // Cards are indexed from 0 to 51.
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(0);  // Deal CA to Player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(13);  // Deal DA to Player.

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(11);  // Deal CQ to Dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);  // Deal C5 to Dealer.

  SPIEL_CHECK_TRUE(!state->IsChanceNode());
  state->ApplyAction(0);  // Player hits.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);  // Deal C9 to Player.

  SPIEL_CHECK_TRUE(!state->IsChanceNode());
  state->ApplyAction(1);  // Player stands.

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(2);  // Deal C3 to Dealer.

  SPIEL_CHECK_TRUE(state->IsTerminal());  // Dealer stands.

  // Player wins.
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1);
}

void DealerBustTest() {
  // Cards are indexed from 0 to 51.
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);  // Deal C9 to Player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);  // Deal C5 to Player.

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(10);  // Deal CJ to Dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(2);  // Deal C3 to Dealer.

  SPIEL_CHECK_TRUE(!state->IsChanceNode());
  state->ApplyAction(1);  // Player stands.

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(21);  // Deal D9 to Dealer.

  // Player wins.
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1);
}

void PlayerBustTest() {
  // Cards are indexed from 0 to 51.
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(9);  // Deal C10 to Player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(22);  // Deal D10 to Player.

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);  // Deal C9 to Dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);  // Deal C5 to Dealer.

  SPIEL_CHECK_TRUE(!state->IsChanceNode());
  state->ApplyAction(0);  // Player hits.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(21);  // Deal D9 to Player.

  // Player loses.
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), -1);
}

void BasicBlackjackTests() {
  testing::LoadGameTest("blackjack");
  testing::RandomSimTest(*LoadGame("blackjack"), 100);
  NoBustPlayerWinTest();
  PlayerBustTest();
  DealerBustTest();
}

}  // namespace
}  // namespace blackjack
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::blackjack::BasicBlackjackTests();
}
