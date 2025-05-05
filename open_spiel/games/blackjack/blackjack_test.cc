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

#include "open_spiel/games/blackjack/blackjack.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
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

void DealersFirstCardHiddenTest() {
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();

  // Deal cards to player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(9);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(22);

  // Deal cards to dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);

  // Cast state to blackjack state.
  auto blackjack_state = dynamic_cast<BlackjackState *>(state.get());
  SPIEL_CHECK_TRUE(blackjack_state != nullptr);
  std::set<int> visible_cards = blackjack_state->VisibleCards();

  // Dealer's first card.
  SPIEL_CHECK_TRUE(visible_cards.find(8) == visible_cards.end());

  // Remaining cards.
  SPIEL_CHECK_TRUE(visible_cards.find(9) != visible_cards.end());
  SPIEL_CHECK_TRUE(visible_cards.find(22) != visible_cards.end());
  SPIEL_CHECK_TRUE(visible_cards.find(4) != visible_cards.end());
}

void InfoStateStringDoesNotContainDealerFirstCardTest() {
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();

  // Deal cards to player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(9);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(22);

  // Deal cards to dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);

  std::string info_state_string = state->InformationStateString(0);
  SPIEL_CHECK_EQ(info_state_string, "9 22 4");
}

void ResamplingHistoryTest() {
  std::shared_ptr<const Game> game = LoadGame("blackjack");
  std::unique_ptr<State> state = game->NewInitialState();

  // Deal cards to player.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(9);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(22);

  // Deal cards to dealer.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(8);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(4);

  auto b_original = dynamic_cast<BlackjackState *>(state.get());
  std::string original_observation_string = b_original->ObservationString(0);

  std::vector<float> random_seeds = {0.12345, 0.6123, 0.0101};
  // Resample from infostate.
  for (float seed : random_seeds) {
    std::unique_ptr<State> resampled_state =
        state->ResampleFromInfostate(0, [seed]() { return seed; });

    auto b_resampled = dynamic_cast<BlackjackState *>(resampled_state.get());

    // All cards should be the same except for the dealer's first card.
    SPIEL_CHECK_EQ(b_original->cards(0)[0], b_resampled->cards(0)[0]);
    SPIEL_CHECK_EQ(b_original->cards(0)[0], b_resampled->cards(0)[0]);
    SPIEL_CHECK_EQ(b_original->cards(1)[1], b_resampled->cards(1)[1]);

    SPIEL_CHECK_NE(b_original->cards(1)[0], b_resampled->cards(1)[0]);

    // Check that dealer's first card is not visible.
    std::set<int> visible_cards = b_resampled->VisibleCards();
    SPIEL_CHECK_TRUE(visible_cards.find(b_resampled->cards(1)[0]) ==
                     visible_cards.end());

    // Observation strings should be the same.
    SPIEL_CHECK_TRUE(original_observation_string ==
                     b_resampled->ObservationString(0));
  }
}

void BasicBlackjackTests() {
  testing::LoadGameTest("blackjack");
  testing::RandomSimTest(*LoadGame("blackjack"), 100);
  NoBustPlayerWinTest();
  PlayerBustTest();
  DealerBustTest();
  DealersFirstCardHiddenTest();
  InfoStateStringDoesNotContainDealerFirstCardTest();
  ResamplingHistoryTest();
}

}  // namespace
}  // namespace blackjack
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::blackjack::BasicBlackjackTests();
}
