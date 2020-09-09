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

// helper methods
std::shared_ptr<const TarokGame> NewTarokGame(
    const open_spiel::GameParameters& params) {
  return std::static_pointer_cast<const TarokGame>(LoadGame("tarok", params));
}

std::unique_ptr<TarokState> StateAfterActions(
    const open_spiel::GameParameters& params,
    const std::vector<open_spiel::Action>& actions) {
  auto state = NewTarokGame(params)->NewInitialTarokState();
  for (auto const& action : actions) {
    state->ApplyAction(action);
  }
  return state;
}

bool AllActionsInOtherActions(
    const std::vector<open_spiel::Action>& actions,
    const std::vector<open_spiel::Action>& other_actions) {
  for (auto const& action : actions) {
    if (std::find(other_actions.begin(), other_actions.end(), action) ==
        other_actions.end()) {
      return false;
    }
  }
  return true;
}

open_spiel::Action CardLongNameToAction(const std::string& long_name,
                                        const std::array<Card, 54>& deck) {
  for (int i = 0; i < deck.size(); i++) {
    if (deck.at(i).long_name == long_name) return i;
  }
  open_spiel::SpielFatalError("Invalid long_name!");
  return -1;
}

std::vector<open_spiel::Action> CardLongNamesToActions(
    const std::vector<std::string>& long_names,
    const std::array<Card, 54>& deck) {
  std::vector<open_spiel::Action> actions;
  actions.reserve(long_names.size());
  for (auto const long_name : long_names) {
    actions.push_back(CardLongNameToAction(long_name, deck));
  }
  return actions;
}

// testing
void BasicGameTests() {
  testing::LoadGameTest("tarok");
  testing::ChanceOutcomesTest(*LoadGame("tarok"));
  testing::RandomSimTest(*LoadGame("tarok"), 100);
}

void CardDeckShufflingSeedTest() {
  auto game = NewTarokGame(
      GameParameters({{"rng_seed", open_spiel::GameParameter(0)}}));

  // subsequent shuffles within the same game should be different
  auto state1 = game->NewInitialTarokState();
  state1->ApplyAction(0);
  auto state2 = game->NewInitialTarokState();
  state2->ApplyAction(0);
  SPIEL_CHECK_NE(state1->PlayerCards(0), state2->PlayerCards(0));

  game = NewTarokGame(
      GameParameters({{"rng_seed", open_spiel::GameParameter(0)}}));
  // shuffles should be the same when recreating a game with the same seed
  auto state3 = game->NewInitialTarokState();
  state3->ApplyAction(0);
  auto state4 = game->NewInitialTarokState();
  state4->ApplyAction(0);
  SPIEL_CHECK_EQ(state1->PlayerCards(0), state3->PlayerCards(0));
  SPIEL_CHECK_EQ(state2->PlayerCards(0), state4->PlayerCards(0));
}

}  // namespace tarok
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tarok::BasicGameTests();
  open_spiel::tarok::CardDeckShufflingSeedTest();
}
