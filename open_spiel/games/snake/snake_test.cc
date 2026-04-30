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

#include "open_spiel/games/snake/snake.h"

#include <memory>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace snake {
namespace {

namespace testing = open_spiel::testing;

void BasicTwoPlayerTests() {
  testing::LoadGameTest("snake");
  testing::ChanceOutcomesTest(*LoadGame("snake"));
  testing::RandomSimTest(*LoadGame("snake"), 30);
}

void BasicFourPlayerTests() {
  std::shared_ptr<const Game> game =
      LoadGame("snake", {{"players", GameParameter(4)}});
  testing::ChanceOutcomesTest(*game);
  testing::RandomSimTest(*game, 30);
}

// Verify head-to-head collisions kill both snakes.
void HeadToHeadCollisionTest() {
  // 5x5 board: player 0 starts at (2, 1), player 1 at (2, 3). If they both
  // step toward each other, both new heads land on (2, 2).
  std::shared_ptr<const Game> game = LoadGame(
      "snake", {{"rows", GameParameter(5)}, {"columns", GameParameter(5)}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(state->LegalActions()[0]);  // Place initial fruit.
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  state->ApplyActions({kEast, kWest});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Both snakes are dead; neither scored anything.
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 0.0);
  SPIEL_CHECK_EQ(returns[1], 0.0);
}

}  // namespace
}  // namespace snake
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::snake::BasicTwoPlayerTests();
  open_spiel::snake::BasicFourPlayerTests();
  open_spiel::snake::HeadToHeadCollisionTest();
}
