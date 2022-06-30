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

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace coin_game {

void BasicCoinGameTests() {
  testing::LoadGameTest("coin_game");
  testing::RandomSimTest(*LoadGame("coin_game"), 10);
  testing::RandomSimTest(
      *LoadGame("coin_game",
                {
                    {"players", GameParameter(3)},
                    {"rows", GameParameter(7)},
                    {"columns", GameParameter(10)},
                    {"num_extra_coin_colors", GameParameter(2)},
                    {"episode_length", GameParameter(100)},
                    {"num_coins_per_color", GameParameter(2)},
                }),
      10);
}

void GetAllStatesTest() {
  // Getting all states (on a small game) can find corner case bugs.
  const std::shared_ptr<const Game> game =
      LoadGame("coin_game", {{"players", GameParameter(2)},
                             {"rows", GameParameter(2)},
                             {"columns", GameParameter(3)},
                             {"num_extra_coin_colors", GameParameter(0)},
                             {"episode_length", GameParameter(2)},
                             {"num_coins_per_color", GameParameter(2)}});
  auto states = algorithms::GetAllStates(*game,
                                         /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/false);
  SPIEL_CHECK_EQ(states.size(), 4296);
}
}  // namespace coin_game
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::coin_game::BasicCoinGameTests();
  open_spiel::coin_game::GetAllStatesTest();
}
