// Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
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

#include "open_spiel/games/battleship.h"

#include <iostream>

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace battleship {
namespace {

namespace testing = open_spiel::testing;

void BasicBattleshipTest() {
  testing::LoadGameTest("battleship");
  testing::NoChanceOutcomesTest(*LoadGame("battleship"));
  testing::RandomSimTestWithUndo(*LoadGame("battleship"), 100);

  for (int num_shots = 1; num_shots <= 3; ++num_shots) {
    const std::map<std::string, GameParameter> params{
        {"board_width", GameParameter(2)},
        {"board_height", GameParameter(2)},
        {"ship_sizes", GameParameter("[1;2]")},
        {"ship_values", GameParameter("[1;2]")},
        {"num_shots", GameParameter(num_shots)},
        {"allow_repeated_shots", GameParameter(false)},
        {"loss_multiplier", GameParameter(2.0)}};

    const auto game = GameRegisterer::CreateByName("battleship", params);
    testing::RandomSimTestWithUndo(*game, 100);
    return;
  }
}
}  // namespace
}  // namespace battleship
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::battleship::BasicBattleshipTest();
}
