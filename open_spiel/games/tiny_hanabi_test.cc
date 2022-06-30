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

#include "open_spiel/games/tiny_hanabi.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tiny_hanabi {
namespace {

namespace testing = open_spiel::testing;

void BasicTinyHanabiTests() {
  testing::LoadGameTest("tiny_hanabi");
  testing::ChanceOutcomesTest(*LoadGame("tiny_hanabi"));
  testing::CheckChanceOutcomes(*LoadGame("tiny_hanabi"));
  testing::RandomSimTest(*LoadGame("tiny_hanabi"), 100);
}

void CountStates() {
  std::shared_ptr<const Game> game = LoadGame("tiny_hanabi");
  auto states =
      open_spiel::algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                           /*include_terminals=*/true,
                                           /*include_chance_states=*/false);
  // 4 initial deals
  // 13 action states (1 no action, 3 first-player-only, 3*3 both players)
  SPIEL_CHECK_EQ(states.size(), 4 * 13);
}

}  // namespace
}  // namespace tiny_hanabi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tiny_hanabi::BasicTinyHanabiTests();
  open_spiel::tiny_hanabi::CountStates();
}
