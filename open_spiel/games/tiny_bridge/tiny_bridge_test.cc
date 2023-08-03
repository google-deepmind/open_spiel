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

#include "open_spiel/games/tiny_bridge.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tiny_bridge {
namespace {

namespace testing = open_spiel::testing;

void BasicTinyBridge2pTests() {
  testing::LoadGameTest("tiny_bridge_2p");
  testing::ChanceOutcomesTest(*LoadGame("tiny_bridge_2p"));
  testing::CheckChanceOutcomes(*LoadGame("tiny_bridge_2p"));
  testing::RandomSimTest(*LoadGame("tiny_bridge_2p"), 100);
}

void BasicTinyBridge4pTests() {
  testing::LoadGameTest("tiny_bridge_4p");
  testing::ChanceOutcomesTest(*LoadGame("tiny_bridge_4p"));
  testing::RandomSimTest(*LoadGame("tiny_bridge_4p"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("tiny_bridge_4p"), 1);
}

void CountStates2p() {
  std::shared_ptr<const Game> game = LoadGame("tiny_bridge_2p");
  auto states =
      open_spiel::algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                           /*include_terminals=*/true,
                                           /*include_chance_states=*/false);
  // Chance nodes are not counted.
  // For each of 420 deals:
  //   64 combinations of bids
  //   *2 for initial pass
  //   *2 for terminal pass
  //   -1 for double-counting the auction with a single 'Pass'
  //  => 420 * (64 * 4 - 1) = 107100 states
  SPIEL_CHECK_EQ(states.size(), 107100);
}

}  // namespace
}  // namespace tiny_bridge
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tiny_bridge::BasicTinyBridge2pTests();
  open_spiel::tiny_bridge::BasicTinyBridge4pTests();
  open_spiel::tiny_bridge::CountStates2p();
}
