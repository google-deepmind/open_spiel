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

#include "open_spiel/games/coordinated_mp.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace coordinated_mp {
namespace {

namespace testing = open_spiel::testing;

void BasicCoordinatedMPTests() {
  testing::LoadGameTest("coordinated_mp");
  testing::ChanceOutcomesTest(*LoadGame("coordinated_mp"));
  testing::RandomSimTest(*LoadGame("coordinated_mp"), 100);
}

void CountStates() {
  std::shared_ptr<const Game> game = LoadGame("coordinated_mp");
  auto states = algorithms::GetAllStates(*game,
                                         /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/true);
  SPIEL_CHECK_EQ(states.size(), 15);
}

}  // namespace
}  // namespace coordinated_mp
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::coordinated_mp::BasicCoordinatedMPTests();
  open_spiel::coordinated_mp::CountStates();
  open_spiel::testing::CheckChanceOutcomes(
      *open_spiel::LoadGame("coordinated_mp"));
  open_spiel::testing::RandomSimTest(*open_spiel::LoadGame("coordinated_mp"),
                                     /*num_sims=*/10);
}
