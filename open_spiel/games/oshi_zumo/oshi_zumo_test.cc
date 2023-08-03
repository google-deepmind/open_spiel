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

#include "open_spiel/games/oshi_zumo.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace oshi_zumo {
namespace {

namespace testing = open_spiel::testing;

void BasicOshiZumoTests() {
  testing::LoadGameTest("oshi_zumo");
  testing::NoChanceOutcomesTest(*LoadGame("oshi_zumo"));
  testing::RandomSimTest(*LoadGame("oshi_zumo"), 100);
}

void CountStates() {
  std::shared_ptr<const Game> game =
      LoadGame("oshi_zumo", {{"horizon", open_spiel::GameParameter(5)},
                             {"coins", open_spiel::GameParameter(5)}});
  auto states = algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/true);
  std::cerr << states.size() << std::endl;
  SPIEL_CHECK_EQ(states.size(), 146);
}

}  // namespace
}  // namespace oshi_zumo
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::oshi_zumo::BasicOshiZumoTests();
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(
      "oshi_zumo", {{"horizon", open_spiel::GameParameter(5)}});
  open_spiel::oshi_zumo::CountStates();
  open_spiel::testing::RandomSimTest(*game, /*num_sims=*/10);
}
