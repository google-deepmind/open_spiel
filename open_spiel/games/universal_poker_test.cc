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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace universal_poker {
namespace {

namespace testing = open_spiel::testing;

void BasicUniversalPokerTests() {
  testing::LoadGameTest("universal_poker");
  testing::ChanceOutcomesTest(*LoadGame("universal_poker"));
  testing::RandomSimTest(*LoadGame("universal_poker"), 100);

  //testing::RandomSimBenchmark("leduc_poker", 10000, false);
  //testing::RandomSimBenchmark("universal_poker", 10000, false);

  std::shared_ptr<const Game> gFivePlayers =
          LoadGame({{"name", GameParameter(std::string("universal_poker"))}, {"players", GameParameter(5)}});
  testing::RandomSimTest(*gFivePlayers, 100);
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::universal_poker::BasicUniversalPokerTests(); }
