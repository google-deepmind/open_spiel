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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace hex {
namespace {

namespace testing = open_spiel::testing;

void TestBoardOrientation() {
  std::shared_ptr<const Game> game = LoadGame(
      "hex", {{"num_cols", GameParameter(3)}, {"num_rows", GameParameter(4)}});
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(4);
  state->ApplyAction(5);
  state->ApplyAction(7);
  state->ApplyAction(8);
  state->ApplyAction(10);
  // Black wins
  std::cout << state << std::endl;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void BasicHexTests() {
  testing::LoadGameTest("hex(num_cols=5,num_rows=5)");
  testing::NoChanceOutcomesTest(*LoadGame("hex(num_cols=5,num_rows=5)"));
  testing::RandomSimTest(*LoadGame("hex(num_cols=5,num_rows=5)"), 100);
  testing::RandomSimTest(*LoadGame("hex"), 5);
  testing::RandomSimTest(*LoadGame("hex(num_cols=2,num_rows=3)"), 10);
  testing::RandomSimTest(*LoadGame("hex(num_cols=2,num_rows=2)"), 10);
}

}  // namespace
}  // namespace hex
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::hex::BasicHexTests();
  open_spiel::hex::TestBoardOrientation();
}
