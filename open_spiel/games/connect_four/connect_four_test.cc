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

#include "open_spiel/games/connect_four.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace connect_four {
namespace {

namespace testing = open_spiel::testing;

void BasicConnectFourTests() {
  testing::LoadGameTest("connect_four");
  testing::NoChanceOutcomesTest(*LoadGame("connect_four"));
  testing::RandomSimTest(*LoadGame("connect_four"), 100);
}

void FastLoss() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  auto state = game->NewInitialState();
  state->ApplyAction(3);
  state->ApplyAction(3);
  state->ApplyAction(4);
  state->ApplyAction(4);
  state->ApplyAction(2);
  state->ApplyAction(2);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(1);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
  SPIEL_CHECK_EQ(state->ToString(),
                 ".......\n"
                 ".......\n"
                 ".......\n"
                 ".......\n"
                 "..ooo..\n"
                 ".xxxx..\n");
}

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void CheckFullBoardDraw() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  ConnectFourState state(game,
      "ooxxxoo\n"
      "xxoooxx\n"
      "ooxxxoo\n"
      "xxoooxx\n"
      "ooxxxoo\n"
      "xxoooxx\n");
  SPIEL_CHECK_EQ(state.ToString(),
                 "ooxxxoo\n"
                 "xxoooxx\n"
                 "ooxxxoo\n"
                 "xxoooxx\n"
                 "ooxxxoo\n"
                 "xxoooxx\n");
  SPIEL_CHECK_TRUE(state.IsTerminal());
  SPIEL_CHECK_EQ(state.Returns(), (std::vector<double>{0, 0}));
}

}  // namespace
}  // namespace connect_four
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::connect_four::BasicConnectFourTests();
  open_spiel::connect_four::FastLoss();
  open_spiel::connect_four::BasicSerializationTest();
  open_spiel::connect_four::CheckFullBoardDraw();
}
