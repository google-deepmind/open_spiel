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

#include <memory>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace phantom_ttt {
namespace {

namespace testing = open_spiel::testing;

void ClassicalVsAbrubptTest() {
  std::shared_ptr<const Game> classical_game =
      LoadGame("phantom_ttt(gameversion=classical)");
  std::shared_ptr<const Game> abrupt_game =
      LoadGame("phantom_ttt(gameversion=abrupt)");
  std::unique_ptr<State> classical_state = classical_game->NewInitialState();
  classical_state->ApplyAction(4);
  classical_state->ApplyAction(4);
  SPIEL_CHECK_EQ(classical_state->CurrentPlayer(), 1);
  std::unique_ptr<State> abrupt_state = abrupt_game->NewInitialState();
  abrupt_state->ApplyAction(4);
  abrupt_state->ApplyAction(4);
  SPIEL_CHECK_EQ(abrupt_state->CurrentPlayer(), 0);
}

void BasicPhantomTTTTests() {
  testing::LoadGameTest("phantom_ttt");
  testing::NoChanceOutcomesTest(*LoadGame("phantom_ttt"));
  testing::RandomSimTest(*LoadGame("phantom_ttt"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("phantom_ttt"), 1);
}

}  // namespace
}  // namespace phantom_ttt
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::phantom_ttt::BasicPhantomTTTTests();
  open_spiel::phantom_ttt::ClassicalVsAbrubptTest();
}
