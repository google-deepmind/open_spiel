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

#include "open_spiel/games/hanabi.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace hanabi {
namespace {

namespace testing = open_spiel::testing;

void BasicHanabiTests() {
  testing::LoadGameTest("hanabi");
  testing::ChanceOutcomesTest(*LoadGame("hanabi"));
  testing::RandomSimTest(*LoadGame("hanabi"), 100);
  for (int players = 3; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("hanabi", {{"players", GameParameter(players)}}), 100);
  }

  auto observer = LoadGame("hanabi")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("hanabi"), observer);
}

}  // namespace
}  // namespace hanabi
}  // namespace open_spiel

int main(int argc, char **argv) { open_spiel::hanabi::BasicHanabiTests(); }
