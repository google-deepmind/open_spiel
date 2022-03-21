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

#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace quoridor {
namespace {

namespace testing = open_spiel::testing;

void BasicQuoridorTests() {
  testing::LoadGameTest("quoridor(board_size=5)");
  testing::NoChanceOutcomesTest(*LoadGame("quoridor()"));
  testing::RandomSimTest(*LoadGame("quoridor"), 10);

  for (int i = 5; i <= 13; i++) {
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("quoridor(board_size=", i, ")")), 5);
  }

  for (int i = 2; i <= 4; i++) {
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("quoridor(board_size=9,players=", i, ")")), 5);
  }

  testing::RandomSimTest(*LoadGame("quoridor(board_size=9,wall_count=5)"), 3);

  // Ansi colors!
  testing::RandomSimTest(
      *LoadGame("quoridor", {{"board_size", GameParameter(9)},
                             {"ansi_color_output", GameParameter(true)}}),
      3);
  testing::RandomSimTest(
      *LoadGame("quoridor", {{"board_size", GameParameter(9)},
                             {"ansi_color_output", GameParameter(true)},
                             {"players", GameParameter(3)}}),
      3);
  testing::RandomSimTest(
      *LoadGame("quoridor(board_size=5,ansi_color_output=True)"), 3);
  testing::RandomSimTest(
      *LoadGame("quoridor(board_size=5,ansi_color_output=True,players=3)"), 3);
}

}  // namespace
}  // namespace quoridor
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::quoridor::BasicQuoridorTests(); }
