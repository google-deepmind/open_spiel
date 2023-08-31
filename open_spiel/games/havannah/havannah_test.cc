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
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace havannah {
namespace {

namespace testing = open_spiel::testing;

void BasicHavannahTests() {
  testing::LoadGameTest("havannah(board_size=4)");
  testing::NoChanceOutcomesTest(*LoadGame("havannah(board_size=4)"));
  testing::RandomSimTest(*LoadGame("havannah"), 10);

  // All the sizes we care about.
  for (int i = 3; i <= 13; i++) {
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("havannah(board_size=", i, ")")), 10);
  }

  // Run many tests hoping swap happens at least once.
  testing::RandomSimTest(*LoadGame("havannah(board_size=3,swap=True)"), 20);

  // Ansi colors!
  testing::RandomSimTest(
      *LoadGame("havannah", {{"board_size", GameParameter(6)},
                             {"ansi_color_output", GameParameter(true)}}),
      3);
  testing::RandomSimTest(
      *LoadGame("havannah(board_size=5,ansi_color_output=True)"), 3);
}

}  // namespace
}  // namespace havannah
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::havannah::BasicHavannahTests(); }
