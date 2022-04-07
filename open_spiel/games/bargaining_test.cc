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

#include "open_spiel/games/bargaining.h"

#include <array>
#include <iostream>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

// This is set to false by default because it complicates tests on github CI.
ABSL_FLAG(bool, enable_instances_file_test, false,
          "Whether to test loading of an instances file.");

namespace open_spiel {
namespace bargaining {
namespace {

constexpr const char* kInstancesFilename =
    "third_party/open_spiel/games/bargaining_instances1000.txt";
constexpr int kFileNumInstances = 1000;

namespace testing = open_spiel::testing;

void BasicBargainingTests() {
  testing::LoadGameTest("bargaining");

  // Game creation and legal actions are fairly heavy, so only run 1 sim.
  testing::RandomSimTest(*LoadGame("bargaining"), 100);
}

void BasicBargainingFromInstancesFileTests() {
  // Game creation and legal actions are fairly heavy, so only run 1 sim.
  std::shared_ptr<const Game> game = LoadGame(
      absl::StrCat("bargaining(instances_file=", kInstancesFilename, ")"));

  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  SPIEL_CHECK_EQ(bargaining_game->AllInstances().size(), kFileNumInstances);

  testing::RandomSimTest(*game, 100);
}

}  // namespace
}  // namespace bargaining
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  open_spiel::bargaining::BasicBargainingTests();
  if (absl::GetFlag(FLAGS_enable_instances_file_test)) {
    open_spiel::bargaining::BasicBargainingFromInstancesFileTests();
  }
}
