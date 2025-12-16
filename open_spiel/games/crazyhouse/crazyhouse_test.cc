// Copyright 2025 George Weinberg
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

#include <string>

#include "open_spiel/games/crazyhouse/crazyhouse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crazyhouse {
namespace {

namespace testing = open_spiel::testing;

void BasicTests() {
  testing::LoadGameTest("crazyhouse");
  auto game = open_spiel::LoadGame("crazyhouse");
  auto state = game->NewInitialState();

  std::cout << state->ToString() << std::endl;
  // whoo hoo all pass! 
}


}  // namespace
}  // namespace crazyhouse
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::crazyhouse::BasicTests();
}
