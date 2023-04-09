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
namespace schnapsen {
namespace {

namespace testing = open_spiel::testing;

void BasicTicTacToeTests() {
  testing::LoadGameTest("schnapsen");
  testing::NoChanceOutcomesTest(*LoadGame("schnapsen"));
  testing::RandomSimTest(*LoadGame("schnapsen"), 200);
}

}  // namespace
}  // namespace schnapsen
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::schnapsen::BasicTicTacToeTests();
}
