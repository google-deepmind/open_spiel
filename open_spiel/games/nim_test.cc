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
namespace nim {
namespace {

namespace testing = open_spiel::testing;

void BasicNimTests() {
  testing::LoadGameTest("nim");
  testing::RandomSimTest(*LoadGame("nim"), 100);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes", GameParameter("100;200;300")},
                }),
      10);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes", GameParameter("10000;2000;3000;12414;1515;53252;1;35126")},
                }),
      10);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes", GameParameter("1;2;3;4;5;6;7;8;9;10")},
                    {"is_misere", GameParameter(false)},
                }),
      10);
}

}  // namespace
}  // namespace nim
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::nim::BasicNimTests();
}
