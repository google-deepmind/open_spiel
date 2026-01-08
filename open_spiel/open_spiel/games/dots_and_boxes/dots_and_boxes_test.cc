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
//
// Contributed by Wannes Meert, Giuseppe Marra, and Pieter Robberechts
// for the KU Leuven course Machine Learning: Project.

#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace dots_and_boxes {
namespace {

namespace testing = open_spiel::testing;

void BasicDotsAndBoxesTests() {
  std::cout << "Test dots and boxes\n";
  testing::LoadGameTest("dots_and_boxes");
  testing::NoChanceOutcomesTest(*LoadGame("dots_and_boxes"));
  testing::RandomSimTest(*LoadGame("dots_and_boxes"), 100);
}

}  // namespace
}  // namespace dots_and_boxes
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dots_and_boxes::BasicDotsAndBoxesTests();
}
