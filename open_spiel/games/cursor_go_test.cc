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

#include "open_spiel/games/cursor_go.h"

#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace cursor_go {
namespace {

namespace testing = open_spiel::testing;

constexpr int kBoardSize = 19;
constexpr float kKomi = 7.5;

void BasicGoTests() {
  testing::LoadGameTest("cursor_go");
  testing::NoChanceOutcomesTest(*LoadGame("cursor_go(board_size=5)"));
  testing::RandomSimTest(*LoadGame("cursor_go(board_size=3)"), 100);
}

}  // namespace
}  // namespace cursor_go
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::cursor_go::BasicGoTests(); }
