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

#include <string>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tic_tac_toe {
namespace {

namespace testing = open_spiel::testing;

void BasicTicTacToeTests() {
  testing::LoadGameTest("tic_tac_toe");
  testing::NoChanceOutcomesTest(*LoadGame("tic_tac_toe"));
  testing::RandomSimTest(*LoadGame("tic_tac_toe"), 100);
}

void TestStateStruct() {
  auto game = LoadGame("tic_tac_toe");
  auto state = game->NewInitialState();
  TicTacToeState* ttt_state = static_cast<TicTacToeState*>(state.get());
  auto state_struct = ttt_state->ToStruct();
  // Test state/state_struct -> json string.
  SPIEL_CHECK_EQ(state_struct->ToJson(), ttt_state->ToJson());
  std::string state_json =
      "{\"board\":[\".\",\".\",\".\",\".\",\".\",\".\",\".\",\".\",\".\"],"
      "\"current_player\":\"x\"}";
  SPIEL_CHECK_EQ(state_struct->ToJson(), state_json);
  // Test json string -> state_struct.
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 TicTacToeStateStruct(state_json).ToJson());
}

}  // namespace
}  // namespace tic_tac_toe
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tic_tac_toe::BasicTicTacToeTests();
  open_spiel::tic_tac_toe::TestStateStruct();
}
