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

void TestObservationStruct() {
  auto game = LoadGame("tic_tac_toe");
  auto state = game->NewInitialState();
  state->ApplyAction(4);  // Player 0 plays in the center.
  TicTacToeState* ttt_state = static_cast<TicTacToeState*>(state.get());
  auto obs_struct = ttt_state->ToObservationStruct(0);
  std::string obs_json =
      "{\"board\":[\".\",\".\",\".\",\".\",\"x\",\".\",\".\",\".\",\".\"],"
      "\"current_player\":\"o\"}";
  SPIEL_CHECK_EQ(obs_struct->ToJson(), obs_json);
  SPIEL_CHECK_EQ(nlohmann::json::parse(obs_json).dump(),
                 TicTacToeObservationStruct(obs_json).ToJson());
}

void TestActionStruct() {
  auto game = LoadGame("tic_tac_toe");
  auto state = game->NewInitialState();
  auto* ttt_state = static_cast<TicTacToeState*>(state.get());

  // Test ActionToStruct.
  Action action_id = 4;  // Player 0 plays in the center.
  auto action_struct = ttt_state->ActionToStruct(0, action_id);
  std::string action_json = "{\"col\":1,\"row\":1}";
  SPIEL_CHECK_EQ(action_struct->ToJson(), action_json);

  // Test ApplyActionStruct.
  auto state2 = game->NewInitialState();
  state2->ApplyActionStruct(*action_struct);
  SPIEL_CHECK_EQ(state2->ToString(), "...\n.x.\n...");

  // Test JSON parsing.
  SPIEL_CHECK_EQ(nlohmann::json::parse(action_json).dump(),
                 TicTacToeActionStruct(action_json).ToJson());

  // Test StructToAction.
  SPIEL_CHECK_EQ(action_id, ttt_state->StructToAction(*action_struct));
}

}  // namespace
}  // namespace tic_tac_toe
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::tic_tac_toe::BasicTicTacToeTests();
  open_spiel::tic_tac_toe::TestStateStruct();
  open_spiel::tic_tac_toe::TestObservationStruct();
  open_spiel::tic_tac_toe::TestActionStruct();
}
