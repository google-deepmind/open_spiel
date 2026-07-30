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
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/games/fox_and_geese/fox_and_geese.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace fox_and_geese {
namespace {

namespace testing = open_spiel::testing;

void BasicFoxAndGeeseTests() {
  testing::LoadGameTest("fox_and_geese");
  testing::NoChanceOutcomesTest(*LoadGame("fox_and_geese"));
  testing::RandomSimTest(*LoadGame("fox_and_geese"), 100);
}

void TestStateStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  FoxAndGeeseState* ttt_state = static_cast<FoxAndGeeseState*>(state.get());
  auto state_struct = ttt_state->ToStruct();
  // Test state/state_struct -> json string.
  SPIEL_CHECK_EQ(state_struct->ToJson(), ttt_state->ToJson());
  std::string state_json =
      "{\"board\":[\".\",\".\",\".\",\".\",\".\",\".\",\".\",\".\",\".\"],"
      "\"current_player\":\"x\"}";
  SPIEL_CHECK_EQ(state_struct->ToJson(), state_json);
  // Test json string -> state_struct.
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 FoxAndGeeseStateStruct(state_json).ToJson());
}

void TestObservationStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  state->ApplyAction(4);  // Player 0 plays in the center.
  FoxAndGeeseState* ttt_state = static_cast<FoxAndGeeseState*>(state.get());
  auto obs_struct = ttt_state->ToObservationStruct(0);
  std::string obs_json =
      "{\"board\":[\".\",\".\",\".\",\".\",\"x\",\".\",\".\",\".\",\".\"],"
      "\"current_player\":\"o\"}";
  SPIEL_CHECK_EQ(obs_struct->ToJson(), obs_json);
  SPIEL_CHECK_EQ(nlohmann::json::parse(obs_json).dump(),
                 FoxAndGeeseObservationStruct(obs_json).ToJson());
}

void TestActionStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  auto* ttt_state = static_cast<FoxAndGeeseState*>(state.get());

  // Test ActionToStruct.
  Action action_id = 4;  // Player 0 plays in the center.
  auto action_struct = ttt_state->ActionToStruct(0, action_id);
  std::string action_json = "{\"col\":1,\"row\":1}";
  SPIEL_CHECK_EQ(action_struct->ToJson(), action_json);

  // Test ApplyActionStruct.
  auto state2 = game->NewInitialState();
  Status status = state2->ApplyActionStruct(*action_struct);
  SPIEL_CHECK_TRUE(status.ok());
  SPIEL_CHECK_EQ(state2->ToString(), "...\n.x.\n...");

  // Test ValidateActionStruct with valid action.
  auto state3 = game->NewInitialState();
  SPIEL_CHECK_TRUE(state3->ValidateActionStruct(*action_struct).ok());

  // Test ValidateActionStruct with invalid action (cell already occupied).
  state3->ApplyAction(4);  // Play in center
  Status validation_status = state3->ValidateActionStruct(*action_struct);
  SPIEL_CHECK_FALSE(validation_status.ok());

  // Test JSON parsing.
  SPIEL_CHECK_EQ(nlohmann::json::parse(action_json).dump(),
                 FoxAndGeeseActionStruct(action_json).ToJson());

  // Test StructToActions.
  std::vector<Action> expected_actions = {action_id};
  SPIEL_CHECK_EQ(expected_actions, ttt_state->StructToActions(*action_struct));
}

}  // namespace
}  // namespace fox_and_geese
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::fox_and_geese::BasicFoxAndGeeseTests();
  open_spiel::fox_and_geese::TestStateStruct();
  open_spiel::fox_and_geese::TestObservationStruct();
  open_spiel::fox_and_geese::TestActionStruct();
}
