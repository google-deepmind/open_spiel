// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/games/pokerkit_wrapper/pokerkit_wrapper_struct.h"

#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace pokerkit_wrapper {
namespace {


void TestPokerkitStateStructDefaults() {
  PokerkitStateStruct s;
  SPIEL_CHECK_EQ(s.observation.size(), 0);
  SPIEL_CHECK_EQ(s.legal_actions.size(), 0);
  SPIEL_CHECK_EQ(s.current_player, 0);
  SPIEL_CHECK_EQ(s.is_terminal, false);
  SPIEL_CHECK_EQ(s.stacks.size(), 0);
  SPIEL_CHECK_EQ(s.bets.size(), 0);
  SPIEL_CHECK_EQ(s.board_cards.size(), 0);
  SPIEL_CHECK_EQ(s.hole_cards.size(), 0);
  SPIEL_CHECK_EQ(s.pots.size(), 0);
  SPIEL_CHECK_EQ(s.burn_cards.size(), 0);
  SPIEL_CHECK_EQ(s.mucked_cards.size(), 0);
  SPIEL_CHECK_EQ(s.poker_hand_histories.size(), 0);
}

void TestToJsonBase() {
  PokerkitStateStruct state_struct;
  state_struct.observation = {"json_test"};
  state_struct.current_player = 0;
  state_struct.is_terminal = false;
  state_struct.legal_actions = {0};
  state_struct.stacks = {10};
  state_struct.bets = {1};
  state_struct.board_cards = {5};
  state_struct.hole_cards = {{6}};
  state_struct.pots = {11};
  state_struct.burn_cards = {7};
  state_struct.mucked_cards = {8};
  state_struct.poker_hand_histories = {{"phh_test_a"}, {"phh_test_b"}};

  nlohmann::json expected_json;
  expected_json["observation"] = std::vector<std::string>{"json_test"};
  expected_json["legal_actions"] = std::vector<int>{0};
  expected_json["current_player"] = 0;
  expected_json["is_terminal"] = false;
  expected_json["stacks"] = std::vector<int>{10};
  expected_json["bets"] = std::vector<int>{1};
  expected_json["board_cards"] = std::vector<int>{5};
  expected_json["hole_cards"] = std::vector<std::vector<int>>{{6}};
  expected_json["pots"] = std::vector<int>{11};
  expected_json["burn_cards"] = std::vector<int>{7};
  expected_json["mucked_cards"] = std::vector<int>{8};
  expected_json["poker_hand_histories"] =
      std::vector<std::vector<std::string>>{{"phh_test_a"}, {"phh_test_b"}};

  nlohmann::json actual_json = state_struct.to_json_base();
  SPIEL_CHECK_EQ(actual_json, expected_json);
}

void TestToJsonString() {
  PokerkitStateStruct state_struct;
  state_struct.current_player = 1;
  state_struct.is_terminal = true;

  std::string json_str = state_struct.ToJson();

  // Parse it back to validate
  nlohmann::json parsed_json = nlohmann::json::parse(json_str);

  SPIEL_CHECK_EQ(parsed_json["current_player"], 1);
  SPIEL_CHECK_EQ(parsed_json["is_terminal"], true);
  // Check for presence of other keys with default values
  SPIEL_CHECK_TRUE(parsed_json.contains("observation"));
  SPIEL_CHECK_TRUE(parsed_json["observation"].empty());
}

}  // namespace
}  // namespace pokerkit_wrapper
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);

  open_spiel::pokerkit_wrapper::TestPokerkitStateStructDefaults();
  open_spiel::pokerkit_wrapper::TestToJsonBase();
  open_spiel::pokerkit_wrapper::TestToJsonString();
}
