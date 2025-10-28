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

#include "open_spiel/games/repeated_pokerkit/repeated_pokerkit_struct.h"

#include <map>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/games/pokerkit_wrapper/pokerkit_wrapper_struct.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace repeated_pokerkit {
namespace {

void TestRepeatedPokerkitStateStructDefaults() {
  RepeatedPokerkitStateStruct s;
  SPIEL_CHECK_EQ(s.hand_number, 0);
  SPIEL_CHECK_EQ(s.is_terminal, false);
  SPIEL_CHECK_EQ(s.stacks.size(), 0);
  SPIEL_CHECK_EQ(s.dealer, kInvalidPlayer);
  SPIEL_CHECK_EQ(s.seat_to_player.size(), 0);
  SPIEL_CHECK_EQ(s.player_to_seat.size(), 0);
  SPIEL_CHECK_EQ(s.small_blind, kInvalidBlindBetSizeOrBringIn);
  SPIEL_CHECK_EQ(s.big_blind, kInvalidBlindBetSizeOrBringIn);
  SPIEL_CHECK_EQ(s.small_bet_size, kInvalidBlindBetSizeOrBringIn);
  SPIEL_CHECK_EQ(s.big_bet_size, kInvalidBlindBetSizeOrBringIn);
  SPIEL_CHECK_EQ(s.bring_in, kInvalidBlindBetSizeOrBringIn);
  SPIEL_CHECK_EQ(s.hand_returns.size(), 0);

  pokerkit_wrapper::PokerkitStateStruct& pokerkit_state_struct =
      s.pokerkit_state_struct;
  SPIEL_CHECK_EQ(pokerkit_state_struct.observation.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.legal_actions.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.current_player, 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.is_terminal, false);
  SPIEL_CHECK_EQ(pokerkit_state_struct.stacks.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.bets.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.board_cards.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.hole_cards.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.pots.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.burn_cards.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.mucked_cards.size(), 0);
  SPIEL_CHECK_EQ(pokerkit_state_struct.poker_hand_histories.size(), 0);
}

void TestToJsonBase() {
  repeated_pokerkit::RepeatedPokerkitStateStruct state_struct;
  state_struct.hand_number = 1;
  state_struct.is_terminal = true;
  state_struct.stacks = {100, 200};
  state_struct.dealer = 1;
  state_struct.seat_to_player = {{0, 1}, {1, 0}};
  state_struct.player_to_seat = {{0, 1}, {1, 0}};
  state_struct.small_blind = 10;
  state_struct.big_blind = 20;
  state_struct.small_bet_size = 10;
  state_struct.big_bet_size = 20;
  state_struct.bring_in = 5;
  state_struct.hand_returns = {{10.0, -10.0}};
  pokerkit_wrapper::PokerkitStateStruct& pokerkit_state_struct =
      state_struct.pokerkit_state_struct;
  pokerkit_state_struct.observation = {"json_test"};
  pokerkit_state_struct.current_player = 0;
  pokerkit_state_struct.is_terminal = false;
  pokerkit_state_struct.legal_actions = {0};
  pokerkit_state_struct.stacks = {10};
  pokerkit_state_struct.bets = {1};
  pokerkit_state_struct.board_cards = {5};
  pokerkit_state_struct.hole_cards = {{6}};
  pokerkit_state_struct.pots = {11};
  pokerkit_state_struct.burn_cards = {7};
  pokerkit_state_struct.mucked_cards = {8};
  pokerkit_state_struct.poker_hand_histories = {{"phh_test_a"}, {"phh_test_b"}};

  nlohmann::json expected_json;
  expected_json["hand_number"] = 1;
  expected_json["is_terminal"] = true;
  expected_json["stacks"] = std::vector<int>{100, 200};
  expected_json["dealer"] = 1;
  expected_json["seat_to_player"] = std::map<int, int>{{0, 1}, {1, 0}};
  expected_json["player_to_seat"] = std::map<int, int>{{0, 1}, {1, 0}};
  expected_json["small_blind"] = 10;
  expected_json["big_blind"] = 20;
  expected_json["small_bet_size"] = 10;
  expected_json["big_bet_size"] = 20;
  expected_json["bring_in"] = 5;
  expected_json["hand_returns"] =
      std::vector<std::vector<float>>{{10.0, -10.0}};
  expected_json["pokerkit_state_struct"]["observation"] =
      std::vector<std::string>{"json_test"};
  expected_json["pokerkit_state_struct"]["legal_actions"] = std::vector<int>{0};
  expected_json["pokerkit_state_struct"]["current_player"] = 0;
  expected_json["pokerkit_state_struct"]["is_terminal"] = false;
  expected_json["pokerkit_state_struct"]["stacks"] = std::vector<int>{10};
  expected_json["pokerkit_state_struct"]["bets"] = std::vector<int>{1};
  expected_json["pokerkit_state_struct"]["board_cards"] = std::vector<int>{5};
  expected_json["pokerkit_state_struct"]["hole_cards"] =
      std::vector<std::vector<int>>{{6}};
  expected_json["pokerkit_state_struct"]["pots"] = std::vector<int>{11};
  expected_json["pokerkit_state_struct"]["burn_cards"] = std::vector<int>{7};
  expected_json["pokerkit_state_struct"]["mucked_cards"] = std::vector<int>{8};
  expected_json["pokerkit_state_struct"]["poker_hand_histories"] =
      std::vector<std::vector<std::string>>{{"phh_test_a"}, {"phh_test_b"}};

  nlohmann::json actual_json = state_struct.to_json_base();
  SPIEL_CHECK_EQ(actual_json["pokerkit_state_struct"],
                 expected_json["pokerkit_state_struct"]);
  SPIEL_CHECK_EQ(actual_json, expected_json);
}

void TestToJsonString() {
  RepeatedPokerkitStateStruct state_struct;
  state_struct.pokerkit_state_struct.current_player = 1;
  state_struct.pokerkit_state_struct.is_terminal = true;

  std::string json_str = state_struct.ToJson();

  // Parse it back to validate
  nlohmann::json parsed_json = nlohmann::json::parse(json_str);
  auto parsed_pokerkit_state_struct = parsed_json["pokerkit_state_struct"];
  SPIEL_CHECK_EQ(parsed_pokerkit_state_struct["current_player"], 1);
  SPIEL_CHECK_EQ(parsed_pokerkit_state_struct["is_terminal"], true);
  // Check for presence of other keys with default values
  SPIEL_CHECK_TRUE(parsed_pokerkit_state_struct.contains("observation"));
  SPIEL_CHECK_TRUE(parsed_pokerkit_state_struct["observation"].empty());
}

}  // namespace
}  // namespace repeated_pokerkit
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);

  open_spiel::repeated_pokerkit::TestRepeatedPokerkitStateStructDefaults();
  open_spiel::repeated_pokerkit::TestToJsonBase();
  open_spiel::repeated_pokerkit::TestToJsonString();
}
