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

#include "open_spiel/games/stones_and_gems.h"

#include "open_spiel/abseil-cpp/absl/container/node_hash_map.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "unordered_map"

namespace open_spiel {
namespace stones_and_gems {
namespace {

namespace testing = open_spiel::testing;

void BasicStonesNGemsTests() {
  testing::LoadGameTest("stones_and_gems");
  testing::ChanceOutcomesTest(*LoadGame("stones_and_gems"));
  testing::RandomSimTest(*LoadGame("stones_and_gems"), 100);
}

void BasicStonesNGemsTestsWithParams() {
  constexpr const char kTestDefaultGrid[] =
      "6|7|20|2\n"
      "19|19|19|19|19|19\n"
      "19|01|01|01|01|19\n"
      "19|02|02|01|01|19\n"
      "19|01|01|01|01|19\n"
      "19|00|03|01|02|19\n"
      "19|05|02|05|01|07\n"
      "19|19|19|19|19|19";

  testing::ChanceOutcomesTest(
      *LoadGame("stones_and_gems",
                {{"magic_wall_steps", GameParameter(20)},
                 {"blob_chance", GameParameter(50)},
                 {"blob_max_percentage", GameParameter(0.25)},
                 {"rng_seed", GameParameter(1)},
                 {"grid", GameParameter(std::string(kTestDefaultGrid))}}));
}

void ExtendedStonesNGemsTest() {
  constexpr const char kTestDefaultGrid[] =
      "6|7|20|2\n"
      "19|19|19|19|19|19\n"
      "19|01|01|01|03|19\n"
      "19|02|02|01|01|19\n"
      "19|01|01|01|02|19\n"
      "19|00|03|01|02|19\n"
      "19|05|02|05|01|07\n"
      "19|19|19|19|19|19";

  constexpr const char kStateToString[] =
      "SSSSSS\n"
      "S   oS\n"
      "S..  S\n"
      "S   .S\n"
      "S@o .S\n"
      "S*.* C\n"
      "SSSSSS\n"
      "time left: 20, gems required: 2, gems collectred: 0";

  constexpr const char kStateSerialize[] =
      "6,7,20,20,0,10,0,50,-1,1,2,0,0,0,1,42,0\n"
      "19,1,19,2,19,3,19,4,19,5,19,6\n"
      "19,7,1,8,1,9,1,10,3,11,19,12\n"
      "19,13,2,14,2,15,1,16,1,17,19,18\n"
      "19,19,1,20,1,21,1,22,2,23,19,24\n"
      "19,25,0,26,3,27,1,28,2,29,19,30\n"
      "19,31,5,32,2,33,5,34,1,35,7,36\n"
      "19,37,19,38,19,39,19,40,19,41,19,42";

  // observation tensor index along with corresponding IDs
  const int offset = 6 * 7;
  const absl::node_hash_map<int, int> obs_ids_init{
      {0 * offset + 25, 26},  {1 * offset + 7, 8},    {1 * offset + 8, 9},
      {1 * offset + 9, 10},   {1 * offset + 15, 16},  {1 * offset + 16, 17},
      {1 * offset + 19, 20},  {1 * offset + 20, 21},  {1 * offset + 21, 22},
      {1 * offset + 27, 28},  {1 * offset + 34, 35},  {2 * offset + 13, 14},
      {2 * offset + 14, 15},  {2 * offset + 22, 23},  {2 * offset + 28, 29},
      {2 * offset + 32, 33},  {3 * offset + 10, 11},  {3 * offset + 26, 27},
      {4 * offset + 31, 32},  {4 * offset + 33, 34},  {5 * offset + 35, 36},
      {11 * offset + 0, 1},   {11 * offset + 1, 2},   {11 * offset + 2, 3},
      {11 * offset + 3, 4},   {11 * offset + 4, 5},   {11 * offset + 5, 6},
      {11 * offset + 6, 7},   {11 * offset + 11, 12}, {11 * offset + 12, 13},
      {11 * offset + 17, 18}, {11 * offset + 18, 19}, {11 * offset + 23, 24},
      {11 * offset + 24, 25}, {11 * offset + 29, 30}, {11 * offset + 30, 31},
      {11 * offset + 36, 37}, {11 * offset + 37, 38}, {11 * offset + 38, 39},
      {11 * offset + 39, 40}, {11 * offset + 40, 41}, {11 * offset + 41, 42},
  };

  const absl::node_hash_map<int, int> obs_ids_after{
      {0 * offset + 31, 26},  {1 * offset + 7, 8},    {1 * offset + 8, 9},
      {1 * offset + 9, 10},   {1 * offset + 15, 16},  {1 * offset + 19, 20},
      {1 * offset + 20, 21},  {1 * offset + 21, 22},  {1 * offset + 27, 28},
      {1 * offset + 34, 35},  {1 * offset + 25, 43},  {1 * offset + 10, 44},
      {2 * offset + 13, 14},  {2 * offset + 14, 15},  {2 * offset + 22, 23},
      {2 * offset + 28, 29},  {2 * offset + 32, 33},  {3 * offset + 16, 11},
      {3 * offset + 26, 27},  {4 * offset + 33, 34},  {5 * offset + 35, 36},
      {11 * offset + 0, 1},   {11 * offset + 1, 2},   {11 * offset + 2, 3},
      {11 * offset + 3, 4},   {11 * offset + 4, 5},   {11 * offset + 5, 6},
      {11 * offset + 6, 7},   {11 * offset + 11, 12}, {11 * offset + 12, 13},
      {11 * offset + 17, 18}, {11 * offset + 18, 19}, {11 * offset + 23, 24},
      {11 * offset + 24, 25}, {11 * offset + 29, 30}, {11 * offset + 30, 31},
      {11 * offset + 36, 37}, {11 * offset + 37, 38}, {11 * offset + 38, 39},
      {11 * offset + 39, 40}, {11 * offset + 40, 41}, {11 * offset + 41, 42},
  };

  std::shared_ptr<const Game> game =
      LoadGame("stones_and_gems",
               {{"magic_wall_steps", GameParameter(20)},
                {"blob_chance", GameParameter(50)},
                {"blob_max_percentage", GameParameter(0.25)},
                {"rng_seed", GameParameter(1)},
                {"grid", GameParameter(std::string(kTestDefaultGrid))},
                {"obs_show_ids", GameParameter(true)}});
  std::unique_ptr<State> state = game->NewInitialState();

  // Check max utility
  SPIEL_CHECK_EQ(game->MaxUtility(), 20 + 2 + (2 * 10));

  // Check string functions
  SPIEL_CHECK_EQ(state->ToString(), std::string(kStateToString));
  SPIEL_CHECK_EQ(state->Serialize(), std::string(kStateSerialize));

  // Check the observation tensor IDs
  int i = 0;
  for (const auto& t : state->ObservationTensor()) {
    if (obs_ids_init.find(i) != obs_ids_init.end()) {
      SPIEL_CHECK_EQ(obs_ids_init.at(i), t);
    } else {
      SPIEL_CHECK_EQ(0, t);
    }
    ++i;
  }

  // Collect first diamond
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kDown);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 10);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kNone);

  // Ensure observation tensor IDs tracked properly
  i = 0;
  for (const auto& t : state->ObservationTensor()) {
    if (obs_ids_after.find(i) != obs_ids_after.end()) {
      SPIEL_CHECK_EQ(obs_ids_after.at(i), t);
    } else {
      SPIEL_CHECK_EQ(0, t);
    }
    ++i;
  }

  // Continue towards exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 10);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kNone);

  // Collect second diamond
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 20);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kNone);

  // Continue towards exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 20);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kNone);

  // Move to exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(stones_and_gems::Directions::kRight);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 15);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 35);
}

}  // namespace
}  // namespace stones_and_gems
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::stones_and_gems::BasicStonesNGemsTests();
  open_spiel::stones_and_gems::BasicStonesNGemsTestsWithParams();
  open_spiel::stones_and_gems::ExtendedStonesNGemsTest();
}
