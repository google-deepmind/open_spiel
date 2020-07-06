// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/bd_mines.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

#include "unordered_map"

namespace open_spiel {
namespace bd_mines {
namespace {

namespace testing = open_spiel::testing;


void BasicBDMinesTests() {
  testing::LoadGameTest("bd_mines");
  testing::ChanceOutcomesTest(*LoadGame("bd_mines"));
  testing::RandomSimTest(*LoadGame("bd_mines"), 100);
}

void BasicBDMinesTestsWithParams() {
  constexpr const char kTestDefaultGrid[] =
      "6,7,20,2\n"
      "19,19,19,19,19,19\n"
      "19,01,01,01,01,19\n"
      "19,02,02,01,01,19\n"
      "19,01,01,01,01,19\n"
      "19,00,03,01,02,19\n"
      "19,05,02,05,01,07\n"
      "19,19,19,19,19,19";
  
  testing::ChanceOutcomesTest(
      *LoadGame("bd_mines", {{"magic_wall_steps", GameParameter(20)},
                             {"amoeba_chance", GameParameter(50)},
                             {"amoeba_max_percentage", GameParameter(0.25)},
                             {"rng_seed", GameParameter(1)},
                             {"grid", GameParameter(std::string(kTestDefaultGrid))}
                             }));
}

void ExtendedBDMinesTest() {
  constexpr const char kTestDefaultGrid[] =
      "6,7,20,2\n"
      "19,19,19,19,19,19\n"
      "19,01,01,01,03,19\n"
      "19,02,02,01,01,19\n"
      "19,01,01,01,02,19\n"
      "19,00,03,01,02,19\n"
      "19,05,02,05,01,07\n"
      "19,19,19,19,19,19";

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
      "6,7,20,20,0,10,0,50,-1,1,2,0,0,0,1,27,0\n"
      "19,19,19,19,19,19\n"
      "19,1,1,1,3,19\n"
      "19,2,2,1,1,19\n"
      "19,1,1,1,2,19\n"
      "19,0,3,1,2,19\n"
      "19,5,2,5,1,7\n"
      "19,19,19,19,19,19";

    // observation tensor index along with corresponding IDs
    const int offset = 6*7;
    const std::unordered_map<int, int> obs_ids_init {
      {0*offset + 25, 15},  {3*offset + 10, 8}, {3*offset + 26, 16}, 
      {4*offset + 31, 19}, {4*offset + 33, 20}, {5*offset + 35, 21}, 
      {11*offset + 0, 1}, {11*offset + 1, 2}, {11*offset + 2, 3},
      {11*offset + 3, 4}, {11*offset + 4, 5}, {11*offset + 5, 6},
      {11*offset + 6, 7}, {11*offset + 11, 9},
      {11*offset + 12, 10}, {11*offset + 17, 11},
      {11*offset + 18, 12}, {11*offset + 23, 13},
      {11*offset + 24, 14}, {11*offset + 29, 17},
      {11*offset + 30, 18},
      {11*offset + 36, 22}, {11*offset + 37, 23}, {11*offset + 38, 24},
      {11*offset + 39, 25}, {11*offset + 40, 26}, {11*offset + 41, 27},
    };

    const std::unordered_map<int, int> obs_ids_after {
      {0*offset + 31, 15},  {3*offset + 16, 8}, {3*offset + 26, 16}, 
      {4*offset + 33, 20}, {5*offset + 35, 21}, 
      {11*offset + 0, 1}, {11*offset + 1, 2}, {11*offset + 2, 3},
      {11*offset + 3, 4}, {11*offset + 4, 5}, {11*offset + 5, 6},
      {11*offset + 6, 7}, {11*offset + 11, 9},
      {11*offset + 12, 10}, {11*offset + 17, 11},
      {11*offset + 18, 12}, {11*offset + 23, 13},
      {11*offset + 24, 14}, {11*offset + 29, 17},
      {11*offset + 30, 18},
      {11*offset + 36, 22}, {11*offset + 37, 23}, {11*offset + 38, 24},
      {11*offset + 39, 25}, {11*offset + 40, 26}, {11*offset + 41, 27},
    };

  std::shared_ptr<const Game> game =
      LoadGame("bd_mines", {{"magic_wall_steps", GameParameter(20)},
                             {"amoeba_chance", GameParameter(50)},
                             {"amoeba_max_percentage", GameParameter(0.25)},
                             {"rng_seed", GameParameter(1)},
                             {"grid", GameParameter(std::string(kTestDefaultGrid))},
                             {"obs_show_ids", GameParameter(true)}
                             });
  std::unique_ptr<State> state = game->NewInitialState();

  // Check max utility
  SPIEL_CHECK_EQ(game->MaxUtility(), 20+2+(2*10));

  // Check string functions
  SPIEL_CHECK_EQ(state->ToString(), std::string(kStateToString));
  SPIEL_CHECK_EQ(state->Serialize(), std::string(kStateSerialize));

  // Check the observation tensor IDs
  int i = 0;
  for (const auto & t : state->ObservationTensor()) {
    if (obs_ids_init.find(i) != obs_ids_init.end()) {
      SPIEL_CHECK_EQ(obs_ids_init.at(i), t);
    }
    else {
      SPIEL_CHECK_EQ(0, t);
    }
    ++i;
  }

  // Collect first diamond
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kDown);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 10);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kNone);

  // Ensure observation tensor IDs tracked properly
  i = 0;
  for (const auto & t : state->ObservationTensor()) {
    if (obs_ids_after.find(i) != obs_ids_after.end()) {
      SPIEL_CHECK_EQ(obs_ids_after.at(i), t);
    }
    else {
      SPIEL_CHECK_EQ(0, t);
    }
    ++i;
  }

  // Continue towards exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 10);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kNone);

  // Collect second diamond
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 20);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kNone);

  // Continue towards exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kRight);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 20);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kNone);

  // Move to exit
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kRight);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 15);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 35);
}

}  // namespace
}  // namespace bd_mines
}  // namespace open_spiel

int main(int argc, char** argv) { 
  open_spiel::bd_mines::BasicBDMinesTests(); 
  open_spiel::bd_mines::BasicBDMinesTestsWithParams(); 
  open_spiel::bd_mines::ExtendedBDMinesTest(); 
}
