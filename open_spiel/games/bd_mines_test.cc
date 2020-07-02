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
      "19,01,01,01,01,19\n"
      "19,02,02,01,01,19\n"
      "19,01,01,01,01,19\n"
      "19,00,03,01,02,19\n"
      "19,05,02,05,01,07\n"
      "19,19,19,19,19,19";

    constexpr const char kStateToString[] = 
      "SSSSSS\n"
      "S    S\n"
      "S..  S\n"
      "S    S\n"
      "S@o .S\n"
      "S*.* C\n"
      "SSSSSS\n"
      "time left: 20, gems required: 2, gems collectred: 0";

    constexpr const char kStateSerialize[] = 
      "6,7,20,20,0,10,0,50,-1,1,2,0,0,0,0\n"
      "19,19,19,19,19,19\n"
      "19,1,1,1,1,19\n"
      "19,2,2,1,1,19\n"
      "19,1,1,1,1,19\n"
      "19,0,3,1,2,19\n"
      "19,5,2,5,1,7\n"
      "19,19,19,19,19,19";

  std::shared_ptr<const Game> game =
      LoadGame("bd_mines", {{"magic_wall_steps", GameParameter(20)},
                             {"amoeba_chance", GameParameter(50)},
                             {"amoeba_max_percentage", GameParameter(0.25)},
                             {"rng_seed", GameParameter(1)},
                             {"grid", GameParameter(std::string(kTestDefaultGrid))}
                             });
  std::unique_ptr<State> state = game->NewInitialState();

  // Check string functions
  SPIEL_CHECK_EQ(state->ToString(), std::string(kStateToString));
  SPIEL_CHECK_EQ(state->Serialize(), std::string(kStateSerialize));

  // Collect first diamond
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kDown);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 10);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(bd_mines::Directions::kNone);

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
