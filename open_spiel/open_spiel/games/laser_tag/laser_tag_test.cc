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

#include "open_spiel/games/laser_tag/laser_tag.h"

#include <memory>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace laser_tag {
namespace {

namespace testing = open_spiel::testing;

// Spawn location values for the default map only.
constexpr int kTopLeftSpawnOutcome = kNumInitiativeChanceOutcomes;
constexpr int kTopRightSpawnOutcome = kNumInitiativeChanceOutcomes + 1;

void BasicLaserTagTests() {
  testing::LoadGameTest("laser_tag");
  testing::ChanceOutcomesTest(*LoadGame("laser_tag"));
  testing::RandomSimTest(*LoadGame("laser_tag"), 100);
}

void SimpleTagTests(int horizon, bool zero_sum, std::string grid) {
  std::shared_ptr<const Game> game =
      LoadGame("laser_tag", {{"horizon", GameParameter(horizon)},
                             {"zero_sum", GameParameter(zero_sum)},
                             {"grid", GameParameter(grid)}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopRightSpawnOutcome);  // Spawn B top-right
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopLeftSpawnOutcome);  // Spawn A top-left

  // Both facing south
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({0, 1});  // A: Turn left, B: Turn right.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({6, 1});  // A: Stand, B: Turn right.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({6, 2});  // A: Stand, B: Move forward.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({6, 0});  // A: Stand, B: Turn left.
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({9, 9});  // stand-off!
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit1Action);  // chance node: player 1 first

  std::cout << state->ToString() << std::endl;

  if (horizon == -1) {
    // End of episode (since horizon = -1)
    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->PlayerReward(0), zero_sum ? -1 : 0);
    SPIEL_CHECK_EQ(state->PlayerReward(1), 1);
    SPIEL_CHECK_EQ(state->PlayerReturn(0), zero_sum ? -1 : 0);
    SPIEL_CHECK_EQ(state->PlayerReturn(1), 1);
    return;
  } else {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->PlayerReward(0), zero_sum ? -1 : 0);
    SPIEL_CHECK_EQ(state->PlayerReward(1), 1);
    SPIEL_CHECK_EQ(state->PlayerReturn(0), zero_sum ? -1 : 0);
    SPIEL_CHECK_EQ(state->PlayerReturn(1), 1);
  }

  std::cout << state->ToString() << std::endl;

  // horizon > 0, continue... do it again!
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopLeftSpawnOutcome);  // Spawn A at top-left again
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyActions({9, 9});  // stand-off!
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopRightSpawnOutcome);  // Spawn B at top-right again
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  // Immediate tag reward goes to player 0.
  SPIEL_CHECK_EQ(state->PlayerReward(0), 1);
  SPIEL_CHECK_EQ(state->PlayerReward(1), zero_sum ? -1 : 0);

  // Now they have a tag each. In a zero-sum game, their returns are both 0.
  // Otherwise, they each have 1.
  SPIEL_CHECK_EQ(state->PlayerReturn(0), zero_sum ? 0 : 1);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), zero_sum ? 0 : 1);
}

void BasicLaserTagTestsBigGrid() {
  constexpr const char big_grid[] =
      ".....S................\n"
      "S..***....*.....S**...\n"
      "...*S..****...*......*\n"
      ".......*S.**..*...****\n"
      "..**...*......*......*\n"
      "..S....*......**....**\n"
      "**....***.....*S....**\n"
      "S......*.....**......S\n"
      "*...*........S**......\n"
      "**..**....**........**\n"
      "*....................S\n";
  testing::ChanceOutcomesTest(
      *LoadGame("laser_tag", {{"grid", GameParameter(std::string(big_grid))}}));
  testing::RandomSimTest(
      *LoadGame("laser_tag", {{"grid", GameParameter(std::string(big_grid))}}),
      10);
}

void BasicPartiallyObservableLaserTagTests() {
  testing::ChanceOutcomesTest(
      *LoadGame("laser_tag", {{"fully_obs", GameParameter(false)}}));

  testing::RandomSimTest(
      *LoadGame("laser_tag", {{"fully_obs", GameParameter(false)}}), 100);
}

std::vector<float> get_obs_tensor_from_string(const std::string& obs_string,
                                              int obs_grid_size) {
  std::vector<float> tensor(4 * obs_grid_size, 0.0);

  int num_newlines = 0;
  for (int i = 0; i < obs_string.length(); i++) {
    switch (obs_string[i]) {
      case 'A':
        tensor[i - num_newlines] = 1.0;
        break;
      case 'B':
        tensor[obs_grid_size + i - num_newlines] = 1.0;
        break;
      case '.':
        tensor[2 * obs_grid_size + i - num_newlines] = 1.0;
        break;
      case '*':
        tensor[3 * obs_grid_size + i - num_newlines] = 1.0;
        break;
      case '\n':
        num_newlines += 1;
        break;
      default:
        // Reached 'O' in "Orientations"
        SPIEL_CHECK_EQ(obs_string[i], 'O');
        return tensor;
    }
  }
  return tensor;
}

void PartiallyObservableLaserTagDefaultObsTests() {
  float tolerence = 0.0001;
  std::shared_ptr<const Game> game =
      LoadGame("laser_tag", {{"fully_obs", GameParameter(false)},
                             {"obs_front", GameParameter(17)},
                             {"obs_back", GameParameter(2)},
                             {"obs_side", GameParameter(10)},
                             {"grid", GameParameter(laser_tag::kDefaultGrid)}});
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(kTopRightSpawnOutcome);  // Spawn B top-right
  state->ApplyAction(kTopLeftSpawnOutcome);   // Spawn A top-left

  // A.....B
  // .......
  // ..*.*..
  // .**.**.
  // ..*.*..
  // .......
  // .......
  //
  // Both A and B facing south

  int obs_grid_size = (17 + 2 + 1) * (2 * 10 + 1);
  std::string expected_obs_string_A =
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "****.......**********\n"
      "****.......**********\n"
      "****..*.*..**********\n"
      "****.**.**.**********\n"
      "****..*.*..**********\n"
      "****.......**********\n"
      "****B.....A**********\n"
      "*********************\n"
      "*********************\n"
      "Orientations: 1 1\n";
  std::vector<float> expected_obs_tensor_A =
      get_obs_tensor_from_string(expected_obs_string_A, obs_grid_size);

  std::string expected_obs_string_B =
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "**********.......****\n"
      "**********.......****\n"
      "**********..*.*..****\n"
      "**********.**.**.****\n"
      "**********..*.*..****\n"
      "**********.......****\n"
      "**********B.....A****\n"
      "*********************\n"
      "*********************\n"
      "Orientations: 1 1\n";
  std::vector<float> expected_obs_tensor_B =
      get_obs_tensor_from_string(expected_obs_string_B, obs_grid_size);

  SPIEL_CHECK_EQ(expected_obs_string_A, state->ObservationString(0));
  SPIEL_CHECK_EQ(expected_obs_string_B, state->ObservationString(1));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_A, state->ObservationTensor(0), tolerence));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_B, state->ObservationTensor(1), tolerence));

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({0, 1});             // A: Turn left, B: Turn right.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  // .......
  // ..A.B..
  // ..*.*..
  // .**.**.
  // ..*.*..
  // .......
  // .......
  //
  // A facing east, B facing west

  expected_obs_string_A =
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********.......*****\n"
      "*********...*...*****\n"
      "*********.B***..*****\n"
      "*********.......*****\n"
      "*********.A***..*****\n"
      "*********...*...*****\n"
      "*********.......*****\n"
      "Orientations: 2 3\n";
  expected_obs_tensor_A =
      get_obs_tensor_from_string(expected_obs_string_A, obs_grid_size);

  expected_obs_string_B =
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*********************\n"
      "*****.......*********\n"
      "*****...*...*********\n"
      "*****..***A.*********\n"
      "*****.......*********\n"
      "*****..***B.*********\n"
      "*****...*...*********\n"
      "*****.......*********\n"
      "Orientations: 2 3\n";
  expected_obs_tensor_B =
      get_obs_tensor_from_string(expected_obs_string_B, obs_grid_size);

  SPIEL_CHECK_EQ(expected_obs_string_A, state->ObservationString(0));
  SPIEL_CHECK_EQ(expected_obs_string_B, state->ObservationString(1));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_A, state->ObservationTensor(0), tolerence));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_B, state->ObservationTensor(1), tolerence));
}

void PartiallyObservableLaserTagSmallObsTests() {
  float tolerence = 0.0001;
  std::shared_ptr<const Game> game =
      LoadGame("laser_tag", {{"fully_obs", GameParameter(false)},
                             {"obs_front", GameParameter(2)},
                             {"obs_back", GameParameter(1)},
                             {"obs_side", GameParameter(1)},
                             {"grid", GameParameter(laser_tag::kDefaultGrid)}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopRightSpawnOutcome);  // Spawn B top-right
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(kTopLeftSpawnOutcome);  // Spawn A top-left

  // A.....B
  // .......
  // ..*.*..
  // .**.**.
  // ..*.*..
  // .......
  // .......
  //
  // Both A and B facing south

  std::string expected_obs_string_A =
      "..*\n"
      "..*\n"
      ".A*\n"
      "***\n"
      "Orientations: 1 -1\n";
  std::vector<float> expected_obs_tensor_A = {
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // Plane 0: A
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Plane 1: B
      1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,  // Plane 2: .
      0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1   // Plane 3: *
  };

  std::string expected_obs_string_B =
      "*..\n"
      "*..\n"
      "*B.\n"
      "***\n"
      "Orientations: -1 1\n";
  std::vector<float> expected_obs_tensor_B = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Plane 0: A
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // Plane 1: B
      0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,  // Plane 2: .
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1   // Plane 3: *
  };

  SPIEL_CHECK_EQ(expected_obs_string_A, state->ObservationString(0));
  SPIEL_CHECK_EQ(expected_obs_string_B, state->ObservationString(1));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_A, state->ObservationTensor(0), tolerence));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_B, state->ObservationTensor(1), tolerence));

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({0, 1});             // A: Turn left, B: Turn right.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  state->ApplyActions({2, 2});             // A: Move forward, B: Move forward.
  state->ApplyAction(kChanceInit0Action);  // chance node: player 0 first

  // .......
  // ..A.B..
  // ..*.*..
  // .**.**.
  // ..*.*..
  // .......
  // .......
  //
  // A facing east, B facing west

  expected_obs_string_A =
      ".B*\n"
      "...\n"
      ".A*\n"
      "...\n"
      "Orientations: 2 3\n";
  expected_obs_tensor_A = {
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // Plane 0: A
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Plane 1: B
      1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,  // Plane 2: .
      0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0   // Plane 3: *
  };

  expected_obs_string_B =
      "*A.\n"
      "...\n"
      "*B.\n"
      "...\n"
      "Orientations: 2 3\n";
  expected_obs_tensor_B = {
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Plane 0: A
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // Plane 1: B
      0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,  // Plane 2: .
      1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0   // Plane 3: *
  };

  SPIEL_CHECK_EQ(expected_obs_string_A, state->ObservationString(0));
  SPIEL_CHECK_EQ(expected_obs_string_B, state->ObservationString(1));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_A, state->ObservationTensor(0), tolerence));
  SPIEL_CHECK_TRUE(
      AllNear(expected_obs_tensor_B, state->ObservationTensor(1), tolerence));
}

}  // namespace
}  // namespace laser_tag
}  // namespace open_spiel

namespace laser_tag = open_spiel::laser_tag;

int main(int argc, char **argv) {
  laser_tag::SimpleTagTests(-1, true, laser_tag::kDefaultGrid);
  laser_tag::SimpleTagTests(-1, false, laser_tag::kDefaultGrid);
  laser_tag::SimpleTagTests(1000, true, laser_tag::kDefaultGrid);
  laser_tag::SimpleTagTests(1000, false, laser_tag::kDefaultGrid);
  laser_tag::BasicLaserTagTests();
  laser_tag::BasicLaserTagTestsBigGrid();
  laser_tag::BasicPartiallyObservableLaserTagTests();
  laser_tag::PartiallyObservableLaserTagSmallObsTests();
  laser_tag::PartiallyObservableLaserTagDefaultObsTests();
}
