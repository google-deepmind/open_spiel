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

#include "open_spiel/games/laser_tag.h"

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
}
