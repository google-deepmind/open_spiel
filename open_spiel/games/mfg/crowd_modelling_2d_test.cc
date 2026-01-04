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

#include "open_spiel/games/mfg/crowd_modelling_2d.h"

#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crowd_modelling_2d {
namespace {

namespace testing = open_spiel::testing;

void TestLoad() {
  testing::LoadGameTest("mfg_crowd_modelling_2d");
  auto game = LoadGame("mfg_crowd_modelling_2d");
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kMeanField);
  auto state = game->NewInitialState();
  auto cloned = state->Clone();
  SPIEL_CHECK_EQ(state->ToString(), cloned->ToString());
  testing::ChanceOutcomesTest(*game);
}

void TestLoadWithParams() {
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=100,horizon=1000,"
      "only_distribution_reward=true)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->ObservationTensorShape()[0], 1000 + 2 * 100 + 1);
}

void TestLoadWithParams2() {
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=100,horizon=1000,forbidden_states=[0|0;0|1]"
      ",initial_distribution=[0|2;0|3],initial_distribution_value=[0.5;0.5]"
      ",positional_reward=[1|3;1|4],positional_reward_value=[0.2;0.8]"
      ",noise_intensity=0.5,crowd_aversion_coef=0.4,with_congestion=true"
      ")");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->ObservationTensorShape()[0], 1000 + 2 * 100 + 1);
}

void TestRandomPlay() {
  testing::LoadGameTest("mfg_crowd_modelling_2d(size=10,horizon=20)");
  testing::RandomSimTest(
      *LoadGame("mfg_crowd_modelling_2d(size=10,horizon=20)"), 3);
}

void TestReward() {
  auto game = LoadGame("mfg_crowd_modelling_2d(size=10,horizon=20)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(55);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // This expected reward assumes that the game is initialized with
  // a uniform state distribution.
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 6.60517);
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 6.60517);

  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 0.);
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 6.60517);
}

void TestDistRewardOnly() {
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=10,horizon=20"
      ",only_distribution_reward=true)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(55);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 4.60517);
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 4.60517);
}

void TestPositionalReward() {
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=10,horizon=20"
      ",positional_reward=[1|2;2|2],positional_reward_value=[0.5;0.5])");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(55);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 5.30517);
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 5.30517);
}

void TestProcess() {
  auto split_string_list0 = ProcessStringParam("[]", 5);
  SPIEL_CHECK_EQ(split_string_list0.size(), 0);
  auto split_string_list1 = ProcessStringParam("[0|0;0|1]", 5);
  SPIEL_CHECK_EQ(split_string_list1.size(), 2);
  auto split_string_list2 = ProcessStringParam("[0|2;0|3;0|4]", 5);
  SPIEL_CHECK_EQ(split_string_list2.size(), 3);
  auto split_string_list3 = ProcessStringParam("[0.5;0.5]", 5);
  SPIEL_CHECK_EQ(split_string_list3.size(), 2);
}

void TestLegalActions() {
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=5,horizon=10,forbidden_states=[0|0;0|1;1|0]"
      ",initial_distribution=[1|1],initial_distribution_value=[1.0])");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  // Legal action will be the state in the initial distribution.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({6}));
  state->ApplyAction(6);
  // Legal actions are moving right, down or no movement.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({2, 3, 4}));
  // Go right.
  state->ApplyAction(3);
  // Chance node. No forbidden states around and all actions are legal.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({0, 1, 2, 3, 4}));
  // Go left.
  state->ApplyAction(1);
  // Mean field node; legal actions will be empty.
  SPIEL_CHECK_TRUE(state->LegalActions().empty());
  std::vector<double> dist(25);
  state->UpdateDistribution(dist);
  // Back to starting point.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({2, 3, 4}));
  // Stay in the same position.
  state->ApplyAction(2);
  // Chance node. The legal actions should be the same.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({2, 3, 4}));
}

void TestNoiseIntensity() {
  // Same as the setting in TestLegalActions() test above, except the noise
  // intensity.
  auto game = LoadGame(
      "mfg_crowd_modelling_2d(size=5,horizon=10,forbidden_states=[0|0;0|1;1|0]"
      ",initial_distribution=[1|1],initial_distribution_value=[1.0]"
      ",noise_intensity=0.5)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({6}));
  state->ApplyAction(6);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({2, 3, 4}));
  state->ApplyAction(3);
  // Now we are at a chance node.
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({0, 1, 2, 3, 4}));
  // Neutral action should have a probability of 0.5+0.1 and others 0.1, i.e.
  // 0.5 / 5.
  ActionsAndProbs expected_outcomes(
      {{0, 0.1}, {1, 0.1}, {2, 0.6}, {3, 0.1}, {4, 0.1}});
  SPIEL_CHECK_EQ(state->ChanceOutcomes(), expected_outcomes);
}

}  // namespace
}  // namespace crowd_modelling_2d
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::crowd_modelling_2d::TestLegalActions();
  open_spiel::crowd_modelling_2d::TestLoad();
  open_spiel::crowd_modelling_2d::TestLoadWithParams();
  open_spiel::crowd_modelling_2d::TestLoadWithParams2();
  open_spiel::crowd_modelling_2d::TestRandomPlay();
  open_spiel::crowd_modelling_2d::TestReward();
  open_spiel::crowd_modelling_2d::TestDistRewardOnly();
  open_spiel::crowd_modelling_2d::TestPositionalReward();
  open_spiel::crowd_modelling_2d::TestProcess();
  open_spiel::crowd_modelling_2d::TestNoiseIntensity();
}
