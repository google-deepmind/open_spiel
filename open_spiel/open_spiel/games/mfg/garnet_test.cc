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

#include "open_spiel/games/mfg/garnet.h"

#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace garnet {
namespace {

namespace testing = open_spiel::testing;

void TestLoad() {
  testing::LoadGameTest("mfg_garnet");
  auto game = LoadGame("mfg_garnet");
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kMeanField);
  auto state = game->NewInitialState();
  auto cloned = state->Clone();
  SPIEL_CHECK_EQ(state->ToString(), cloned->ToString());
  testing::ChanceOutcomesTest(*game);
}

void TestLoadWithParams() {
  auto game = LoadGame("mfg_garnet(size=100,horizon=1000)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->ObservationTensorShape()[0], 1000 + 100 + 1);
}

void CheckStatesEqual(const State& a, const State& b) {
  const GarnetState& left = open_spiel::down_cast<const GarnetState&>(a);
  const GarnetState& right = open_spiel::down_cast<const GarnetState&>(b);
  SPIEL_CHECK_EQ(left.ToString(), right.ToString());
  SPIEL_CHECK_FLOAT_EQ(left.Rewards()[0], right.Rewards()[0]);
  SPIEL_CHECK_FLOAT_EQ(left.Returns()[0], right.Returns()[0]);
  SPIEL_CHECK_EQ(left.CurrentPlayer(), right.CurrentPlayer());
  auto left_distrib = left.Distribution();
  auto right_distrib = right.Distribution();
  SPIEL_CHECK_EQ(left_distrib.size(), right_distrib.size());
  for (int i = 0; i < left_distrib.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(left_distrib[i], right_distrib[i]);
  }
}

void TestRandomPlay() {
  testing::LoadGameTest("mfg_garnet(size=10,horizon=20)");
  testing::RandomSimTest(*LoadGame("mfg_garnet(size=10,horizon=20)"), 3);
}

}  // namespace
}  // namespace garnet
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::garnet::TestLoad();
  open_spiel::garnet::TestLoadWithParams();
  open_spiel::garnet::TestRandomPlay();
}
