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

#include "open_spiel/games/mfg/crowd_modelling.h"

#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crowd_modelling {
namespace {

namespace testing = open_spiel::testing;

void TestLoad() {
  testing::LoadGameTest("mfg_crowd_modelling");
  auto game = LoadGame("mfg_crowd_modelling");
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kMeanField);
  auto state = game->NewInitialState();
  auto cloned = state->Clone();
  SPIEL_CHECK_EQ(state->ToString(), cloned->ToString());
  testing::ChanceOutcomesTest(*game);
}

void TestLoadWithParams() {
  auto game = LoadGame("mfg_crowd_modelling(size=100,horizon=1000)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->ObservationTensorShape()[0], 1000 + 100 + 1);
}

std::vector<double> RandomDistribution(int num_states, std::mt19937& rng) {
  std::uniform_real_distribution<double> rand(0, 1);
  std::vector<double> distrib;
  for (int i = 0; i < num_states; ++i) {
    distrib.push_back(rand(rng));
  }
  double sum = std::accumulate(distrib.begin(), distrib.end(), 0.);
  for (int i = 0; i < num_states; ++i) {
    distrib[i] /= sum;
  }
  return distrib;
}

void CheckStatesEqual(const State& a, const State& b) {
  const CrowdModellingState& left =
      open_spiel::down_cast<const CrowdModellingState&>(a);
  const CrowdModellingState& right =
      open_spiel::down_cast<const CrowdModellingState&>(b);
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
  std::mt19937 rng(7);
  // TODO(author15): Should we adapt and use testing::RandomSimTest instead?
  auto game = LoadGame("mfg_crowd_modelling(size=10,horizon=20)");
  auto state = game->NewInitialState();
  int t = 0;
  int num_moves = 0;
  while (!state->IsTerminal()) {
    SPIEL_CHECK_LT(state->MoveNumber(), game->MaxMoveNumber());
    SPIEL_CHECK_EQ(state->MoveNumber(), num_moves);
    CheckStatesEqual(*game->DeserializeState(state->Serialize()), *state);
    auto cloned = state->Clone();
    CheckStatesEqual(*state, *cloned);
    testing::CheckLegalActionsAreSorted(*game, *state);
    if (state->CurrentPlayer() == kChancePlayerId) {
      ActionsAndProbs outcomes = state->ChanceOutcomes();
      Action action = open_spiel::SampleAction(outcomes, rng).first;
      SPIEL_CHECK_EQ(state->LegalActions().size(), outcomes.size());
      state->ApplyAction(action);
      ++num_moves;
    } else if (state->CurrentPlayer() == kMeanFieldPlayerId) {
      auto support = state->DistributionSupport();
      state->UpdateDistribution(RandomDistribution(support.size(), rng));
    } else {
      SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(rng)];
      state->ApplyAction(action);
      ++t;
      ++num_moves;
    }
  }
  SPIEL_CHECK_EQ(t, 20);
  SPIEL_CHECK_EQ(state->MoveNumber(), game->MaxMoveNumber());
  SPIEL_CHECK_EQ(state->MoveNumber(), num_moves);
}

void TestReward() {
  auto game = LoadGame("mfg_crowd_modelling(size=10,horizon=20)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(5);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // This expected reward assumes that the game is initialized with
  // a uniform state distribution.
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 1. + std::log(10));
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 1. + std::log(10));

  state->ApplyAction(1);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kMeanFieldPlayerId);
  SPIEL_CHECK_FLOAT_EQ(state->CurrentPlayer(), kMeanFieldPlayerId);
  SPIEL_CHECK_FLOAT_EQ(state->Rewards()[0], 0.);
  SPIEL_CHECK_FLOAT_EQ(state->Returns()[0], 1. + std::log(10));
}

}  // namespace
}  // namespace crowd_modelling
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::crowd_modelling::TestLoad();
  open_spiel::crowd_modelling::TestLoadWithParams();
  open_spiel::crowd_modelling::TestRandomPlay();
  open_spiel::crowd_modelling::TestReward();
}
