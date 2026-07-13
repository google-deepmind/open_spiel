// Copyright 2026 DeepMind Technologies Limited
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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace social_deduction {
namespace {

void BasicSocialDeductionTests() {
  testing::LoadGameTest("social_deduction");
  testing::RandomSimTest(*LoadGame("social_deduction"), 10);
  testing::RandomSimTest(*LoadGame("social_deduction",
                                   {{"players", GameParameter(5)},
                                    {"imposters", GameParameter(1)},
                                    {"max_rounds", GameParameter(4)},
                                    {"observation_noise", GameParameter(0.0)}}),
                         10);
}

void HiddenRoleObservationTest() {
  std::shared_ptr<const Game> game =
      LoadGame("social_deduction", {{"players", GameParameter(4)},
                                    {"imposters", GameParameter(1)},
                                    {"max_rounds", GameParameter(3)},
                                    {"observation_noise", GameParameter(0.0)}});
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(1);  // Player 0 is the single imposter.

  const std::string imposter_observation = state->ObservationString(0);
  SPIEL_CHECK_NE(imposter_observation.find("Role: Imposter"),
                 std::string::npos);
  SPIEL_CHECK_NE(imposter_observation.find("ImposterTeam: 0"),
                 std::string::npos);

  const std::string innocent_observation = state->ObservationString(1);
  SPIEL_CHECK_NE(innocent_observation.find("Role: Innocent"),
                 std::string::npos);
  SPIEL_CHECK_EQ(innocent_observation.find("ImposterTeam"), std::string::npos);
}

void InnocentsCanEliminateImposterTest() {
  std::shared_ptr<const Game> game =
      LoadGame("social_deduction", {{"players", GameParameter(4)},
                                    {"imposters", GameParameter(1)},
                                    {"max_rounds", GameParameter(3)},
                                    {"observation_noise", GameParameter(0.0)}});
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(1);  // Player 0 is the single imposter.
  state->ApplyAction(1);  // Public signal: player 0 suspicious.

  state->ApplyAction(0);  // Player 0: SKIP.
  state->ApplyAction(2);  // Player 1: ACCUSE_PLAYER_0.
  state->ApplyAction(2);  // Player 2: ACCUSE_PLAYER_0.
  state->ApplyAction(2);  // Player 3: ACCUSE_PLAYER_0.

  state->ApplyAction(1);  // Player 0 votes for player 1.
  state->ApplyAction(0);  // Player 1 votes for player 0.
  state->ApplyAction(0);  // Player 2 votes for player 0.
  state->ApplyAction(0);  // Player 3 votes for player 0.

  SPIEL_CHECK_TRUE(state->IsTerminal());
  const std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], -1.0);
  SPIEL_CHECK_EQ(returns[1], 1.0);
  SPIEL_CHECK_EQ(returns[2], 1.0);
  SPIEL_CHECK_EQ(returns[3], 1.0);
}

}  // namespace
}  // namespace social_deduction
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::social_deduction::BasicSocialDeductionTests();
  open_spiel::social_deduction::HiddenRoleObservationTest();
  open_spiel::social_deduction::InnocentsCanEliminateImposterTest();
}
