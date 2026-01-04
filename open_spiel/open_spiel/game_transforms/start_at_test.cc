// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/game_transforms/start_at.h"
#include <memory>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace start_at {
namespace {

namespace testing = open_spiel::testing;

void BasicStartAtTests() {
  testing::LoadGameTest("start_at(history=0;1;0,game=kuhn_poker())");
  testing::LoadGameTest(
      "start_at(history=4;3;3;2;0;4;4;4;4;0,game=connect_four())");
  testing::RandomSimTest(
      *LoadGame("start_at(history=0;1,game=kuhn_poker())"), 100);
}

void StartsAtCorrectHistoryTest() {
  std::shared_ptr<const Game> game = LoadGame(
      "start_at(history=0;1;0;0,game=kuhn_poker())");
  std::unique_ptr<State> initial_state = game->NewInitialState();
  const StartAtTransformationState& state = open_spiel::down_cast<
      const StartAtTransformationState&>(*initial_state);

  const std::string expected_observation_string = "011";
  {
    const std::vector<Action> expected_history = {};
    SPIEL_CHECK_EQ(state.History(), expected_history);
    SPIEL_CHECK_EQ(state.ObservationString(0), expected_observation_string);
  }
  {
    const std::vector<Action> expected_history = {0, 1, 0, 0};
    SPIEL_CHECK_EQ(state.GetWrappedState().History(), expected_history);
    SPIEL_CHECK_EQ(state.GetWrappedState().ObservationString(0),
                   expected_observation_string);
  }
}


}  // namespace
}  // namespace start_at
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::start_at::BasicStartAtTests();
  open_spiel::start_at::StartsAtCorrectHistoryTest();
}
