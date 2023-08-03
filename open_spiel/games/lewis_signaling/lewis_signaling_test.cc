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

#include "open_spiel/games/lewis_signaling.h"

#include <array>
#include <iostream>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace lewis_signaling {
namespace {

namespace testing = open_spiel::testing;

void BasicLewisSignalingTests() {
  testing::RandomSimTest(*LoadGame("lewis_signaling"), 100);
}

void DefaultParamsTest() {
  std::vector<double> def_pay = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < kDefaultNumStates; ++i) {
    for (int j = 0; j < kDefaultNumStates; ++j) {
      std::shared_ptr<const Game> game = LoadGame("lewis_signaling");
      std::unique_ptr<State> state = game->NewInitialState();

      state->ApplyAction(i);  // set state to i
      SPIEL_CHECK_TRUE(state->CurrentPlayer() ==
                       static_cast<int>(Players::kSender));
      state->ApplyAction(0);  // message 0
      SPIEL_CHECK_TRUE(state->CurrentPlayer() ==
                       static_cast<int>(Players::kReceiver));
      state->ApplyAction(j);  // action j
      SPIEL_CHECK_TRUE(state->IsTerminal());
      SPIEL_CHECK_EQ(state->PlayerReturn(0),
                     def_pay[i * kDefaultNumStates + j]);
      SPIEL_CHECK_EQ(state->PlayerReturn(1),
                     def_pay[i * kDefaultNumStates + j]);
      std::cout << state->ToString() << std::endl;
    }
  }
}

void LargePayoffMatrixTest() {
  std::vector<double> pay = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::string pay_str = "1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1";
  int num_states = 4;
  GameParameters params = {{"num_states", GameParameter(num_states)},
                           {"payoffs", GameParameter(pay_str)}};
  for (int i = 0; i < num_states; ++i) {
    for (int j = 0; j < num_states; ++j) {
      std::shared_ptr<const Game> game = LoadGame("lewis_signaling", params);
      std::unique_ptr<State> state = game->NewInitialState();

      state->ApplyAction(i);  // set state to i
      SPIEL_CHECK_TRUE(state->CurrentPlayer() ==
                       static_cast<int>(Players::kSender));
      state->ApplyAction(0);  // message 0
      SPIEL_CHECK_TRUE(state->CurrentPlayer() ==
                       static_cast<int>(Players::kReceiver));
      state->ApplyAction(j);  // action j
      SPIEL_CHECK_TRUE(state->IsTerminal());
      SPIEL_CHECK_EQ(state->PlayerReturn(0), pay[i * num_states + j]);
      SPIEL_CHECK_EQ(state->PlayerReturn(1), pay[i * num_states + j]);
      std::cout << state->ToString() << std::endl;
    }
  }
}

}  // namespace
}  // namespace lewis_signaling
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::lewis_signaling::BasicLewisSignalingTests();
  open_spiel::lewis_signaling::DefaultParamsTest();
  open_spiel::lewis_signaling::LargePayoffMatrixTest();
}
