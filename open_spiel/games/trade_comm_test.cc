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

#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace trade_comm {
namespace {

namespace testing = open_spiel::testing;

void BasicTradeCommTests() {
  testing::RandomSimTest(*LoadGame("trade_comm"), 100);
}

void SuccessfulTradeTest() {
  std::shared_ptr<const Game> game = LoadGame("trade_comm");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(26);  // first player gets item 2, second gets 6
  state->ApplyAction(1);   // Utterance 1
  state->ApplyAction(8);   // Utterance 8
  state->ApplyAction(10 + 9 + 9 + 5);                  // giving 2 for 6
  state->ApplyAction(10 + 9 + 9 + 9 + 9 + 9 + 9 + 2);  // giving 6 for 2
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1.0);
}

}  // namespace
}  // namespace trade_comm
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::trade_comm::BasicTradeCommTests();
  open_spiel::trade_comm::SuccessfulTradeTest();
}
