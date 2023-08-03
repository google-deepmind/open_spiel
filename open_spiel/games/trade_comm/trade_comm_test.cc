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

#include "open_spiel/games/trade_comm.h"

#include <array>
#include <iostream>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace trade_comm {
namespace {

namespace testing = open_spiel::testing;

void BasicTradeCommTests() {
  testing::RandomSimTest(*LoadGame("trade_comm"), 100);
}

void SuccessfulTradeDifferentItemsTest() {
  std::shared_ptr<const Game> game = LoadGame("trade_comm");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(26);  // allocate: first player gets item 2, second gets 6
  state->ApplyAction(1);   // Utterance 1
  state->ApplyAction(8);   // Utterance 8
  state->ApplyAction(10 + 2 * 10 + 6);  // giving 2 for 6
  state->ApplyAction(10 + 6 * 10 + 2);  // giving 6 for 2
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1.0);
}

void SuccessfulTradeSameItemsTest() {
  std::shared_ptr<const Game> game = LoadGame("trade_comm");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(33);  // allocate: first player gets item 3, second gets 3
  state->ApplyAction(2);   // Utterance 2
  state->ApplyAction(8);   // Utterance 8
  state->ApplyAction(10 + 3 * 10 + 3);  // giving 3 for 3
  state->ApplyAction(10 + 3 * 10 + 3);  // giving 3 for 3
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1.0);
}

void UnsuccessfulTradesTest() {
  std::shared_ptr<const Game> game = LoadGame("trade_comm");

  // P0 gets item 7, p1 gets item 1.
  // Only successful trade is {7, 1, 1, 7} which corresponds to p0 plays 7:1
  // and p1 plays 1:7
  for (std::array<int, 4> trade : std::vector<std::array<int, 4>>({
           // Format: { p0 giving, p0 getting, p1 giving, p0 getting }
           {0, 1, 1, 7},  // p0 mismatching the give
           {7, 2, 1, 7},  // p0 mismatching the get
           {7, 1, 3, 7},  // p1 mismatching the give
           {7, 1, 1, 4}   // p1 mismatching the get
       })) {
    std::unique_ptr<State> state = game->NewInitialState();
    state->ApplyAction(71);  // first player gets item 7, second gets 1
    state->ApplyAction(0);   // Utterance 0
    state->ApplyAction(6);   // Utterance 6
    state->ApplyAction(10 + trade[0] * 10 + trade[1]);
    state->ApplyAction(10 + trade[2] * 10 + trade[3]);
    std::cout << state->ToString() << std::endl;
    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->PlayerReturn(0), 0.0);
    SPIEL_CHECK_EQ(state->PlayerReturn(1), 0.0);
  }
}

}  // namespace
}  // namespace trade_comm
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::trade_comm::BasicTradeCommTests();
  open_spiel::trade_comm::SuccessfulTradeDifferentItemsTest();
  open_spiel::trade_comm::SuccessfulTradeSameItemsTest();
  open_spiel::trade_comm::UnsuccessfulTradesTest();
}
