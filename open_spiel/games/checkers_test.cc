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

#include "open_spiel/games/checkers.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace checkers {
namespace {

namespace testing = open_spiel::testing;

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("checkers");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void BasicCheckersTests() {
  testing::LoadGameTest("checkers");
  testing::NoChanceOutcomesTest(*LoadGame("checkers"));
  testing::RandomSimTest(*LoadGame("checkers"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("checkers"), 10);
}

// Board:
// 8........
// 7..*.....
// 6........
// 5....+.o.
// 4.....o..
// 3+.......
// 2........
// 1o.o.....
//  abcdefgh
// Player 0 should only have moves to do a double jump and crown a piece at b8 
void MultipleJumpTest() {
  std::shared_ptr<const Game> checkers = LoadGame("checkers(rows=8,columns=8)");
  CheckersState cstate(checkers, 8, 8, "0..........*.................+.o......o..+...............o.o.....");
  
  cstate.ApplyAction(cstate.LegalActions()[0]);
  cstate.ApplyAction(cstate.LegalActions()[0]);
  SPIEL_CHECK_EQ(cstate.BoardAt(0, 1), CellState::kWhiteCrowned);
}

}  // namespace
}  // namespace checkers
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::checkers::BasicSerializationTest();
  open_spiel::checkers::BasicCheckersTests();
  open_spiel::checkers::MultipleJumpTest();
}
