// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/twenty_forty_eight/2048.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace twenty_forty_eight {
namespace {

namespace testing = open_spiel::testing;

void BasicSimulationTests() { testing::RandomSimTest(*LoadGame("2048"), 100); }

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void RandomSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  for (int i = 0; i < 20; ++i) {
    std::cout << state->ToString() << std::endl;
    std::cout << state->LegalActions().size() << std::endl;
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void Basic2048Tests() {
  testing::LoadGameTest("2048");
  testing::ChanceOutcomesTest(*LoadGame("2048"));
  testing::RandomSimTest(*LoadGame("2048"), 100);
}

// Board:
//    0    0    0    0
//    2    0    0    0
//    2    0    0    0
//    2    0    0    0
// 4 should be formed in the bottom left corner and not on the cell above it
void MultipleMergePossibleTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwentyFortyEightState* cstate =
      static_cast<TwentyFortyEightState*>(state.get());
  cstate->SetCustomBoard({0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0});
  cstate->ApplyAction(kMoveDown);
  SPIEL_CHECK_EQ(cstate->BoardAt(3, 0).value, 4);
}

// Board:
//    2    4    0    4
//    0    2    0    2
//    0    0    0    0
//    0    2    0    0
// 4 should not be merged again with the newly formed 4 in 2nd column
void OneMergePerTurnTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwentyFortyEightState* cstate =
      static_cast<TwentyFortyEightState*>(state.get());
  cstate->SetCustomBoard({2, 4, 0, 4, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0});
  cstate->ApplyAction(kMoveDown);
  SPIEL_CHECK_EQ(cstate->BoardAt(2, 1).value, 4);
  SPIEL_CHECK_EQ(cstate->BoardAt(3, 1).value, 4);
}

// Board:
//    4    8    2    4
//    2    4    8   16
//   16  128   64  128
//    2    8    2    8
// This should be a terminal state
void TerminalStateTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwentyFortyEightState* cstate =
      static_cast<TwentyFortyEightState*>(state.get());
  cstate->SetCustomBoard(
      {4, 8, 2, 4, 2, 4, 8, 16, 16, 128, 64, 128, 2, 8, 2, 8});
  SPIEL_CHECK_EQ(cstate->IsTerminal(), true);
}

// Board:
//    4    8    2    4
//    2    4    8   16
// 1024  128   64  128
// 1024    8    2    8
// Taking down action should win from this state
void GameWonTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwentyFortyEightState* cstate =
      static_cast<TwentyFortyEightState*>(state.get());
  cstate->SetCustomBoard(
      {4, 8, 2, 4, 2, 4, 8, 16, 1024, 128, 64, 128, 1024, 8, 2, 8});
  cstate->ApplyAction(kMoveDown);
  SPIEL_CHECK_EQ(cstate->IsTerminal(), true);
  SPIEL_CHECK_EQ(cstate->Returns()[0], 2048);
}

// Board:
//    0    0    0    0
//    0    0    0    0
//    0    0    0    0
//    2    0    0    2
// Down should not be a legal action here as it does not change the board
void BoardNotChangedTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwentyFortyEightState* cstate =
      static_cast<TwentyFortyEightState*>(state.get());
  cstate->SetCustomBoard({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2});
  for (Action action : cstate->LegalActions()) {
    SPIEL_CHECK_NE(action, kMoveDown);
  }
}

}  // namespace
}  // namespace twenty_forty_eight
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::twenty_forty_eight::BasicSimulationTests();
  open_spiel::twenty_forty_eight::BasicSerializationTest();
  open_spiel::twenty_forty_eight::RandomSerializationTest();
  open_spiel::twenty_forty_eight::Basic2048Tests();
  open_spiel::twenty_forty_eight::MultipleMergePossibleTest();
  open_spiel::twenty_forty_eight::OneMergePerTurnTest();
  open_spiel::twenty_forty_eight::TerminalStateTest();
  open_spiel::twenty_forty_eight::GameWonTest();
  open_spiel::twenty_forty_eight::BoardNotChangedTest();
}
