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

#include "open_spiel/games/checkers/checkers.h"

#include "open_spiel/abseil-cpp/absl/types/optional.h"
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

void RandomSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("checkers");
  std::unique_ptr<State> state = game->NewInitialState();
  for (int i = 0; i < 20; ++i) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void BasicCheckersTests() {
  testing::LoadGameTest("checkers");
  testing::NoChanceOutcomesTest(*LoadGame("checkers"));
  testing::RandomSimTest(*LoadGame("checkers"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("checkers"), 10);

  // 10x10 Board
  testing::RandomSimTest(
      *LoadGame("checkers",
                {{"rows", GameParameter(10)}, {"columns", GameParameter(10)}}),
      100);
  testing::RandomSimTestWithUndo(
      *LoadGame("checkers",
                {{"rows", GameParameter(10)}, {"columns", GameParameter(10)}}),
      10);

  // 12x12 Board
  testing::RandomSimTest(
      *LoadGame("checkers",
                {{"rows", GameParameter(12)}, {"columns", GameParameter(12)}}),
      100);
  testing::RandomSimTestWithUndo(
      *LoadGame("checkers",
                {{"rows", GameParameter(12)}, {"columns", GameParameter(12)}}),
      10);

  auto observer = LoadGame("checkers")
                      ->MakeObserver(absl::nullopt,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("checkers"), observer);
}

// Board:
// 8........
// 7..*.....
// 6........
// 5....+.o.
// 4.....o..
// 3+.......
// 2...+....
// 1o.o.....
//  abcdefgh
// Player 0 should be able to do a double jump and crown a piece at b8
void MultipleJumpTest() {
  std::shared_ptr<const Game> game = LoadGame("checkers");
  std::unique_ptr<State> state = game->NewInitialState();
  CheckersState* cstate = static_cast<CheckersState*>(state.get());
  cstate->SetCustomBoard(
      "0..........*.................+.o......o..+..........+....o.o.....");
  cstate->ApplyAction(cstate->LegalActions()[0]);
  // Confirm that player 0 is given only one action (f4 token is in the middle
  // of a multiple jump) and there's a capture opportunity for c1 piece as well
  // (which cannot be moved in this extra move)
  SPIEL_CHECK_EQ(cstate->LegalActions().size(), 1);
  cstate->ApplyAction(cstate->LegalActions()[0]);
  SPIEL_CHECK_EQ(cstate->BoardAt(0, 1), CellState::kWhiteKing);
  SPIEL_CHECK_EQ(cstate->BoardAt(1, 2), CellState::kEmpty);
  SPIEL_CHECK_EQ(cstate->BoardAt(3, 4), CellState::kEmpty);
}

// Board:
// 8...8....
// 7........
// 6........
// 5....+...
// 4........
// 3+.......
// 2........
// 1........
//  abcdefgh
// Player 0 should be able to move the crowned piece backwards
void CrownedPieceCanMoveBackwardsTest() {
  std::shared_ptr<const Game> game = LoadGame("checkers");
  std::unique_ptr<State> state = game->NewInitialState();
  CheckersState* cstate = static_cast<CheckersState*>(state.get());
  cstate->SetCustomBoard(
      "0...8........................+...........+.......................");
  std::vector<Action> legal_actions = cstate->LegalActions();
  cstate->ApplyAction(legal_actions[0]);
  SPIEL_CHECK_EQ(cstate->BoardAt(1, 4), CellState::kWhiteKing);
}

// Board:
// 8........
// 7....+.+.
// 6........
// 5....+.o.
// 4.....o..
// 3+.......
// 2........
// 1o.o.....
//  abcdefgh
// Player 0 move should end after piece crowned
void MoveShouldEndAfterPieceCrownedTest() {
  std::shared_ptr<const Game> game = LoadGame("checkers");
  std::unique_ptr<State> state = game->NewInitialState();
  CheckersState* cstate = static_cast<CheckersState*>(state.get());
  cstate->SetCustomBoard(
      "0............+.+.............+.o......o..+...............o.o.....");
  cstate->ApplyAction(cstate->LegalActions()[0]);
  cstate->ApplyAction(cstate->LegalActions()[0]);
  SPIEL_CHECK_EQ(cstate->CurrentPlayer(), 1);
}

}  // namespace
}  // namespace checkers
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::checkers::BasicSerializationTest();
  open_spiel::checkers::RandomSerializationTest();
  open_spiel::checkers::BasicCheckersTests();
  open_spiel::checkers::MultipleJumpTest();
  open_spiel::checkers::CrownedPieceCanMoveBackwardsTest();
  open_spiel::checkers::MoveShouldEndAfterPieceCrownedTest();
}
