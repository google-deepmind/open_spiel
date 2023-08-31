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

#include "open_spiel/games/mancala/mancala.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace mancala {
namespace {

namespace testing = open_spiel::testing;

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("mancala");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void BasicMancalaTests() {
  testing::LoadGameTest("mancala");
  testing::NoChanceOutcomesTest(*LoadGame("mancala"));
  testing::RandomSimTest(*LoadGame("mancala"), 100);
}

// Board:
// -0-0-0-4-0-0-
// 0-----------0
// -0-0-1-0-0-0-
// Player 0 taking action 3 should capture the opponents 4 beans
void CaptureWhenOppositePitNotEmptyTest() {
  std::shared_ptr<const Game> game = LoadGame("mancala");
  std::unique_ptr<State> state = game->NewInitialState();
  MancalaState* mstate = static_cast<MancalaState*>(state.get());
  mstate->SetBoard({0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0});

  // Check for exactly one legal move.
  std::vector<Action> legal_actions = mstate->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);

  // Check that it's 3
  SPIEL_CHECK_EQ(legal_actions[0], 3);

  mstate->ApplyAction(legal_actions[0]);
  // Check if Player 0 home pit has 5 beans
  SPIEL_CHECK_EQ(mstate->BoardAt(7), 5);
}

// Board:
// -0-0-0-0-4-0-
// 0-----------0
// -0-0-1-0-0-0-
// Player 0 taking action 3 should not result in any captures
void DoNotCaptureWhenOppositePitIsEmptyTest() {
  std::shared_ptr<const Game> game = LoadGame("mancala");
  std::unique_ptr<State> state = game->NewInitialState();
  MancalaState* mstate = static_cast<MancalaState*>(state.get());
  mstate->SetBoard({0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0});

  // Check for exactly one legal move.
  std::vector<Action> legal_actions = mstate->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);

  // Check that it's 3
  SPIEL_CHECK_EQ(legal_actions[0], 3);

  mstate->ApplyAction(legal_actions[0]);
  // Check if no capture has taken place
  SPIEL_CHECK_EQ(mstate->BoardAt(7), 0);
  SPIEL_CHECK_EQ(mstate->BoardAt(3), 0);
  SPIEL_CHECK_EQ(mstate->BoardAt(4), 1);
  SPIEL_CHECK_EQ(mstate->BoardAt(9), 4);
}

// Board:
// -0-0-0-0-0-1-
// 0-----------0
// -1-0-0-0-0-8-
// Player 0 taking action 6 should not put beans in opponents home pit
void DoNotAddBeanToOpponentsHomePitTest() {
  std::shared_ptr<const Game> game = LoadGame("mancala");
  std::unique_ptr<State> state = game->NewInitialState();
  MancalaState* mstate = static_cast<MancalaState*>(state.get());
  mstate->SetBoard({0, 1, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 0});

  // Check for exactly two legal move.
  std::vector<Action> legal_actions = mstate->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2);

  // Check that it's 1 & 6
  SPIEL_CHECK_EQ(legal_actions[0], 1);
  SPIEL_CHECK_EQ(legal_actions[1], 6);

  mstate->ApplyAction(legal_actions[1]);
  // Check if no bean is put into opponents home pit
  SPIEL_CHECK_EQ(mstate->BoardAt(0), 0);
  SPIEL_CHECK_EQ(mstate->BoardAt(7), 1);
  SPIEL_CHECK_EQ(mstate->BoardAt(8), 2);
  SPIEL_CHECK_EQ(mstate->BoardAt(1), 2);
}

}  // namespace
}  // namespace mancala
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::mancala::BasicSerializationTest();
  open_spiel::mancala::BasicMancalaTests();
  open_spiel::mancala::CaptureWhenOppositePitNotEmptyTest();
  open_spiel::mancala::DoNotCaptureWhenOppositePitIsEmptyTest();
  open_spiel::mancala::DoNotAddBeanToOpponentsHomePitTest();
}
