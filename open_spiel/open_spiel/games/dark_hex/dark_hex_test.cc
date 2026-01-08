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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace dark_hex {
namespace {

namespace testing = open_spiel::testing;

void GameBlackWinWithCollisionAndObs() {
  std::shared_ptr<const Game> game =
      LoadGame("dark_hex", {{"num_cols", GameParameter(3)},
                            {"num_rows", GameParameter(3)},
                            {"obstype", GameParameter("reveal-numturns")}});
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> lm = state->LegalActions();  // initial legal moves
  // . . .
  //  . . .
  //   . . .
  state->ApplyAction(4);
  // . . .
  //  . B .  B represents black-stone
  //   . . .
  // Check White's possible moves before rejection
  SPIEL_CHECK_EQ(state->LegalActions(), lm);
  state->ApplyAction(4);
  // . . .
  //  . R .  R represents rejection
  //   . . .
  // Check White's possible moves after rejection
  lm.erase(std::remove(lm.begin(), lm.end(), 4), lm.end());
  SPIEL_CHECK_EQ(state->LegalActions(), lm);
  // . . .
  //  . B .  W represents white-stone
  //   . W .
  state->ApplyAction(7);
  // . . .
  //  . B .  Black now knows the whites move
  //   . R .
  state->ApplyAction(7);
  // Check blacks info on number of turns
  std::string str = state->ObservationString(state->CurrentPlayer());
  SPIEL_CHECK_EQ(str.back(), '4');
  // . . .
  //  . B .
  //   B W .
  state->ApplyAction(6);
  // . . W
  //  . B .
  //   B W .
  state->ApplyAction(2);
  // . B W
  //  . B .
  //   B W .
  state->ApplyAction(1);
  // Black wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void GameBlackWinsMaximumCollisions() {
  // White follows exact moves black is playing, black does the same
  // B . .   R . .   B W .   B R .   B W B
  //  . . .   . . .   . . .   . . .   . . .    ...
  //   . . .   . . .   . . .   . . .   . . .
  std::shared_ptr<const Game> game = LoadGame(
      "dark_hex",
      {{"num_cols", GameParameter(3)}, {"num_rows", GameParameter(3)}});
  std::unique_ptr<State> state = game->NewInitialState();
  std::array play_seq = {0, 1, 4, 2, 7, 5, 8, 6};  // 3 is the terminal move
  for (int i = 0; i < play_seq.size(); ++i) {
    state->ApplyAction(play_seq[i]);  // player moves
    state->ApplyAction(play_seq[i]);  // other player tries to move
  }
  state->ApplyAction(3);
  // Black wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void GameUnevenBoardBlackWin() {
  std::shared_ptr<const Game> game = LoadGame(
      "dark_hex",
      {{"num_cols", GameParameter(4)}, {"num_rows", GameParameter(3)}});
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(8);
  state->ApplyAction(5);
  state->ApplyAction(4);
  state->ApplyAction(1);
  state->ApplyAction(0);
  std::cout << state->ObservationString(0) << std::endl;
  std::cout << state->ObservationString(1) << std::endl;
  // Black wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void GameUnevenBoardWhiteWin() {
  std::shared_ptr<const Game> game = LoadGame(
      "dark_hex",
      {{"num_cols", GameParameter(4)}, {"num_rows", GameParameter(3)}});
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(8);
  state->ApplyAction(5);
  state->ApplyAction(9);
  state->ApplyAction(4);
  state->ApplyAction(10);
  state->ApplyAction(6);
  state->ApplyAction(11);
  state->ApplyAction(7);
  std::cout << state->ObservationString(0) << std::endl;
  std::cout << state->ObservationString(1) << std::endl;
  // White wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), -1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1.0);
}

void ClassicalDarkHexTests() {
  testing::LoadGameTest("dark_hex");
  testing::NoChanceOutcomesTest(*LoadGame("dark_hex"));
  testing::RandomSimTest(*LoadGame("dark_hex(num_cols=5,num_rows=5)"), 10);
  testing::LoadGameTest("dark_hex(obstype=reveal-numturns)");
  GameBlackWinWithCollisionAndObs();
  GameBlackWinsMaximumCollisions();
  GameUnevenBoardBlackWin();
  GameUnevenBoardWhiteWin();
}

void AbruptDHCustomTest() {
  std::shared_ptr<const Game> game =
      LoadGame("dark_hex", {{"num_cols", GameParameter(2)},
                            {"num_rows", GameParameter(2)},
                            {"gameversion", GameParameter("adh")}});
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0);
  state->ApplyAction(0);
  state->ApplyAction(2);
  // Black wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void AbruptDarkHexTests() {
  testing::LoadGameTest("dark_hex(gameversion=adh)");
  testing::NoChanceOutcomesTest(*LoadGame("dark_hex(gameversion=adh)"));
  testing::RandomSimTest(
      *LoadGame("dark_hex(num_cols=3,num_rows=3,gameversion=adh)"), 3);
  AbruptDHCustomTest();
}

}  // namespace
}  // namespace dark_hex
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dark_hex::ClassicalDarkHexTests();
  open_spiel::dark_hex::AbruptDarkHexTests();
}
