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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace dark_hex {
namespace {

namespace testing = open_spiel::testing;

void GameBlackWinWithCollisionAndObs() {
  std::shared_ptr<const Game> game =
      LoadGame("dark_hex", {{"board_size", GameParameter(3)},
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
  std::shared_ptr<const Game> game =
      LoadGame("dark_hex", {{"board_size", GameParameter(3)}});
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

void BasicDarkHexTests() {
  testing::LoadGameTest("dark_hex");
  testing::NoChanceOutcomesTest(*LoadGame("dark_hex"));
  testing::RandomSimTest(*LoadGame("dark_hex(board_size=5)"), 10);
  testing::LoadGameTest("dark_hex(obstype=reveal-numturns)");
  GameBlackWinWithCollisionAndObs();
  GameBlackWinsMaximumCollisions();
}

}  // namespace
}  // namespace dark_hex
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::dark_hex::BasicDarkHexTests(); }
