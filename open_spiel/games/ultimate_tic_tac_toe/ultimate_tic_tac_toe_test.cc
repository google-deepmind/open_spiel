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

#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace ultimate_tic_tac_toe {
namespace {

namespace testing = open_spiel::testing;

void BasicUltimateTicTacToeTests() {
  testing::LoadGameTest("ultimate_tic_tac_toe");
  testing::NoChanceOutcomesTest(*LoadGame("ultimate_tic_tac_toe"));
  testing::RandomSimTest(*LoadGame("ultimate_tic_tac_toe"), 100);
}

// Regression: free-move `Choose local board` used to leave the target
// board's internal current player stale, so the following cell was
// placed for the wrong player. This 15-action sequence has X free-move
// into board 2 and place at (0,1); without the fix the cell is 'o'.
void FreeMoveUsesMetaCurrentPlayerRegression() {
  std::shared_ptr<const Game> game = LoadGame("ultimate_tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  const std::vector<Action> actions = {0, 3, 3, 7, 5, 0, 2, 0,
                                       1, 1, 3, 0, 0, 2, 1};
  for (Action a : actions) {
    state->ApplyAction(a);
  }
  // ToString()'s first line is "sub0row0 sub1row0 sub2row0"; index 9 is
  // board 2 row 0 col 1 -- the cell the last action targeted.
  const std::string s = state->ToString();
  SPIEL_CHECK_GE(s.size(), 11u);
  SPIEL_CHECK_EQ(s[9], 'x');
}

}  // namespace
}  // namespace ultimate_tic_tac_toe
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::ultimate_tic_tac_toe::BasicUltimateTicTacToeTests();
  open_spiel::ultimate_tic_tac_toe::FreeMoveUsesMetaCurrentPlayerRegression();
}
