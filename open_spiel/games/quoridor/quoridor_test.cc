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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace quoridor {
namespace {

namespace testing = open_spiel::testing;

void BasicQuoridorTests() {
  testing::LoadGameTest("quoridor(board_size=5)");
  testing::NoChanceOutcomesTest(*LoadGame("quoridor()"));
  testing::RandomSimTest(*LoadGame("quoridor"), 10);

  for (int i = 5; i <= 13; i++) {
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("quoridor(board_size=", i, ")")), 5);
  }

  for (int i = 2; i <= 4; i++) {
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("quoridor(board_size=9,players=", i, ")")), 5);
  }

  testing::RandomSimTest(*LoadGame("quoridor(board_size=9,wall_count=5)"), 3);

  // Ansi colors!
  testing::RandomSimTest(
      *LoadGame("quoridor", {{"board_size", GameParameter(9)},
                             {"ansi_color_output", GameParameter(true)}}),
      3);
  testing::RandomSimTest(
      *LoadGame("quoridor", {{"board_size", GameParameter(9)},
                             {"ansi_color_output", GameParameter(true)},
                             {"players", GameParameter(3)}}),
      3);
  testing::RandomSimTest(
      *LoadGame("quoridor(board_size=5,ansi_color_output=True)"), 3);
  testing::RandomSimTest(
      *LoadGame("quoridor(board_size=5,ansi_color_output=True,players=3)"), 3);
}

// A 4-player game in which one pawn ends up fully boxed in with no walls
// left, so its only legal action is the forced pass. The pass must be encoded
// as the relative "stay in place" action (base_for_relative_, cell (2,2) in
// virtual coordinates, id 2 * 17 + 2 = 36 on a 9x9 board), not the pawn's
// absolute cell id: the absolute id gets decoded as a relative move, which
// teleports the pawn or indexes board_ out of bounds.
void ForcedPassIsRelativeTest() {
  std::shared_ptr<const Game> game = LoadGame("quoridor(players=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  const std::vector<Action> actions = {
      163, 215, 63,  87,  73,  199, 91,  141, 1,   267, 173, 185, 187, 195,
      15,  19,  2,   111, 51,  25,  29,  247, 109, 177, 99,  255, 221, 253,
      233, 137, 34,  81,  70,  133, 209, 57,  79,  159, 38,  11,  2,   70,
      241, 70,  70,  2,   38,  2,   167, 70,  123, 34,  227, 2,   34,  38,
      34,  2,   38,  34,  2,   2,   34,  38,  70,  38,  34,  34,  2,   38,
      34,  38,  70,  2,   34,  34,  38,  34,  38,  34,  34,  38,  38,  38,
      2,   34,  38,  34,  38,  38,  38,  38,  70,  70,  34,  34,  34,  34,
      34,  2,   34,  38,  70,  2,   2,   34,  34,  70,  2,   34,  70,  2,
      2,   70,  2,   70,  2,   2,   70,  2,   34,  38,  34,  70,  70,  68,
      70,  70,  2,   38,  70,  38,  0,   68};
  for (Action action : actions) {
    state->ApplyAction(action);
  }
  // Player 2's pawn is boxed in at a6 with no walls left: the only legal
  // action is the forced pass.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_EQ(legal.size(), 1);
  SPIEL_CHECK_EQ(legal[0], 36);
  std::string before = state->ToString();
  state->ApplyAction(legal[0]);
  // A pass must not move any pawn or spend a wall.
  SPIEL_CHECK_EQ(state->ToString(), before);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 3);
}

}  // namespace
}  // namespace quoridor
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::quoridor::BasicQuoridorTests();
  open_spiel::quoridor::ForcedPassIsRelativeTest();
}
