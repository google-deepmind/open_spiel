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

#include "open_spiel/games/go.h"

#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace go {
namespace {

namespace testing = open_spiel::testing;

constexpr int kBoardSize = 19;
constexpr float kKomi = 7.5;

void BasicGoTests() {
  GameParameters params;
  params["board_size"] = GameParameter(13);

  testing::LoadGameTest("go");
  testing::NoChanceOutcomesTest(*LoadGame("go"));
  testing::RandomSimTest(*LoadGame("go", params), 3);
  testing::RandomSimTestWithUndo(*LoadGame("go", params), 3);
}

void HandicapTest() {
  std::shared_ptr<const Game> game =
      LoadGame("go", {{"board_size", open_spiel::GameParameter(kBoardSize)},
                      {"komi", open_spiel::GameParameter(kKomi)},
                      {"handicap", open_spiel::GameParameter(2)}});
  GoState state(game, kBoardSize, kKomi, 2);
  SPIEL_CHECK_EQ(state.CurrentPlayer(), ColorToPlayer(GoColor::kWhite));
  SPIEL_CHECK_EQ(state.board().PointColor(MakePoint("d4")), GoColor::kBlack);
  SPIEL_CHECK_EQ(state.board().PointColor(MakePoint("q16")), GoColor::kBlack);
}

void ConcreteActionsAreUsedInTheAPI() {
  int board_size = 13;
  std::shared_ptr<const Game> game =
      LoadGame("go", {{"board_size", open_spiel::GameParameter(board_size)}});
  std::unique_ptr<State> state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->NumDistinctActions(), board_size * board_size + 1);
  SPIEL_CHECK_EQ(state->LegalActions().size(), state->NumDistinctActions());
  for (Action action : state->LegalActions()) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LE(action, board_size * board_size);
  }
}

}  // namespace
}  // namespace go
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::go::BasicGoTests();
  open_spiel::go::HandicapTest();
  open_spiel::go::ConcreteActionsAreUsedInTheAPI();
}
