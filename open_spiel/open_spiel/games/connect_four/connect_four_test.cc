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

#include "open_spiel/games/connect_four/connect_four.h"
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace connect_four {
namespace {

namespace testing = open_spiel::testing;

void BasicConnectFourTests() {
  testing::LoadGameTest("connect_four");
  testing::NoChanceOutcomesTest(*LoadGame("connect_four"));
  testing::RandomSimTest(*LoadGame("connect_four"), 100);
}

void FastLoss() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  auto state = game->NewInitialState();
  state->ApplyAction(3);
  state->ApplyAction(3);
  state->ApplyAction(4);
  state->ApplyAction(4);
  state->ApplyAction(2);
  state->ApplyAction(2);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(1);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
  SPIEL_CHECK_EQ(state->ToString(),
                 ".......\n"
                 ".......\n"
                 ".......\n"
                 ".......\n"
                 "..ooo..\n"
                 ".xxxx..\n");
}

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void CheckFullBoardDraw() {
  std::shared_ptr<const Game> game = LoadGame("connect_four");
  ConnectFourState state(game,
      "ooxxxoo\n"
      "xxoooxx\n"
      "ooxxxoo\n"
      "xxoooxx\n"
      "ooxxxoo\n"
      "xxoooxx\n");
  SPIEL_CHECK_EQ(state.ToString(),
                 "ooxxxoo\n"
                 "xxoooxx\n"
                 "ooxxxoo\n"
                 "xxoooxx\n"
                 "ooxxxoo\n"
                 "xxoooxx\n");
  SPIEL_CHECK_TRUE(state.IsTerminal());
  SPIEL_CHECK_EQ(state.Returns(), (std::vector<double>{0, 0}));
}

void TestStateStruct() {
  auto game = LoadGame("connect_four");
  auto state = game->NewInitialState();
  ConnectFourState* cf_state = static_cast<ConnectFourState*>(state.get());
  SPIEL_CHECK_EQ(cf_state->ToStruct()->ToJson(), cf_state->ToJson());
  cf_state->ApplyAction(3);
  cf_state->ApplyAction(4);
  std::string state_json =
      R"({"board":[)"
      R"([".",".",".","x","o",".","."],)"
      R"([".",".",".",".",".",".","."],)"
      R"([".",".",".",".",".",".","."],)"
      R"([".",".",".",".",".",".","."],)"
      R"([".",".",".",".",".",".","."],)"
      R"([".",".",".",".",".",".","."]],)"
      R"("current_player":"x","is_terminal":false,"winner":""})";
  SPIEL_CHECK_EQ(cf_state->ToJson(), state_json);
  cf_state->ApplyAction(3);
  cf_state->ApplyAction(4);
  cf_state->ApplyAction(3);
  cf_state->ApplyAction(4);
  cf_state->ApplyAction(3);
  state_json =
      R"({"board":[)"
      R"([".",".",".","x","o",".","."],)"
      R"([".",".",".","x","o",".","."],)"
      R"([".",".",".","x","o",".","."],)"
      R"([".",".",".","x",".",".","."],)"
      R"([".",".",".",".",".",".","."],)"
      R"([".",".",".",".",".",".","."]],)"
      R"("current_player":"Terminal","is_terminal":true,"winner":"x"})";
  SPIEL_CHECK_EQ(cf_state->ToJson(), state_json);
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 ConnectFourStateStruct(state_json).ToJson());
}

void ArbitrarySizeTests() {
  // Test on a 4x5 board
  std::shared_ptr<const Game> game_4x5 =
      LoadGame("connect_four",
               {{"rows", GameParameter(4)}, {"columns", GameParameter(5)}});
  SPIEL_CHECK_EQ(game_4x5->MaxGameLength(), 20);
  testing::RandomSimTest(*game_4x5, 10);

  // Check a win condition on a smaller board
  auto state = game_4x5->NewInitialState();
  // Vertical win for player 0 in column 0
  state->ApplyAction(0);  // x
  state->ApplyAction(1);  // o
  state->ApplyAction(0);  // x
  state->ApplyAction(1);  // o
  state->ApplyAction(0);  // x
  state->ApplyAction(1);  // o
  SPIEL_CHECK_EQ(state->ToString(),
                 ".....\n"
                 "xo...\n"
                 "xo...\n"
                 "xo...\n");
  state->ApplyAction(0);  // x
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
  SPIEL_CHECK_EQ(state->ToString(),
                 "x....\n"
                 "xo...\n"
                 "xo...\n"
                 "xo...\n");

  // Horizontal win on 5x6
  std::shared_ptr<const Game> game_5x6 =
      LoadGame("connect_four",
               {{"rows", GameParameter(5)}, {"columns", GameParameter(6)}});
  auto state_5x6 = game_5x6->NewInitialState();
  state_5x6->ApplyAction(0);  // x
  state_5x6->ApplyAction(0);  // o
  state_5x6->ApplyAction(1);  // x
  state_5x6->ApplyAction(1);  // o
  state_5x6->ApplyAction(2);  // x
  state_5x6->ApplyAction(2);  // o
  SPIEL_CHECK_FALSE(state_5x6->IsTerminal());
  state_5x6->ApplyAction(3);  // x wins
  SPIEL_CHECK_TRUE(state_5x6->IsTerminal());
  SPIEL_CHECK_EQ(state_5x6->Returns(), (std::vector<double>{1.0, -1.0}));
  SPIEL_CHECK_EQ(state_5x6->ToString(),
                 "......\n"
                 "......\n"
                 "......\n"
                 "ooo...\n"
                 "xxxx..\n");

  // Test on a 7x8 board
  std::shared_ptr<const Game> game_7x8 =
      LoadGame("connect_four",
               {{"rows", GameParameter(7)}, {"columns", GameParameter(8)}});
  SPIEL_CHECK_EQ(game_7x8->MaxGameLength(), 56);
  testing::RandomSimTest(*game_7x8, 10);

  // Test connect-5 on a 6x7 board.
  std::shared_ptr<const Game> game_c5 =
      LoadGame("connect_four", {{"x_in_row", GameParameter(5)}});
  auto state_c5 = game_c5->NewInitialState();
  // Vertical win for p0
  for (int i = 0; i < 4; ++i) {
    state_c5->ApplyAction(0);  // x
    state_c5->ApplyAction(1);  // o
  }
  SPIEL_CHECK_FALSE(state_c5->IsTerminal());
  state_c5->ApplyAction(0);  // x wins
  SPIEL_CHECK_TRUE(state_c5->IsTerminal());
  SPIEL_CHECK_EQ(state_c5->Returns(), (std::vector<double>{1.0, -1.0}));
}

}  // namespace
}  // namespace connect_four
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::connect_four::BasicConnectFourTests();
  open_spiel::connect_four::ArbitrarySizeTests();
  open_spiel::connect_four::FastLoss();
  open_spiel::connect_four::BasicSerializationTest();
  open_spiel::connect_four::CheckFullBoardDraw();
  open_spiel::connect_four::TestStateStruct();
}
