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

void TestActionStruct() {
  auto game = LoadGame("connect_four");
  auto state = game->NewInitialState();

  // Test ActionToStruct
  for (Action action = 0; action < 7; ++action) {
    auto action_struct = state->ActionToStruct(0, action);
    auto* cf_action =
        dynamic_cast<ConnectFourActionStruct*>(action_struct.get());
    SPIEL_CHECK_TRUE(cf_action != nullptr);
    SPIEL_CHECK_EQ(cf_action->column, action);

    // Test round-trip: ActionToStruct -> StructToActions
    std::vector<Action> actions = state->StructToActions(*action_struct);
    SPIEL_CHECK_EQ(actions.size(), 1);
    SPIEL_CHECK_EQ(actions[0], action);
  }

  // Test JSON round-trip
  ConnectFourActionStruct action_struct;
  action_struct.column = 3;
  std::string json = action_struct.ToJson();
  SPIEL_CHECK_EQ(json, R"({"column":3})");
  ConnectFourActionStruct parsed(json);
  SPIEL_CHECK_EQ(parsed.column, 3);
}

void TestSetStateFromStruct() {
  auto game = LoadGame("connect_four");

  // Test 1: Valid non-terminal state
  {
    ConnectFourStateStruct state_struct;
    state_struct.board = {{".", ".", ".", "x", "o", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."}};
    state_struct.current_player = "x";
    state_struct.is_terminal = false;
    state_struct.winner = "";

    auto state = static_cast<const ConnectFourGame*>(game.get())
                     ->NewInitialState(state_struct);
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  }

  // Test 2: Valid terminal state (X wins)
  {
    ConnectFourStateStruct state_struct;
    state_struct.board = {{".", ".", ".", "x", "o", ".", "."},
                          {".", ".", ".", "x", "o", ".", "."},
                          {".", ".", ".", "x", "o", ".", "."},
                          {".", ".", ".", "x", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."}};
    state_struct.current_player = "Terminal";
    state_struct.is_terminal = true;
    state_struct.winner = "x";

    auto state = static_cast<const ConnectFourGame*>(game.get())
                     ->NewInitialState(state_struct);
    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
  }

  // Test 3: Valid draw state
  {
    ConnectFourStateStruct state_struct;
    state_struct.board = {{"o", "o", "x", "x", "x", "o", "o"},
                          {"x", "x", "o", "o", "o", "x", "x"},
                          {"o", "o", "x", "x", "x", "o", "o"},
                          {"x", "x", "o", "o", "o", "x", "x"},
                          {"o", "o", "x", "x", "x", "o", "o"},
                          {"x", "x", "o", "o", "o", "x", "x"}};
    state_struct.current_player = "Terminal";
    state_struct.is_terminal = true;
    state_struct.winner = "draw";

    auto state = static_cast<const ConnectFourGame*>(game.get())
                     ->NewInitialState(state_struct);
    SPIEL_CHECK_TRUE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{0, 0}));
  }

  // Test 4: JSON round-trip for NewInitialState
  {
    auto state = game->NewInitialState();
    state->ApplyAction(3);
    state->ApplyAction(4);
    std::string json = state->ToJson();

    auto state2 = game->NewInitialState(nlohmann::json::parse(json));
    SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
  }
}



void TestGameParamsStruct() {
  // Test default params
  {
    ConnectFourGameParams params;
    SPIEL_CHECK_EQ(params.game_name, "connect_four");
    SPIEL_CHECK_EQ(params.rows, 6);
    SPIEL_CHECK_EQ(params.columns, 7);
    SPIEL_CHECK_EQ(params.x_in_row, 4);
    SPIEL_CHECK_EQ(params.egocentric_obs_tensor, false);

    auto game = LoadGame(params);
    SPIEL_CHECK_EQ(game->GetType().short_name, "connect_four");
    auto* cf_game = static_cast<const ConnectFourGame*>(game.get());
    SPIEL_CHECK_EQ(cf_game->rows(), 6);
    SPIEL_CHECK_EQ(cf_game->cols(), 7);
  }

  // Test custom params
  {
    ConnectFourGameParams params;
    params.rows = 8;
    params.columns = 9;
    params.x_in_row = 5;

    auto game = LoadGame(params);
    auto* cf_game = static_cast<const ConnectFourGame*>(game.get());
    SPIEL_CHECK_EQ(cf_game->rows(), 8);
    SPIEL_CHECK_EQ(cf_game->cols(), 9);
    SPIEL_CHECK_EQ(cf_game->x_in_row(), 5);
  }

  // Test JSON serialization
  {
    ConnectFourGameParams params;
    params.rows = 5;
    params.columns = 6;
    std::string json = params.ToJson();
    // NOLINTBEGIN
    SPIEL_CHECK_TRUE(json.find("\"rows\":5") != std::string::npos);
    SPIEL_CHECK_TRUE(json.find("\"columns\":6") != std::string::npos);
    SPIEL_CHECK_TRUE(json.find("\"game_name\":\"connect_four\"") !=
                     std::string::npos);
    // NOLINTEND
  }

  // Test LoadGameFromJson equivalent functionality
  {
    auto game = LoadGameFromJson(
        R"({"game_name":"connect_four","rows":4,"columns":5,"x_in_row":3})");
    auto* cf_game = static_cast<const ConnectFourGame*>(game.get());
    SPIEL_CHECK_EQ(cf_game->rows(), 4);
    SPIEL_CHECK_EQ(cf_game->cols(), 5);
    SPIEL_CHECK_EQ(cf_game->x_in_row(), 3);
  }
}

void TestPermissiveValidation() {
  auto game = LoadGame("connect_four");
  const auto* cf_game = static_cast<const ConnectFourGame*>(game.get());

  {
    ConnectFourStateStruct state_struct;
    state_struct.board = {{"x", "x", "x", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."},
                          {".", ".", ".", ".", ".", ".", "."}};
    state_struct.current_player = "o";
    state_struct.is_terminal = false;
    state_struct.winner = "";

    auto state = cf_game->NewInitialState(state_struct, false);
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  }

  {
    ConnectFourStateStruct state_struct;
    state_struct.board = {
        {"x", "o", ".", ".", ".", ".", "."},
        {".", ".", ".", ".", ".", ".", "."},
        {".", ".", ".", ".", ".", ".", "."},
        {".", ".", ".", ".", ".", ".", "."},
        {".", ".", ".", ".", ".", ".", "."},
        {".", ".", ".", ".", ".", ".", "."}};
    state_struct.current_player = "o";
    state_struct.is_terminal = false;
    state_struct.winner = "";

    auto state = cf_game->NewInitialState(state_struct, false);
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  }
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
  open_spiel::connect_four::TestActionStruct();
  open_spiel::connect_four::TestSetStateFromStruct();
  open_spiel::connect_four::TestPermissiveValidation();
  open_spiel::connect_four::TestGameParamsStruct();
}
