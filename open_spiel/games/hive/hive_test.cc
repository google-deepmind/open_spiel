// Copyright 2025 DeepMind Technologies Limited
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

#include "open_spiel/games/hive/hive.h"

#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/hive/hive_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace hive {
namespace {

namespace testing = open_spiel::testing;

// Move generation tests for each bug type. Modified from the original set
// to account for invalid move test cases.
// https://github.com/edre/nokamute/blob/master/data/uhp_tests.txt
constexpr const char* queen_test_game =
    "Base+MLP;InProgress;White[9];wS1;bS1 wS1-;wQ -wS1;bQ bS1-;wQ \\wS1;bQ "
    "bS1/;wG1 /wQ;bG1 bQ\\;wB1 \\wQ;bB1 bQ/;wG2 wB1/;bG2 \\bB1;wA1 /wB1;bA1 "
    "bB1\\;wQ wG2\\;bA2 bA1/";
constexpr const char* queen_valid_moves = "wQ wG2-;wQ -bG2;wQ wG1/";
constexpr const char* queen_invalid_moves = "wQ -bB1;wQ -bQ;wQ \\bS1";

constexpr const char* ant_test_game =
    "Base+MLP;InProgress;White[13];wS1;bB1 wS1-;wQ -wS1;bQ bB1-;wB1 \\wQ;bG1 "
    "bQ/;wB2 \\wB1;bG2 bG1/;wS2 \\wB2;bS1 bG2/;wA1 \\wS1;bB2 bS1/;wA2 "
    "\\wS2;bG3 \\bB2;wA1 -bG1;bA1 \\bG3;wG1 wA2/;bS2 -bA1;wG2 wG1/;bA2 "
    "-bS2;wA3 wG2-;bA3 bS2\\;wG3 wA3\\;bA3 wG3\\";
constexpr const char* ant_valid_moves =
    "wA1 -bG2;wA1 -bS1;wA1 /bG3;wA1 bS2\\;wA1 bA2\\;wA1 /bA2;wA1 bA3-;wA1 "
    "bA3\\;wA1 /bA3;wA1 /wG3;wA1 wG2\\;wA1 wG1\\;wA1 wB2/;wA1 wB1/;wA1 "
    "\\wS1;wA1 \\bB1";
constexpr const char* ant_invalid_moves = "wA1 -bA2;wA1 wA3-";

constexpr const char* grasshopper_test_game =
    "Base+MLP;InProgress;White[11];wG1;bG1 wG1-;wQ /wG1;bQ bG1-;wS1 wQ\\;bA1 "
    "bQ-;wB1 /wS1;bA1 -wQ;wB1 wS1\\;bA2 bQ-;wB1 /wS1;bA2 wG1\\;wB1 wS1\\;bA3 "
    "bQ-;wB1 /wS1;bS1 bQ\\;wB1 wS1;bS1 wB1\\;wB1 /wB1;bA3 -wB1";
constexpr const char* grasshopper_valid_moves =
    "wG1 /wQ;wG1 bA2\\;wG1 bQ-;wG1 \\wB1";
constexpr const char* grasshopper_invalid_moves = "wG1 \\bG1;wG1 bA1/";

constexpr const char* spider_test_game =
    "Base+MLP;InProgress;White[12];wG1;bA1 wG1-;wS1 \\wG1;bQ bA1-;wQ /wG1;bG1 "
    "bQ\\;wG2 wQ\\;bB1 /bG1;wB1 /wG2;bG2 bG1\\;wG3 /wB1;bG2 -bB1;wB2 wG3\\;bA1 "
    "bG1\\;wA1 wB2-;bA1 bB1\\;wA2 wA1/;bA1 bG1-;wS2 wA2-;bA1 bG1\\;wA3 "
    "wS2\\;bA1 wA3-";
constexpr const char* spider_valid_moves = "wS1 \\bQ;wS1 /bQ;wS1 wG1\\;wS1 /wQ";
constexpr const char* spider_invalid_moves = "wS1 -bQ;wS1 wG1/";

constexpr const char* beetle_test_game =
    "Base+MLP;InProgress;White[16];wB1;bB1 wB1-;wQ \\wB1;bQ bB1/;wG1 /wB1;bB2 "
    "bB1\\;wA1 /wG1;bA1 bQ\\;wG2 -wA1;bQ \\bB1;wB2 /wG2;bA2 \\bA1;wG3 "
    "wB2\\;bA2 \\wQ;wA2 wG3-;bB2 wB1\\;wS1 wA2\\;bA1 bB1\\;wS2 wS1-;bA1 "
    "bB1-;wA3 wS2/;bA1 \\wA3;wL -wG1;bM bB1\\;wA1 wG2\\;bM bB2;wA1 wL\\;bB1 "
    "bQ;wL bB1\\;bA1 -wG1";
constexpr const char* beetle_valid_moves = "wB1 wQ;wB1 bB1;wB1 wG1;wB1 bM";
constexpr const char* beetle_invalid_moves = "wB1 bQ;wB1 bB2;wB1 wL;wB1 /wQ";

constexpr const char* mosquito_test_game =
    "Base+MLP;InProgress;White[13];wM;bG1 wM-;wS1 /wM;bQ bG1-;wQ /wS1;bB1 "
    "bG1\\;wB1 /wQ;bB1 wM\\;wS2 /wB1;bA1 bQ-;wB2 wS2\\;bA1 bQ\\;wG1 wB2-;bA1 "
    "bQ-;wG2 wG1/;bA1 bQ\\;wG3 wG2/;bA1 bQ-;wA1 wG3-;bA1 bQ/;wA2 wA1-;bA1 "
    "bQ-;wA3 wA2\\;bA1 /wA3";
constexpr const char* mosquito_valid_moves =
    "wM bQ-;wM bB1\\;wM /wS2;wM \\bG1;wM bG1;wM bB1;wM wS1;wM \\wS1;wM bQ/;wM "
    "-wQ";
constexpr const char* mosquito_invalid_moves = "wM /wA2;wM \\bQ";

constexpr const char* ladybug_test_game =
    "Base+MLP;InProgress;White[15];wL;bL wL/;wQ -wL;bQ bL/;wQ -bL;bA1 bQ/;wB1 "
    "\\wQ;bA1 bQ-;wS1 \\wB1;bA1 bQ/;wB2 \\wS1;bA1 bQ-;wS2 wB2/;bA1 bQ/;wA1 "
    "wS2-;bA1 bQ-;wG1 wA1/;bA1 bQ/;wG2 wG1-;bA1 bQ-;wA2 wG2\\;bA1 bQ/;wA3 "
    "wA2-;bA1 bQ-;wG3 wA3/;bA1 \\wG3;wL bL\\;bQ \\bL";
constexpr const char* ladybug_valid_moves =
    "wL wB1/;wL /wB1;wL wS1-;wL \\bQ;wL bQ/;wL bQ-;wL /wQ;wL wQ\\";
constexpr const char* ladybug_invalid_moves = "wL /wS1;wL bL-";

constexpr const char* pillbug_test_game =
    "Base+MLP;InProgress;White[6];wP;bM wP-;wQ \\wP;bL bM\\;wA1 /wQ;bQ bL/;wA2 "
    "-wQ;bA1 /bL;wA2 wP\\;bM wP/";
constexpr const char* pillbug_valid_moves = "wQ -wA2;wQ -bQ;wA1 bM\\";
constexpr const char* pillbug_invalid_moves = "bM wA1\\;wP -bQ;wQ bM/";

// game states to test basic functionality
constexpr const char* white_wins_game =
    "Base;WhiteWins;Black[8];wS1;bS1 wS1-;wQ -wS1;bQ bS1/;wG1 -wQ;bG1 \\bQ;wG1 "
    "bQ\\;bG2 bQ/;wA1 wQ\\;bA1 bG2/;wA1 bG2\\;bA1 \\bG2;wQ \\wS1;bA1 bG2/;wQ "
    "/bG1";
constexpr const char* white_wins_on_black_turn_game =
    "Base;WhiteWins;White[7];wS1;bS1 wS1-;wQ -wS1;bQ bS1/;wG1 -wQ;bG1 \\bQ;wG1 "
    "bQ\\;bG2 bQ/;wA1 wQ\\;bA1 bG2/;wA1 bG2\\;bA1 /bG1";
constexpr const char* draw_game =
    "Base;Draw;White[11];wS1;bS1 wS1-;wQ -wS1;bQ bS1/;wG1 -wQ;bG1 \\bQ;wG1 "
    "bQ\\;bG2 bQ/;wA1 wQ\\;bA1 bG2/;wA1 bG2\\;bA1 \\bG2;wQ \\wS1;bG1 wA1/;wQ "
    "-bQ;bA1 \\wQ;wB1 -wS1;bG3 bG1-;wB1 /bA1;bG3 -bG2";
constexpr const char* force_pass_game =
    "Base;InProgress;White[7];wA1;bS1 wA1-;wQ -wA1;bQ bS1/;wQ \\wA1;bA1 "
    "bS1\\;wQ -wA1;bA2 bQ\\;wQ \\wA1;bA1 \\wQ;wG1 /wQ;bA2 /wG1";

// uncommonly encountered corner-cases
constexpr const char* beetle_gate_game =
    "Base;InProgress;White[12];wB1;bS1 wB1-;wQ \\wB1;bQ bS1/;wB2 -wQ;bB1 "
    "bQ\\;wS1 /wB2;bB1 bS1;wG1 /wS1;bQ \\bB1;wG2 wG1\\;bB2 bQ/;wG3 wG2\\;bB2 "
    "\\bQ;wA1 wG3-;bB2 wQ;wA2 wA1-;bA1 bQ/;wS2 wA2-;bA1 bB1/;wA3 wS2/;bA1 wA3/";
constexpr const char* beetle_gate_valid_moves =
    "wB1 bB2;wB1 bB1;wB1 /bB1;wB1 wB2\\";

constexpr const char* ladybug_gate_game =
    "Base+L;InProgress;White[14];wL;bG1 wL/;wQ -wL;bQ bG1/;wQ -bG1;bG2 bQ-;wB1 "
    "\\wQ;bB1 bG2-;wS1 \\wB1;bB1 bG2;wS2 \\wS1;bG3 \\bQ;wG1 wS2/;bB2 bG3/;wB2 "
    "wG1/;bB2 bG3;wA1 wB2-;bA1 bB1-;wA2 wA1-;bA1 bB1\\;wG2 wA2-;bA1 bB1-;wG3 "
    "wG2\\;bA1 bB1\\;wA3 wG3\\;bA1 wA3\\";
constexpr const char* ladybug_gate_valid_moves =
    "wL -bB2;wL /bB2;wL /wB1;wL /wS1;wL bQ\\;wL bG1\\;wL /wQ";

constexpr const char* pillbug_gate_game =
    "Base+P;InProgress;White[9];wP;bB1 wP-;wQ /wP;bQ bB1/;wQ wP\\;bQ \\bB1;wQ "
    "/wP;bA1 bQ/;wQ wP\\;bA1 -bQ;wQ /wP;bB2 \\bQ;wQ wP\\;bB2 bQ;bA1 -wP;bB1 wQ";
constexpr const char* pillbug_gate_valid_moves = "bA1 -bB2;bA1 /wP";

void BasicHiveTests() {
  testing::LoadGameTest("hive");
  std::shared_ptr<const open_spiel::Game> game_mlp =
      open_spiel::LoadGame("hive");
  testing::NoChanceOutcomesTest(*game_mlp);
  testing::RandomSimTest(*game_mlp, 5);

  // test all win conditions
  auto state = DeserializeUHPGameAndState(white_wins_game).second;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerWhite), 1.0);
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerBlack), -1.0);

  state = DeserializeUHPGameAndState(white_wins_on_black_turn_game).second;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerWhite), 1.0);
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerBlack), -1.0);

  state = DeserializeUHPGameAndState(draw_game).second;
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerWhite), 0.0);
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(kPlayerBlack), 0.0);

  // pass action
  state = DeserializeUHPGameAndState(force_pass_game).second;
  SPIEL_CHECK_TRUE(state->LegalActions().size() == 1);
  SPIEL_CHECK_TRUE(state->LegalActions().at(0) ==
                   state->StringToAction("pass"));

  // test all expansion variations
  testing::RandomSimTest(*LoadGame("hive(uses_mosquito=false)"), 1);
  testing::RandomSimTest(
      *LoadGame("hive(uses_mosquito=false,uses_ladybug=false)"), 1);
  testing::RandomSimTest(
      *LoadGame("hive(uses_mosquito=false,uses_pillbug=false)"), 1);
  testing::RandomSimTest(
      *LoadGame("hive(uses_ladybug=false,uses_pillbug=false)"), 1);
  testing::RandomSimTest(
      *LoadGame(
          "hive(uses_mosquito=false,uses_ladybug=false,uses_pillbug=false)"),
      1);

  // test with maximum board size
  testing::RandomSimTest(
      *LoadGame(absl::StrFormat("hive(board_size=%d)", kMaxBoardRadius)), 1);

  // test prettyprint with ansi colours
  testing::RandomSimTest(*LoadGame("hive(ansi_color_output=true)"), 1);
}

void TestMoves(std::unique_ptr<State>&& state, const char* valid_moves,
               const char* invalid_moves) {
  std::vector<int> legal_action_mask = state->LegalActionsMask();
  std::vector<std::string> valid_move_list =
      absl::StrSplit(valid_moves, ';', absl::SkipEmpty());
  std::vector<std::string> invalid_move_list =
      absl::StrSplit(invalid_moves, ';', absl::SkipEmpty());

  for (auto& move : valid_move_list) {
    SPIEL_CHECK_TRUE(legal_action_mask[state->StringToAction(move)] == 1);
  }

  for (auto& move : invalid_move_list) {
    SPIEL_CHECK_TRUE(legal_action_mask[state->StringToAction(move)] == 0);
  }
}

void TestBugMoves() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("hive");

  // Base Bugs
  TestMoves(DeserializeUHPGameAndState(queen_test_game).second,
            queen_valid_moves, queen_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(ant_test_game).second, ant_valid_moves,
            ant_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(grasshopper_test_game).second,
            grasshopper_valid_moves, grasshopper_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(spider_test_game).second,
            spider_valid_moves, spider_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(beetle_test_game).second,
            beetle_valid_moves, beetle_invalid_moves);

  // Expansion Bugs
  TestMoves(DeserializeUHPGameAndState(mosquito_test_game).second,
            mosquito_valid_moves, mosquito_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(ladybug_test_game).second,
            ladybug_valid_moves, ladybug_invalid_moves);
  TestMoves(DeserializeUHPGameAndState(pillbug_test_game).second,
            pillbug_valid_moves, pillbug_invalid_moves);

  // Special Cases
  TestMoves(DeserializeUHPGameAndState(beetle_gate_game).second,
            beetle_gate_valid_moves, "");
  TestMoves(DeserializeUHPGameAndState(ladybug_gate_game).second,
            ladybug_gate_valid_moves, "");
  TestMoves(DeserializeUHPGameAndState(pillbug_gate_game).second,
            pillbug_gate_valid_moves, "");
}

}  // namespace
}  // namespace hive
}  // namespace open_spiel

int main(int argc, char** argv) {
  // TODO: perft()
  open_spiel::hive::BasicHiveTests();
  open_spiel::hive::TestBugMoves();
}
