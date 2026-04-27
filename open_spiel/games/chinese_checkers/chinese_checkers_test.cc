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

#include "open_spiel/games/chinese_checkers/chinese_checkers.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chinese_checkers {
namespace {

namespace testing = open_spiel::testing;

void BasicChineseCheckersTests() {
  testing::LoadGameTest("chinese_checkers");
  testing::NoChanceOutcomesTest(*LoadGame("chinese_checkers"));
  // Use shorter games for practical test runtime.
  auto game = LoadGame("chinese_checkers", {{"max_moves", GameParameter(200)}});
  testing::RandomSimTest(*game, 100);
  testing::RandomSimTestWithUndo(*game, 10);
}

void MultiPlayerTests() {
  for (int players : {3, 4, 6}) {
    auto game = LoadGame("chinese_checkers",
                         {{"players", GameParameter(players)},
                          {"max_moves", GameParameter(100)}});
    SPIEL_CHECK_TRUE(game != nullptr);
    testing::RandomSimTest(*game, 50);
    testing::RandomSimTestWithUndo(*game, 5);
  }
}

void GameParametersTest() {
  auto game = LoadGame("chinese_checkers",
                       {{"players", GameParameter(3)},
                        {"max_moves", GameParameter(100)}});
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_EQ(game->NumPlayers(), 3);
  testing::RandomSimTest(*game, 10);
}

void InitialBoardTest() {
  auto game = LoadGame("chinese_checkers");
  auto state = game->NewInitialState();
  auto* cs = static_cast<ChineseCheckersState*>(state.get());

  // Player 0 occupies triangle 0 (North, positions 0-9).
  for (int i = 0; i < kTriangleSize; ++i) {
    SPIEL_CHECK_EQ(cs->BoardAt(kTriangleCells[0][i]), 0);
  }
  // Player 1 occupies triangle 3 (South, positions 111-120).
  for (int i = 0; i < kTriangleSize; ++i) {
    SPIEL_CHECK_EQ(cs->BoardAt(kTriangleCells[3][i]), 1);
  }
  // Center of the board should be empty.
  SPIEL_CHECK_EQ(cs->BoardAt(60), kEmpty);
}

void HopChainTest() {
  auto game = LoadGame("chinese_checkers");
  auto state = game->NewInitialState();
  auto* cs = static_cast<ChineseCheckersState*>(state.get());

  // Clear the board.
  for (int i = 0; i < kNumPositions; ++i) {
    cs->SetBoard(i, kEmpty);
  }

  // Set up a horizontal hop chain scenario on row 8 (center row).
  // Positions on row 8: 56(col4) 57(col6) 58(col8) 59(col10) 60(col12)
  // 61(col14) 62(col16) 63(col18) 64(col20)
  // Place player 0's piece at 56, obstacle pieces at 57 and 59.
  cs->SetBoard(56, 0);
  cs->SetBoard(57, 1);
  cs->SetBoard(59, 1);
  // Keep player 1 with a piece somewhere (required for non-terminal state).
  cs->SetBoard(120, 1);

  // Player 0 hops from 56 over 57 to 58 (direction R = 3).
  Action hop1 = 56 * kNumDirections + 3;
  auto legal = state->LegalActions();
  bool found = false;
  for (Action a : legal) {
    if (a == hop1) { found = true; break; }
  }
  SPIEL_CHECK_TRUE(found);
  state->ApplyAction(hop1);

  // Should still be player 0's turn (mid-chain). Pass should be available.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  legal = state->LegalActions();
  bool has_pass = false;
  bool has_continuation = false;
  for (Action a : legal) {
    if (a == kPassAction) {
      has_pass = true;
    } else {
      has_continuation = true;
    }
  }
  SPIEL_CHECK_TRUE(has_pass);
  SPIEL_CHECK_TRUE(has_continuation);

  // Continue: hop from 58 over 59 to 60.
  Action hop2 = 58 * kNumDirections + 3;
  state->ApplyAction(hop2);
  // No more hops available from 60 in the right direction (61 is empty).
  // Turn should auto-advance to player 1.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void WinConditionTest() {
  auto game = LoadGame("chinese_checkers");
  auto state = game->NewInitialState();
  auto* cs = static_cast<ChineseCheckersState*>(state.get());

  // Clear all pieces.
  for (int i = 0; i < kNumPositions; ++i) {
    cs->SetBoard(i, kEmpty);
  }

  // Place 9 of player 0's pieces already in target triangle 3 (South).
  // Target triangle 3 cells: 111,112,113,114,115,116,117,118,119,120
  for (int i = 0; i < 9; ++i) {
    cs->SetBoard(kTriangleCells[3][i], 0);
  }
  // Place the 10th piece near the target (position 105, adjacent to 113).
  cs->SetBoard(105, 0);

  // Place player 1's pieces somewhere.
  cs->SetBoard(60, 1);

  // Player 0 steps piece from 105 to 113 (direction DR = 5).
  // 105 is at (12, 14), neighbor in direction 5 (DR: +1,+1) = (13, 15) =
  // pos 114.
  // Wait, let me check: kNeighbor[105][5] should give the right neighbor.
  // Actually, target cell 120 is at the tip. Let me use a simpler setup.
  // Place 10th piece at cell 106 (row 12, col 16). kNeighbor[106][5] = ?
  // Let me just use the 10th target cell instead.
  // Place 9 pieces in cells 111-119 and the 10th at a neighbor of cell 120.
  for (int i = 0; i < kNumPositions; ++i) {
    cs->SetBoard(i, kEmpty);
  }

  // Target for player 0 (triangle 3): cells 111-120.
  // Fill cells 111-119 with player 0's pieces.
  for (int i = 0; i < 9; ++i) {
    cs->SetBoard(kTriangleCells[3][i], 0);
  }
  // Cell 120 is (16,12). Its neighbors are 118(15,11) and 119(15,13).
  // Both are already occupied by player 0. So we need to place the 10th piece
  // at a position that can step INTO cell 120.
  // kNeighbor[120] = {118, 119, -1, -1, -1, -1}.
  // So 118 and 119 are neighbors of 120. But they're occupied.
  // Let's leave cell 119 empty and place the 10th piece there instead.
  cs->SetBoard(kTriangleCells[3][8], kEmpty);  // Remove from cell 119.
  cs->SetBoard(kTriangleCells[3][9], 0);       // Place in cell 120.
  // Now place 10th piece at cell 119's neighbor that can step to 119.
  // Cell 119 = (15,13). Neighbors: 116(14,12), 117(14,14), 118(15,11), -1,
  // 120(16,12), -1.
  // 116 is occupied by player 0. Let's use 117 instead.
  // Actually let's simplify: put 9 pieces in target, 1 piece adjacent.
  for (int i = 0; i < kNumPositions; ++i) {
    cs->SetBoard(i, kEmpty);
  }
  // Fill target cells 0-8 (of triangle 3) with player 0.
  for (int i = 0; i < 9; ++i) {
    cs->SetBoard(kTriangleCells[3][i], 0);
  }
  // kTriangleCells[3] = {111,112,113,114,115,116,117,118,119,120}
  // Leave cell 120 (index 9) empty and place piece at neighbor of 120.
  // kNeighbor[120] = {118, 119, -1, -1, -1, -1}
  // 118 is in the target (occupied). Use 119 which is occupied too.
  // So both neighbors of 120 are occupied by player 0.
  // Let me instead leave cell 111 empty and approach it.
  for (int i = 0; i < kNumPositions; ++i) {
    cs->SetBoard(i, kEmpty);
  }
  // Fill cells 112-120 with player 0.
  for (int i = 1; i < kTriangleSize; ++i) {
    cs->SetBoard(kTriangleCells[3][i], 0);
  }
  // Cell 111 = (13,9). kNeighbor[111] = {102,103,-1,112,-1,115}
  // Place piece at 102. Step direction 5 (DR: +1,+1) goes to (13,9)=111.
  // kNeighbor[102][5] = 111 based on the topology.
  cs->SetBoard(102, 0);
  // Place player 1 somewhere.
  cs->SetBoard(60, 1);

  // Player 0 should be able to step from 102 to 111 and win.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  auto legal = state->LegalActions();
  // Find the step action: 102 in direction 5 (DR).
  Action step = 102 * kNumDirections + 5;
  bool found = false;
  for (Action a : legal) {
    if (a == step) { found = true; break; }
  }
  SPIEL_CHECK_TRUE(found);
  state->ApplyAction(step);

  // Game should now be terminal with player 0 winning.
  SPIEL_CHECK_TRUE(state->IsTerminal());
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1.0);   // Winner gets num_players - 1 = 1.
  SPIEL_CHECK_EQ(returns[1], -1.0);  // Loser gets -1.
}

}  // namespace
}  // namespace chinese_checkers
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::chinese_checkers::BasicChineseCheckersTests();
  open_spiel::chinese_checkers::MultiPlayerTests();
  open_spiel::chinese_checkers::GameParametersTest();
  open_spiel::chinese_checkers::InitialBoardTest();
  open_spiel::chinese_checkers::HopChainTest();
  open_spiel::chinese_checkers::WinConditionTest();
}
