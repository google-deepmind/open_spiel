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

#include "open_spiel/games/amazons/amazons.h"

#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace amazons {
namespace {

namespace testing = open_spiel::testing;

void BasicSpielTests() {
  testing::LoadGameTest("amazons");
  testing::RandomSimTest(*LoadGame("amazons"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("amazons"), 5);
}

// Test the given configuration for player 1 win:
// Player 1 = Cross, Player 2 = Nought
// |O # X _ ### ...|
void PlayerOneSimpleWinTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  std::array<CellState, kNumCells> board = {};
  for (int i = 0; i < board.size(); i++) {
    board[i] = CellState::kBlock;
  }
  board[0] = CellState::kNought;
  board[2] = CellState::kCross;
  board[3] = CellState::kEmpty;

  astate->SetState(1, AmazonsState::MoveState::amazon_select, board);

  std::cout << "PlayerOneWinTest: \n" << astate->ToString() << "\n";

  SPIEL_CHECK_TRUE(astate->LegalActions().empty());

  std::cout << "Success!"
            << "\n\n";
}

// Test the given configuration for player 2 win:
// Player 1 = Cross, Player 2 = Nought
// |X # O _ ### ...|
void PlayerTwoSimpleWinTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  std::array<CellState, kNumCells> board = {};
  for (int i = 0; i < board.size(); i++) {
    board[i] = CellState::kBlock;
  }
  board[0] = CellState::kCross;
  board[2] = CellState::kNought;
  board[3] = CellState::kEmpty;

  astate->SetState(0, AmazonsState::MoveState::amazon_select, board);

  std::cout << "PlayerTwoWinTest: \n" << astate->ToString() << "\n";

  SPIEL_CHECK_TRUE(astate->LegalActions().empty());

  std::cout << "Success!"
            << "\n\n";
}

// Test given configuration for player 1 no moves
// .......
// ..OOO..
// ..OXO..
// ..OOO..
// .......
void PlayerOneTrappedByAmazonsTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  std::array<CellState, kNumCells> board = {};
  for (int i = 0; i < board.size(); i++) {
    board[i] = CellState::kEmpty;
  }
  int center = kNumCells / 2 + kNumRows / 2;
  board[center] = CellState::kCross;
  board[center - 1] = board[center + 1] = CellState::kNought;
  board[center - kNumRows] = board[center - kNumRows - 1] =
      board[center - kNumRows + 1] = CellState::kNought;
  board[center + kNumRows] = board[center + kNumRows - 1] =
      board[center + kNumRows + 1] = CellState::kNought;

  astate->SetState(0, AmazonsState::MoveState::amazon_select, board);

  std::cout << "PlayerOneTrappedByAmazonsTest: \n"
            << astate->ToString() << "\n";

  SPIEL_CHECK_TRUE(astate->LegalActions().empty());

  std::cout << "Success!"
            << "\n\n";
}
// Test given configuration for player 1 no moves
// .......
// ..###..
// ..#X#..
// ..###..
// .......
void PlayerOneTrappedByBlocksTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  std::array<CellState, kNumCells> board = {};
  for (int i = 0; i < board.size(); i++) {
    board[i] = CellState::kEmpty;
  }
  int center = kNumCells / 2 + kNumRows / 2;
  board[center] = CellState::kCross;
  board[center - 1] = board[center + 1] = CellState::kBlock;
  board[center - kNumRows] = board[center - kNumRows - 1] =
      board[center - kNumRows + 1] = CellState::kBlock;
  board[center + kNumRows] = board[center + kNumRows - 1] =
      board[center + kNumRows + 1] = CellState::kBlock;

  astate->SetState(0, AmazonsState::MoveState::amazon_select, board);

  std::cout << "PlayerOneTrappedByBlocksTest: \n" << astate->ToString() << "\n";

  SPIEL_CHECK_TRUE(astate->LegalActions().empty());

  std::cout << "Success!"
            << "\n\n";
}

// Verifies multi‑phase move generation from an initial layout, destination/shot
// rays with proper blocking, and correct ActionToString formatting.
void PhasedMoveGenerationAndStringsTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  // Initial state basics.
  SPIEL_CHECK_EQ(astate->CurrentPlayer(), 0);
  SPIEL_CHECK_FALSE(astate->IsTerminal());

  // ActionToString for a known square.
  std::string s = astate->ActionToString(0, 60);  // a4
  SPIEL_CHECK_EQ(s, "X From (7, 1)");

  // amazon_select: only the four X amazons are selectable.
  std::vector<Action> expected_selects = {60, 69, 93, 96};
  std::vector<Action> selects = astate->LegalActions();
  SPIEL_CHECK_EQ(selects.size(), expected_selects.size());
  SPIEL_CHECK_TRUE(std::equal(selects.begin(), selects.end(),
  expected_selects.begin()));

  // Select d1 (93) -> destination_select
  state->ApplyAction(93);
  std::vector<Action> expected_dests = {
  13, 23, 33, 43, 48, 53, 57, 63, 66,
  71, 73, 75, 82, 83, 84, 90, 91, 92, 94, 95};
  std::vector<Action> dests = astate->LegalActions();
  SPIEL_CHECK_EQ(dests.size(), expected_dests.size());
  SPIEL_CHECK_TRUE(std::equal(dests.begin(), dests.end(),
  expected_dests.begin()));

  // Move to f1 (95) -> shot_select
  state->ApplyAction(95);
  // Shot string formatting check (shoot back to d1).
  std::string shot_str = astate->ActionToString(0, 93);
  SPIEL_CHECK_EQ(shot_str, "X Shoot: (10, 4)");

  // Exact shots from f1.
  std::vector<Action> expected_shots = {
  5, 15, 25, 35, 40, 45, 51, 55, 59, 62, 65, 68,
  73, 75, 77, 84, 85, 86, 90, 91, 92, 93, 94};
  std::vector<Action> shots = astate->LegalActions();
  SPIEL_CHECK_EQ(shots.size(), expected_shots.size());
  SPIEL_CHECK_TRUE(std::equal(shots.begin(), shots.end(),
  expected_shots.begin()));
}

// Verifies undo across shot, destination, and selection fully restores
// the board, current player, and legal actions to the initial state.
void UndoRoundTripTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  // Full move: select d1 (93) -> to f1 (95) -> shoot d1 (93)
  state->ApplyAction(93);
  state->ApplyAction(95);
  state->ApplyAction(93);

  // After shot: f1 is X, d1 is block, player switches to 1.
  SPIEL_CHECK_EQ(astate->BoardAt(95), CellState::kCross);
  SPIEL_CHECK_EQ(astate->BoardAt(93), CellState::kBlock);
  SPIEL_CHECK_EQ(astate->BoardAt(96), CellState::kCross);  // other X untouched
  SPIEL_CHECK_EQ(astate->CurrentPlayer(), 1);

  // Undo shot.
  state->UndoAction(0, 93);
  SPIEL_CHECK_EQ(astate->BoardAt(93), CellState::kEmpty);
  SPIEL_CHECK_EQ(astate->CurrentPlayer(), 0);

  // Undo destination.
  state->UndoAction(0, 95);
  SPIEL_CHECK_EQ(astate->BoardAt(95), CellState::kEmpty);

  // Undo amazon select: X back on d1.
  state->UndoAction(0, 93);
  SPIEL_CHECK_EQ(astate->BoardAt(93), CellState::kCross);
  SPIEL_CHECK_EQ(astate->CurrentPlayer(), 0);

  // Initial selectable amazons restored.
  std::vector<Action> expected_selects = {60, 69, 93, 96};
  std::vector<Action> selects = astate->LegalActions();
  SPIEL_CHECK_EQ(selects.size(), expected_selects.size());
  SPIEL_CHECK_TRUE(std::equal(selects.begin(), selects.end(),
  expected_selects.begin()));
}
// Verifies the game becomes terminal only after the shot and Returns()
//  correctly awards the win when the next player has no legal moves.
void TerminalAndReturnsOnShotTest() {
  std::shared_ptr<const Game> game = LoadGame("amazons");
  std::unique_ptr<State> state = game->NewInitialState();
  AmazonsState* astate = static_cast<AmazonsState*>(state.get());

  // Only a single X on an otherwise empty board.
  std::array<CellState, kNumCells> board;
  board.fill(CellState::kEmpty);
  board[55] = CellState::kCross;
  astate->SetState(0, AmazonsState::MoveState::amazon_select, board);

  SPIEL_CHECK_FALSE(astate->IsTerminal());
  std::vector<Action> selects = astate->LegalActions();
  SPIEL_CHECK_EQ(selects.size(), 1);
  SPIEL_CHECK_EQ(selects[0], 55);

  // Select -> move -> still not terminal
  state->ApplyAction(55);
  SPIEL_CHECK_FALSE(astate->IsTerminal());

  // Move right one step
  state->ApplyAction(56);
  SPIEL_CHECK_FALSE(astate->IsTerminal());

  // Shoot left
  state->ApplyAction(54);

  // Now opponent (O) has no amazons -> terminal; X wins.
  SPIEL_CHECK_TRUE(astate->IsTerminal());
  SPIEL_CHECK_EQ(astate->CurrentPlayer(), kTerminalPlayerId);
  auto returns = astate->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0], 1.0);
  SPIEL_CHECK_EQ(returns[1], -1.0);
}


}  // namespace
}  // namespace amazons
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::amazons::BasicSpielTests();
  open_spiel::amazons::PlayerOneSimpleWinTest();
  open_spiel::amazons::PlayerTwoSimpleWinTest();
  open_spiel::amazons::PlayerOneTrappedByAmazonsTest();
  open_spiel::amazons::PlayerOneTrappedByBlocksTest();
  open_spiel::amazons::PhasedMoveGenerationAndStringsTest();
  open_spiel::amazons::UndoRoundTripTest();
  open_spiel::amazons::TerminalAndReturnsOnShotTest();
}
