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

#include "open_spiel/games/amazons.h"

#include <algorithm>
#include <random>

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

}  // namespace
}  // namespace amazons
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::amazons::BasicSpielTests();

  // These tests check whether certain board configurations indicate the correct
  // number of moves
  open_spiel::amazons::PlayerOneSimpleWinTest();
  open_spiel::amazons::PlayerTwoSimpleWinTest();
  open_spiel::amazons::PlayerOneTrappedByAmazonsTest();
  open_spiel::amazons::PlayerOneTrappedByBlocksTest();
}
