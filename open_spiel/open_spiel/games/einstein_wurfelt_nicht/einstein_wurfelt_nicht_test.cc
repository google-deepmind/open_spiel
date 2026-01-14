// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/games/einstein_wurfelt_nicht/einstein_wurfelt_nicht.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace einstein_wurfelt_nicht {
namespace {

namespace testing = open_spiel::testing;

void BasicEinsteinWurfeltNitchTests() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  testing::RandomSimTest(*game, 100, true, true);
  testing::RandomSimTestWithUndo(*game, 1);
}

void BlackPlayerSimpleWinTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  EinsteinWurfeltNichtState* bstate =
      static_cast<EinsteinWurfeltNichtState*>(state.get());

  int values[] = {-1, 2,  -1, -1, -1, -1, -1, -1, 5,  -1, 6,  -1, -1,
                  -1, -1, -1, 3,  -1, -1, 3,  -1, -1, -1, -1, -1};
  Color colors[] = {Color::kEmpty, Color::kWhite, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kBlack, Color::kEmpty, Color::kBlack, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kWhite, Color::kEmpty, Color::kEmpty, Color::kBlack,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty};
  std::array<Cube, k2dMaxBoardSize> board;
  for (int i = 0; i < k2dMaxBoardSize; i++) {
    board[i] = {colors[i], values[i]};
  }

  bstate->SetState(kBlackPlayerId, 2, board, 3, 2);

  std::string expected_state =
      "|__||w2||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||w3||__||__||b3|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state);
  SPIEL_CHECK_EQ(bstate->CurrentPlayer(), kBlackPlayerId);
  SPIEL_CHECK_FALSE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(bstate->LegalActions().size(), 1);
  Action action = 230;  // Move B3 down
  SPIEL_CHECK_EQ(bstate->LegalActions()[0], action);
  SPIEL_CHECK_EQ(bstate->ActionToString(kBlackPlayerId, action), "B3-down");

  bstate->ApplyAction(230);
  std::string expected_state_final =
      "|__||w2||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||w3||__||__||__|\n"
      "|__||__||__||__||b3|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state_final);
  std::vector<double> returns = bstate->Returns();
  SPIEL_CHECK_TRUE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void WhitePlayerSimpleWinTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  EinsteinWurfeltNichtState* bstate =
      static_cast<EinsteinWurfeltNichtState*>(state.get());

  int values[] = {-1, 2,  -1, -1, -1, -1, -1, -1, 5,  -1, 6,  -1, -1,
                  -1, -1, -1, 3,  -1, -1, 3,  -1, -1, -1, -1, -1};
  Color colors[] = {Color::kEmpty, Color::kWhite, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kBlack, Color::kEmpty, Color::kBlack, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kWhite, Color::kEmpty, Color::kEmpty, Color::kBlack,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty};
  std::array<Cube, k2dMaxBoardSize> board;
  for (int i = 0; i < k2dMaxBoardSize; i++) {
    board[i] = {colors[i], values[i]};
  }
  bstate->SetState(kWhitePlayerId, 2, board, 3, 2);

  std::string expected_state =
      "|__||w2||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||w3||__||__||b3|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state);
  SPIEL_CHECK_EQ(bstate->CurrentPlayer(), kWhitePlayerId);
  SPIEL_CHECK_FALSE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(bstate->LegalActions().size(), 1);
  Action action = 22;  // Move W2 to the left
  SPIEL_CHECK_EQ(bstate->LegalActions()[0], action);
  SPIEL_CHECK_EQ(bstate->ActionToString(kWhitePlayerId, action), "W2-left");

  bstate->ApplyAction(action);
  std::string expected_state_final =
      "|w2||__||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||w3||__||__||b3|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state_final);
  std::vector<double> returns = bstate->Returns();
  SPIEL_CHECK_TRUE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0], -1);
  SPIEL_CHECK_EQ(returns[1], 1);
}

void WinByCapturingAllOpponentCubesTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  EinsteinWurfeltNichtState* bstate =
      static_cast<EinsteinWurfeltNichtState*>(state.get());

  int values[] = {-1, -1, -1, -1, -1, -1, -1, -1, 5,  -1, 6,  -1, -1,
                  -1, -1, -1, 3,  -1, -1, 3,  -1, -1, -1, -1, -1};
  Color colors[] = {Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kBlack, Color::kEmpty, Color::kBlack, Color::kEmpty,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kWhite, Color::kEmpty, Color::kEmpty, Color::kBlack,
                    Color::kEmpty, Color::kEmpty, Color::kEmpty, Color::kEmpty,
                    Color::kEmpty};
  std::array<Cube, k2dMaxBoardSize> board;
  for (int i = 0; i < k2dMaxBoardSize; i++) {
    board[i] = {colors[i], values[i]};
  }
  bstate->SetState(kBlackPlayerId, 6, board, 3, 1);

  std::string expected_state =
      "|__||__||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||w3||__||__||b3|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state);
  SPIEL_CHECK_EQ(bstate->CurrentPlayer(), kBlackPlayerId);
  SPIEL_CHECK_FALSE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(bstate->LegalActions().size(), 3);
  Action action = 121;  // Move B6 diagonally down-right
  SPIEL_CHECK_EQ(bstate->LegalActions()[0], action);
  SPIEL_CHECK_EQ(bstate->ActionToString(kBlackPlayerId, action), "B6-diag*");

  bstate->ApplyAction(action);
  std::string expected_state_final =
      "|__||__||__||__||__|\n"
      "|__||__||__||b5||__|\n"
      "|__||__||__||__||__|\n"
      "|__||b6||__||__||b3|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(bstate->ToString(), expected_state_final);
  std::vector<double> returns = bstate->Returns();
  SPIEL_CHECK_TRUE(bstate->IsTerminal());
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void CheckAlternateChancePlayerAndNormalPlayerTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  int previous_player = state->CurrentPlayer();

  while (!state->IsTerminal()) {
    if (state->CurrentPlayer() == open_spiel::kChancePlayerId) {
      state->ApplyAction(state->LegalActions()[0]);
    } else {
      std::vector<open_spiel::Action> legal_actions = state->LegalActions();
      state->ApplyAction(legal_actions[0]);
    }
    int current_player = state->CurrentPlayer();
    if (current_player != open_spiel::kChancePlayerId) {
      SPIEL_CHECK_NE(current_player, previous_player);
    }
    previous_player = current_player;
  }
}

void InitialStateTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), open_spiel::kChancePlayerId);
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void LegalActionsTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    std::vector<open_spiel::Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    state->ApplyAction(legal_actions[0]);
  }

  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_TRUE(returns[0] == 1.0 || returns[1] == 1.0);
}

void InitialBoardSetupTest() {
  // Test the initial setup with empty board
  std::string empty_board_state =
      "|__||__||__||__||__|\n"
      "|__||__||__||__||__|\n"
      "|__||__||__||__||__|\n"
      "|__||__||__||__||__|\n"
      "|__||__||__||__||__|\n";
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->ToString(), empty_board_state);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ChanceOutcomes().size(), kNumCubesPermutations);

  //  Test allocation of black cubes on the board
  state->ApplyAction(0);
  std::string black_board_state =
      "|b1||b2||b3||__||__|\n"
      "|b4||b5||__||__||__|\n"
      "|b6||__||__||__||__|\n"
      "|__||__||__||__||__|\n"
      "|__||__||__||__||__|\n";
  SPIEL_CHECK_EQ(state->ToString(), black_board_state);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ChanceOutcomes().size(), kNumCubesPermutations);

  //  Allocation of cubes on the board changes if a different action is applied
  std::shared_ptr<const open_spiel::Game> game2 =
      open_spiel::LoadGame("einstein_wurfelt_nicht");
  std::unique_ptr<open_spiel::State> state2 = game->NewInitialState();
  SPIEL_CHECK_EQ(state2->ToString(), empty_board_state);
  state2->ApplyAction(1);
  SPIEL_CHECK_NE(state2->ToString(), empty_board_state);
  SPIEL_CHECK_NE(state->ToString(), state2->ToString());

  //  Test allocation of white cubes on the board
  state->ApplyAction(0);
  std::string white_board_state =
      "|b1||b2||b3||__||__|\n"
      "|b4||b5||__||__||__|\n"
      "|b6||__||__||__||w1|\n"
      "|__||__||__||w2||w3|\n"
      "|__||__||w4||w5||w6|\n";
  SPIEL_CHECK_EQ(state->ToString(), white_board_state);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ChanceOutcomes().size(), kNumPlayerCubes);
}

}  // namespace
}  // namespace einstein_wurfelt_nicht
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("einstein_wurfelt_nicht");
  open_spiel::einstein_wurfelt_nicht::BasicEinsteinWurfeltNitchTests();
  open_spiel::einstein_wurfelt_nicht::WinByCapturingAllOpponentCubesTest();
  open_spiel::einstein_wurfelt_nicht::
      CheckAlternateChancePlayerAndNormalPlayerTest();
  open_spiel::einstein_wurfelt_nicht::InitialStateTest();
  open_spiel::einstein_wurfelt_nicht::LegalActionsTest();
  open_spiel::einstein_wurfelt_nicht::BlackPlayerSimpleWinTest();
  open_spiel::einstein_wurfelt_nicht::WhitePlayerSimpleWinTest();
  open_spiel::einstein_wurfelt_nicht::WinByCapturingAllOpponentCubesTest();
  open_spiel::einstein_wurfelt_nicht::InitialBoardSetupTest();
}
