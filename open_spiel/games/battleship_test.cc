// Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
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

#include "open_spiel/games/battleship.h"

#include <iostream>

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace battleship {
namespace {

namespace testing = open_spiel::testing;

void BasicBattleshipTest() {
  testing::LoadGameTest("battleship");
  testing::NoChanceOutcomesTest(*LoadGame("battleship"));
  testing::RandomSimTestWithUndo(*LoadGame("battleship"), 100);

  for (int num_shots = 1; num_shots <= 3; ++num_shots) {
    const auto game =
        LoadGame("battleship", {{"board_width", GameParameter(2)},
                                {"board_height", GameParameter(2)},
                                {"ship_sizes", GameParameter("[1;2]")},
                                {"ship_values", GameParameter("[1;2]")},
                                {"num_shots", GameParameter(num_shots)},
                                {"allow_repeated_shots", GameParameter(false)},
                                {"loss_multiplier", GameParameter(2.0)}});
    testing::RandomSimTestWithUndo(*game, 100);
    return;
  }
}

void TestZeroSumTrait() {
  // We check that when the loss multiplier is 1.0, the game is registered as
  // zero sum.
  auto game = LoadGame("battleship", {{"loss_multiplier", GameParameter(2.0)}});
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kGeneralSum);

  game = LoadGame("battleship", {{"loss_multiplier", GameParameter(1.0)}});
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kZeroSum);
}

void TestTightLayout1() {
  // We construct a 4x1 grid with 2 ships of length 2 each. We want to make sure
  // that the the first ship is not placed at the center of the board.

  const auto game =
      LoadGame("battleship", {{"board_width", GameParameter(4)},
                              {"board_height", GameParameter(1)},
                              {"ship_sizes", GameParameter("[2;2]")},
                              {"ship_values", GameParameter("[1;1]")}});
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0, 2}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[1]), "h_0_2");
  }

  // We now place the first ship to the left, which corresponds to the
  // serialized index 0 as checked above.
  state->ApplyAction(0);

  // We repeat the check for Player 1.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0, 2}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[1]), "h_0_2");
  }

  // We place Player 1's ship to the right.
  state->ApplyAction(Action{2});

  // Now, we need to check that the only remaining action for Player 0 is to
  // place the second ship to the right.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({2}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_2");
  }
  state->ApplyAction(Action{2});

  //... While for Player 1 the only remaining action is to place the ship to the
  // left.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
  }
  state->ApplyAction(Action{0});

  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void TestTightLayout2() {
  // We construct a 2x3 grid with 2 ships of length 2 and 3 respectively. We
  // want to make sure that the the first ship is not placed anywhere
  // vertically.

  const auto game =
      LoadGame("battleship", {{"board_width", GameParameter(3)},
                              {"board_height", GameParameter(2)},
                              {"ship_sizes", GameParameter("[2;3]")},
                              {"ship_values", GameParameter("[1;1]")}});
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0, 1, 3, 4}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[1]), "h_0_1");
    SPIEL_CHECK_EQ(state->ActionToString(actions[2]), "h_1_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[3]), "h_1_1");
  }

  // We now place the first ship to the right on the first row, which
  // corresponds to the serialized index 1 as checked above.
  state->ApplyAction(Action{1});

  // We repeat the check for Player 1.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0, 1, 3, 4}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[1]), "h_0_1");
    SPIEL_CHECK_EQ(state->ActionToString(actions[2]), "h_1_0");
    SPIEL_CHECK_EQ(state->ActionToString(actions[3]), "h_1_1");
  }

  // We place Player 1's ship to the left on the second row.
  state->ApplyAction(Action{3});

  // Now, we need to check that the only remaining action for Player 0 is to
  // place the second ship on the second row.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({3}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_1_0");
  }
  state->ApplyAction(Action{3});

  //... While for Player 1 the only remaining action is to place the second ship
  // on the first row.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const auto actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({0}));
    SPIEL_CHECK_EQ(state->ActionToString(actions[0]), "h_0_0");
  }
  state->ApplyAction(Action{0});

  SPIEL_CHECK_FALSE(state->IsTerminal());
}
}  // namespace
}  // namespace battleship
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::battleship::BasicBattleshipTest();
  open_spiel::battleship::TestZeroSumTrait();
  open_spiel::battleship::TestTightLayout1();
  open_spiel::battleship::TestTightLayout2();
}
