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

#include "open_spiel/algorithms/value_iteration.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace nim {
namespace {

namespace testing = open_spiel::testing;
namespace algorithms = open_spiel::algorithms;

void BasicNimTests() {
  testing::LoadGameTest("nim");
  testing::RandomSimTest(*LoadGame("nim"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("nim"), 10);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes", GameParameter("100;200;300")},
                }),
      10);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes",
                     GameParameter("10000;2000;3000;12414;1515;53252;1;35126")},
                }),
      10);
  testing::RandomSimTest(
      *LoadGame("nim",
                {
                    {"pile_sizes", GameParameter("1;2;3;4;5;6;7;8;9;10")},
                    {"is_misere", GameParameter(false)},
                }),
      10);
}

void SinglePileNormalTest() {
  std::shared_ptr<const Game> game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("100")},
                          {"is_misere", GameParameter(false)},
                      });
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> actions = state->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 100);

  state->ApplyAction(actions.back());
  SPIEL_CHECK_EQ(state->IsTerminal(), 1);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1);
}

void SinglePileMisereTest() {
  std::shared_ptr<const Game> game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("100")},
                      });
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<Action> actions = state->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 100);

  state->ApplyAction(actions.back());
  SPIEL_CHECK_EQ(state->IsTerminal(), 1);
  SPIEL_CHECK_EQ(state->PlayerReturn(0), -1);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1);
}

void VISinglePileNormalTest() {
  std::shared_ptr<const Game> game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("100")},
                          {"is_misere", GameParameter(false)},
                      });
  auto values = algorithms::ValueIteration(*game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 100"], 1);
}

void VISinglePileMisereTest() {
  std::shared_ptr<const Game> game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("100")},
                      });
  auto values = algorithms::ValueIteration(*game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 100"], 1);
}

// See "Winning positions" here
// https://en.wikipedia.org/wiki/Nim
// to understand the "pile_sizes" parameter from the tests below
void VIThreeOnesNormalTest() {
  std::shared_ptr<const Game> normal_game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("1;1;1")},
                          {"is_misere", GameParameter(false)},
                      });
  auto values = algorithms::ValueIteration(*normal_game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 1 1 1"], 1);
}

void VIThreeOnesMisereTest() {
  std::shared_ptr<const Game> game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("1;1;1")},
                      });
  auto values = algorithms::ValueIteration(*game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 1 1 1"], -1);
}

void VIThreePilesTest() {
  std::shared_ptr<const Game> normal_game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("5;8;13")},
                          {"is_misere", GameParameter(false)},
                      });
  auto values = algorithms::ValueIteration(*normal_game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 5 8 13"], -1);
}

void VIFourPilesTest() {
  std::shared_ptr<const Game> normal_game =
      LoadGame("nim", {
                          {"pile_sizes", GameParameter("2;3;8;10")},
                          {"is_misere", GameParameter(false)},
                      });
  auto values = algorithms::ValueIteration(*normal_game, -1, 0.01);
  SPIEL_CHECK_EQ(values["(0): 2 3 8 10"], 1);
}

}  // namespace
}  // namespace nim
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::nim::BasicNimTests();
  open_spiel::nim::SinglePileNormalTest();
  open_spiel::nim::SinglePileMisereTest();
  open_spiel::nim::VISinglePileNormalTest();
  open_spiel::nim::VISinglePileMisereTest();
  open_spiel::nim::VIThreeOnesNormalTest();
  open_spiel::nim::VIThreeOnesMisereTest();
  open_spiel::nim::VIThreePilesTest();
  open_spiel::nim::VIFourPilesTest();
}
