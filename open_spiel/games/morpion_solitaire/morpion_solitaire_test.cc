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

#include "open_spiel/games/morpion_solitaire/morpion_solitaire.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace morpion_solitaire {
namespace {

namespace testing = open_spiel::testing;

void BasicMorpionTests() {
  testing::LoadGameTest("morpion_solitaire");
  testing::RandomSimTest(*LoadGame("morpion_solitaire"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("morpion_solitaire"), 10);
}

void UndoRestoresReturnsTest() {
  auto game = LoadGame("morpion_solitaire");
  auto state = game->NewInitialState();
  const double returns_before_action = state->Returns().at(0);
  const Action action = state->LegalActions().at(0);

  state->ApplyAction(action);
  SPIEL_CHECK_EQ(state->Returns().at(0), returns_before_action + 1);

  state->UndoAction(kDefaultPlayerId, action);
  SPIEL_CHECK_EQ(state->Returns().at(0), returns_before_action);
}

void MoveConversionTest() {
  Line line = Line(Point(4, 5), Point(1, 8));
  SPIEL_CHECK_EQ(line.GetAction(), 375);
  line = Line(Point(9, 3), Point(9, 6));
  SPIEL_CHECK_EQ(line.GetAction(), 93);
}

void LineOverlapsTest() {
  Line line = Line(Point(5, 2), Point(2, 5));
  SPIEL_CHECK_EQ(line.CheckOverlap(Line(Point(6, 1), Point(3, 4))), true);
  SPIEL_CHECK_EQ(line.CheckOverlap(Line(Point(3, 4), Point(0, 7))), true);
  SPIEL_CHECK_EQ(line.CheckOverlap(Line(Point(4, 3), Point(7, 3))), false);
  line = Line(Point(7, 4), Point(10, 7));
  SPIEL_CHECK_EQ(line.CheckOverlap(Line(Point(7, 2), Point(7, 5))), false);
  SPIEL_CHECK_EQ(line.CheckOverlap(Line(Point(5, 2), Point(8, 5))), true);
}

}  // namespace
}  // namespace morpion_solitaire
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::morpion_solitaire::BasicMorpionTests();
  open_spiel::morpion_solitaire::UndoRestoresReturnsTest();
  open_spiel::morpion_solitaire::MoveConversionTest();
  open_spiel::morpion_solitaire::LineOverlapsTest();
}
