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

#include "open_spiel/games/quarto/quarto.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace quarto {
namespace {

namespace testing = open_spiel::testing;

void BasicTests() {
  testing::LoadGameTest("quarto");
  testing::NoChanceOutcomesTest(*LoadGame("quarto"));
  testing::RandomSimTest(*LoadGame("quarto"), 100);
}

void TestLines() {
  SPIEL_CHECK_TRUE(LineHasQuarto({0, 2, 4, 6}));
  SPIEL_CHECK_TRUE(LineHasQuarto({8, 9, 10, 11}));
  SPIEL_CHECK_FALSE(LineHasQuarto({0, 1, 2, 12}));
  SPIEL_CHECK_FALSE(LineHasQuarto({0, 1, 2, kEmptyCell}));
}

void TestTurnSequenceAndWin() {
  auto state = LoadGame("quarto")->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumPieces);

  state->ApplyAction(0);
  auto* quarto_state = static_cast<QuartoState*>(state.get());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(quarto_state->CurrentPhase(), Phase::kPlace);
  SPIEL_CHECK_EQ(quarto_state->SelectedPiece(), 0);
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumCells);

  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(quarto_state->CurrentPhase(), Phase::kSelect);
  SPIEL_CHECK_EQ(quarto_state->BoardAt(0, 0), 0);
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumPieces - 1);

  // Complete the top row with even-numbered pieces, which all share their
  // least-significant attribute. The player placing piece 6 wins.
  for (Action action : {2, 1, 4, 2, 6, 3}) {
    state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(quarto_state->Outcome(), 0);
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({1.0, -1.0}));

  auto restored =
      state->GetGame()->NewInitialState(nlohmann::json::parse(state->ToJson()));
  SPIEL_CHECK_TRUE(restored->IsTerminal());
  SPIEL_CHECK_EQ(restored->ToString(), state->ToString());

  state->UndoAction(0, 3);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(quarto_state->CurrentPhase(), Phase::kPlace);
  SPIEL_CHECK_EQ(quarto_state->SelectedPiece(), 6);
  state->ApplyAction(3);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({1.0, -1.0}));
}

void TestUndo() {
  auto state = LoadGame("quarto")->NewInitialState();
  state->ApplyAction(5);
  state->UndoAction(0, 5);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumPieces);

  state->ApplyAction(5);
  state->ApplyAction(7);
  state->UndoAction(1, 7);
  auto* quarto_state = static_cast<QuartoState*>(state.get());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(quarto_state->CurrentPhase(), Phase::kPlace);
  SPIEL_CHECK_EQ(quarto_state->SelectedPiece(), 5);
  SPIEL_CHECK_EQ(quarto_state->BoardAt(1, 3), kEmptyCell);
}

void TestStructs() {
  auto game = LoadGame("quarto");
  auto state = game->NewInitialState();
  state->ApplyAction(9);

  auto state_struct = state->ToStruct();
  SPIEL_CHECK_EQ(state_struct->ToJson(), state->ToJson());
  auto restored =
      game->NewInitialState(nlohmann::json::parse(state_struct->ToJson()));
  SPIEL_CHECK_EQ(restored->ToString(), state->ToString());

  auto observation_struct = state->ToObservationStruct(0);
  SPIEL_CHECK_EQ(observation_struct->ToJson(), state_struct->ToJson());

  auto action_struct = state->ActionToStruct(1, 6);
  auto* quarto_action = SafeActionCast<QuartoActionStruct>(*action_struct);
  SPIEL_CHECK_EQ(quarto_action->action_type, "place");
  SPIEL_CHECK_EQ(quarto_action->piece, 9);
  SPIEL_CHECK_EQ(quarto_action->row, 1);
  SPIEL_CHECK_EQ(quarto_action->col, 2);
  SPIEL_CHECK_TRUE(state->ValidateActionStruct(*action_struct).ok());
  SPIEL_CHECK_EQ(state->StructToActions(*action_struct),
                 std::vector<Action>({6}));

  auto state2 = game->NewInitialState();
  auto selection_struct = state2->ActionToStruct(0, 12);
  SPIEL_CHECK_TRUE(state2->ApplyActionStruct(*selection_struct).ok());
  SPIEL_CHECK_EQ(static_cast<QuartoState*>(state2.get())->SelectedPiece(), 12);
}

void TestObservationTensor() {
  auto game = LoadGame("quarto");
  auto state = game->NewInitialState();
  state->ApplyAction(3);

  std::vector<float> tensor = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(tensor[3 * (kNumCells + 1) + kNumCells], 1.0);

  state->ApplyAction(5);
  tensor = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(tensor[3 * (kNumCells + 1) + 5], 1.0);
  SPIEL_CHECK_EQ(tensor[3 * (kNumCells + 1) + kNumCells], 0.0);
}

}  // namespace
}  // namespace quarto
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::quarto::BasicTests();
  open_spiel::quarto::TestLines();
  open_spiel::quarto::TestTurnSequenceAndWin();
  open_spiel::quarto::TestUndo();
  open_spiel::quarto::TestStructs();
  open_spiel::quarto::TestObservationTensor();
}
