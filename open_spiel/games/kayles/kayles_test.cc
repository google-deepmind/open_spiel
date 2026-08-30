// Copyright 2026 DeepMind Technologies Limited
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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace kayles {
namespace {

void InitialLegalActionsTest() {
  std::shared_ptr<const Game> game =
      LoadGame("kayles", {{"row_length", GameParameter(5)}});
  std::unique_ptr<State> state = game->NewInitialState();

  SPIEL_CHECK_EQ(game->NumDistinctActions(), 9);
  SPIEL_CHECK_EQ(state->LegalActions(),
                 std::vector<Action>({0, 1, 2, 3, 4, 5, 6, 7, 8}));
}

void PairRemovalTest() {
  std::shared_ptr<const Game> game =
      LoadGame("kayles", {{"row_length", GameParameter(5)}});
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(6);  // Remove the pair beginning at pin 1.

  SPIEL_CHECK_EQ(state->ToString(), "P1: |..||");
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>({0, 3, 4, 8}));
}

void LastMoveWinsTest() {
  std::shared_ptr<const Game> game =
      LoadGame("kayles", {{"row_length", GameParameter(2)}});
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(2);

  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({1.0, -1.0}));
}

void UndoPairTest() {
  std::shared_ptr<const Game> game =
      LoadGame("kayles", {{"row_length", GameParameter(4)}});
  std::unique_ptr<State> state = game->NewInitialState();
  const std::string initial_state = state->ToString();
  const std::vector<Action> initial_actions = state->LegalActions();

  state->ApplyAction(5);
  state->UndoAction(0, 5);

  SPIEL_CHECK_EQ(state->ToString(), initial_state);
  SPIEL_CHECK_EQ(state->LegalActions(), initial_actions);
}

}  // namespace
}  // namespace kayles
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::kayles::InitialLegalActionsTest();
  open_spiel::kayles::PairRemovalTest();
  open_spiel::kayles::LastMoveWinsTest();
  open_spiel::kayles::UndoPairTest();
}
