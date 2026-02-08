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

#include "open_spiel/games/gomoku/gomoku.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace gomoku {
namespace {

namespace testing = open_spiel::testing;

void BasicGomokuTests() {
  testing::LoadGameTest("gomoku");
  testing::NoChanceOutcomesTest(*LoadGame("gomoku"));
  testing::RandomSimTest(*LoadGame("gomoku"), 100);
}


void TestMoveToActionLoop() {
  auto game = LoadGame("gomoku");

  const GomokuGame* gomoku  = dynamic_cast<const GomokuGame*>(game.get());

  SPIEL_CHECK_TRUE(gomoku != nullptr);
  for (Action a = 0; a < gomoku->NumDistinctActions(); ++a) {
    SPIEL_CHECK_EQ(gomoku->MoveToAction(gomoku->ActionToMove(a)), a);
  }
}

void TestObservationTensor() {
  std::shared_ptr<const Game> game = LoadGame("gomoku");
  std::unique_ptr<State> state = game->NewInitialState();


  const int num_cells = game->ObservationTensorShape()[1];
  const int tensor_size = 3 * num_cells;

  std::vector<float> obs(tensor_size);

  // Initial position
  state->ObservationTensor(/*player=*/0, absl::MakeSpan(obs));

  // Planes 0 & 1: no stones
  for (int i = 0; i < 2 * num_cells; ++i) {
    SPIEL_CHECK_EQ(obs[i], 0.0f);
  }
  // Plane 2: Black to move
  for (int i = 2 * num_cells; i < 3 * num_cells; ++i) {
    SPIEL_CHECK_EQ(obs[i], 1.0f);
  }

  // Make two moves: Black 0, White 1
  state->ApplyAction(0);
  // white on move
  state->ObservationTensor(/*player=*/0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(obs[2 * num_cells], 0.0f);
  state->ApplyAction(1);

  state->ObservationTensor(/*player=*/0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(obs[0], 1.0f);

  // White stone at 1
  SPIEL_CHECK_EQ(obs[num_cells + 1], 1.0f);

  // Turn plane: Black again
  SPIEL_CHECK_EQ(obs[2 * num_cells], 1.0f);
}

}  // namespace
}  // namespace gomoku
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::gomoku::BasicGomokuTests();
  open_spiel::gomoku::TestMoveToActionLoop();
  open_spiel::gomoku::TestObservationTensor();
}
