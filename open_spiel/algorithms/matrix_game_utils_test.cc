// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/matrix_game_utils.h"

#include "open_spiel/games/kuhn_poker.h"

namespace open_spiel {
namespace algorithms {
namespace {

void ConvertToMatrixGameTest() {
  std::shared_ptr<const Game> blotto = LoadGame("blotto");
  std::shared_ptr<const matrix_game::MatrixGame> matrix_blotto =
      AsMatrixGame(blotto.get());
  SPIEL_CHECK_GT(matrix_blotto->NumRows(), 0);
  SPIEL_CHECK_GT(matrix_blotto->NumCols(), 0);
  SPIEL_CHECK_EQ(matrix_blotto->NumRows(), 66);
  SPIEL_CHECK_EQ(matrix_blotto->NumCols(), 66);
  std::cout << "Blotto 0,13 = " << matrix_blotto->RowActionName(0) << " vs "
            << matrix_blotto->ColActionName(13)
            << " -> utils: " << matrix_blotto->PlayerUtility(Player{0}, 0, 13)
            << "," << matrix_blotto->PlayerUtility(Player{1}, 0, 13)
            << std::endl;
}

void ExtensiveToMatrixGameTest() {
  // This just does a conversion and checks the sizes. The real test of this
  // method is currently in python/tests/matrix_utils_test, which solves the
  // converted game to ensure it has the same value.
  std::shared_ptr<const Game> kuhn_game = LoadGame("kuhn_poker");
  std::shared_ptr<const matrix_game::MatrixGame> kuhn_matrix_game =
      ExtensiveToMatrixGame(*kuhn_game);
  SPIEL_CHECK_EQ(kuhn_matrix_game->NumRows(), 64);
  SPIEL_CHECK_EQ(kuhn_matrix_game->NumCols(), 64);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::ConvertToMatrixGameTest();
  open_spiel::algorithms::ExtensiveToMatrixGameTest();
}
