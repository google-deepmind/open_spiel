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

#include "open_spiel/algorithms/tensor_game_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

void ConvertToTensorGameTest() {
  std::shared_ptr<const Game> blotto = LoadGame("blotto(players=3)");
  std::shared_ptr<const tensor_game::TensorGame> tensor_blotto =
      AsTensorGame(blotto.get());
  SPIEL_CHECK_EQ(tensor_blotto->Shape()[0], 66);
  SPIEL_CHECK_EQ(tensor_blotto->Shape()[1], 66);
  SPIEL_CHECK_EQ(tensor_blotto->Shape()[2], 66);
  std::cout << "Blotto 0,15,3 = " << tensor_blotto->ActionName(Player{0}, 0)
            << " vs " << tensor_blotto->ActionName(Player{1}, 15) << " vs "
            << tensor_blotto->ActionName(Player{2}, 3) << " -> utils: "
            << tensor_blotto->PlayerUtility(Player{0}, {0, 15, 3}) << ","
            << tensor_blotto->PlayerUtility(Player{1}, {0, 15, 3}) << ","
            << tensor_blotto->PlayerUtility(Player{2}, {0, 15, 3}) << std::endl;
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::ConvertToTensorGameTest();
}
