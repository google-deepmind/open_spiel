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

#include "open_spiel/game_transforms/normal_form_extensive_game.h"

namespace open_spiel {
namespace {

void ExtensiveToTensorGameTest() {
  // This just does a conversion and checks the sizes.
  std::shared_ptr<const Game> auction_game =
      LoadGame("first_sealed_auction(players=3,max_value=4)");
  std::shared_ptr<const tensor_game::TensorGame> auction_tensor_game =
      ExtensiveToTensorGame(*auction_game);
  SPIEL_CHECK_EQ(auction_tensor_game->Shape()[0], 24);
  SPIEL_CHECK_EQ(auction_tensor_game->Shape()[1], 24);
  SPIEL_CHECK_EQ(auction_tensor_game->Shape()[2], 24);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::ExtensiveToTensorGameTest();
}
