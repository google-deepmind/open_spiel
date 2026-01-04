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

#ifndef OPEN_SPIEL_ALGORITHMS_TENSOR_GAMES_UTILS_H_
#define OPEN_SPIEL_ALGORITHMS_TENSOR_GAMES_UTILS_H_

#include <memory>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/tensor_game.h"

namespace open_spiel {
namespace algorithms {

// Similar to open_spiel::LoadGame but returns specifically a tensor game type
// so that the subclass's specific methods are accessible.

std::shared_ptr<const tensor_game::TensorGame> LoadTensorGame(
    const std::string& name);

// Clones a normal-form game and returns it as a TensorGame.

std::shared_ptr<const tensor_game::TensorGame> AsTensorGame(
    const NormalFormGame* game);

std::shared_ptr<const tensor_game::TensorGame> AsTensorGame(const Game* game);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_TENSOR_GAME_UTILS_H_
