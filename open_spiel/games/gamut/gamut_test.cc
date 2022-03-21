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

#include "open_spiel/games/gamut/gamut.h"

#include "open_spiel/matrix_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tensor_game.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace gamut {
namespace {

void BasicLoadGamutTest() {
  GamutGenerator generator("gamut.jar");

  // See the documentation at http://gamut.stanford.edu/ for the commands needed
  // to generate the various different games.

  // Using a string of arguments.
  std::shared_ptr<const Game> game1 = generator.GenerateGame(
      "-g RandomGame -players 4 -normalize -min_payoff 0 "
      "-max_payoff 150 -actions 2 4 5 7");
  SPIEL_CHECK_TRUE(game1 != nullptr);

  // Using a vector of arguments.
  std::shared_ptr<const Game> game2 = generator.GenerateGame(
      {"-g", "RandomGame", "-players", "4", "-normalize", "-min_payoff", "0",
       "-max_payoff", "150", "-actions", "2", "4", "5", "7"});
  SPIEL_CHECK_TRUE(game2 != nullptr);

  // As a matrix game.
  std::shared_ptr<const matrix_game::MatrixGame> game3 =
      generator.GenerateMatrixGame(
          {"-g", "RandomGame", "-players", "2", "-normalize", "-min_payoff",
           "0", "-max_payoff", "150", "-actions", "10", "15"});
  SPIEL_CHECK_TRUE(game3 != nullptr);

  std::shared_ptr<const tensor_game::TensorGame> game4 =
      generator.GenerateTensorGame(
          {"-g", "RandomGame", "-players", "4", "-normalize", "-min_payoff",
           "0", "-max_payoff", "150", "-actions", "2", "4", "5", "7"});
  SPIEL_CHECK_TRUE(game4 != nullptr);
}

}  // namespace
}  // namespace gamut
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  open_spiel::gamut::BasicLoadGamutTest();
}
