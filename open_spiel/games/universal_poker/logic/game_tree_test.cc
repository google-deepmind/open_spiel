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

#include "open_spiel/games/universal_poker/logic/game_tree.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

void BasicGameTreeTests() {
  const std::string gameDesc(
      "GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 "
      "1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = "
      "3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

  acpc_cpp::ACPCGame acpc_game(gameDesc);

  std::srand(std::time(nullptr));

  for (int i = 0; i < 100; i++) {
    GameNode node(&acpc_game);
    std::cout << node.ToString() << std::endl;
    while (!node.IsFinished()) {
      uint32_t actions = node.GetActionCount();
      uint32_t action = std::rand() % actions;

      std::cout << "Choose Action: " << action << std::endl;
      node.ApplyAction(action);
      std::cout << node.ToString() << std::endl;
    }
  }
}

void HoldemGameTreeTests() {
  const std::string gameDescHoldem(
      "GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 4\nstack = 20000 "
      "20000\nblind = 100 50\nfirstPlayer = 2 1 1 1\nnumSuits = 4\nnumRanks = "
      "13\nnumHoleCards = 2\nnumBoardCards = 0 3 1 1\nEND GAMEDEF");

  acpc_cpp::ACPCGame acpc_game(gameDescHoldem);

  std::srand(std::time(nullptr));

  for (int i = 0; i < 100; i++) {
    GameNode node(&acpc_game);
    while (!node.IsFinished()) {
      uint32_t actions = node.GetActionCount();
      uint32_t action = std::rand() % actions;
      node.ApplyAction(action);
    }

    std::cout << node.ToString() << std::endl;
  }
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::logic::BasicGameTreeTests();
  open_spiel::universal_poker::logic::HoldemGameTreeTests();
}
