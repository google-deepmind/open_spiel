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
  const std::string gamedef(
      "GAMEDEF\n"
      "nolimit\n"
      "numPlayers = 2\n"
      "numRounds = 2\n"
      "stack = 1200 1200\n"
      "blind = 100 100\n"
      "firstPlayer = 1 1\n"
      "numSuits = 2\n"
      "numRanks = 3\n"
      "numHoleCards = 1\n"
      "numBoardCards = 0 1\n"
      "END GAMEDEF");

  acpc_cpp::ACPCGame acpc_game(gamedef);

  std::srand(std::time(nullptr));

  for (int i = 0; i < 100; ++i) {
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
  // This is the "holdem.nolimit.2p.reverse_blinds.game" ACPC example.
  const std::string holdem_nolimit_2p_reverse_blinds(
      "GAMEDEF\n"
      "nolimit\n"
      "numPlayers = 2\n"
      "numRounds = 4\n"
      "stack = 20000 20000\n"
      "blind = 100 50\n"
      "firstPlayer = 2 1 1 1\n"
      "numSuits = 4\n"
      "numRanks = 13\n"
      "numHoleCards = 2\n"
      "numBoardCards = 0 3 1 1\n"
      "END GAMEDEF");
  acpc_cpp::ACPCGame acpc_game(holdem_nolimit_2p_reverse_blinds);

  std::srand(std::time(nullptr));

  for (int i = 0; i < 100; ++i) {
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
