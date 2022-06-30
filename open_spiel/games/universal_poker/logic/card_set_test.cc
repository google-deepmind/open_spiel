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

#include "open_spiel/games/universal_poker/logic/card_set.h"

#include <iostream>

namespace open_spiel {
namespace universal_poker {
namespace logic {

void BasicCardSetTests() {
  CardSet cs("AhKsQhJhTh");

  std::cout << "CardSet: " << cs.ToString() << std::endl;
  for (auto card : cs.ToCardArray()) {
    std::cout << "Card: " << card << std::endl;
  }
  std::cout << "Rank: " << cs.RankCards() << std::endl;
  std::cout << "Count Cards: " << cs.NumCards() << std::endl;

  CardSet deck(4, 13);
  std::cout << "CardSet: " << deck.ToString() << std::endl;
  std::cout << "Rank: " << deck.RankCards() << std::endl;
  std::cout << "Count Cards: " << deck.NumCards() << std::endl;

  for (auto combo : deck.SampleCards(3)) {
    std::cout << "CardSet: " << combo.ToString() << std::endl;
  }

  for (auto combo : deck.SampleCards(1)) {
    std::cout << "CardSet: " << combo.ToString() << std::endl;
  }
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::logic::BasicCardSetTests();
}
