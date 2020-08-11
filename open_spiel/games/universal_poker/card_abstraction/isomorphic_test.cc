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

#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/games/universal_poker/card_abstraction/isomorphic.h"
#include "open_spiel/spiel.h"

#include <vector>

namespace open_spiel {
namespace universal_poker {
namespace card_abstraction {

void IsomorphicAbstractionTests() {
  logic::CardSet AcKc("AcKc");
  logic::CardSet AdKd("AdKd");
  logic::CardSet AsKs("AsKs");
  logic::CardSet empty;
  logic::CardSet flop("QcJcTc");

  std::vector<int> card_per_round{ 2, 3 };
  IsomorphicCardAbstraction flopEmAbstractor(card_per_round);

  SPIEL_CHECK_EQ(
    flopEmAbstractor.abstract(AcKc, empty),
    flopEmAbstractor.abstract(AdKd, empty));
  SPIEL_CHECK_EQ(
    flopEmAbstractor.abstract(AcKc, empty),
    flopEmAbstractor.abstract(AsKs, empty));

  SPIEL_CHECK_EQ(
    flopEmAbstractor.abstract(AdKd, flop),
    flopEmAbstractor.abstract(AsKs, flop)
  );

  SPIEL_CHECK_NE(
    flopEmAbstractor.abstract(AcKc, flop),
    flopEmAbstractor.abstract(AsKs, flop)
  );

}

}  // namespace card_abstraction
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::card_abstraction::IsomorphicAbstractionTests();
}
