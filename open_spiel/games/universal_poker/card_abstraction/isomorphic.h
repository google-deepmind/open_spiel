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

#ifndef OPEN_SPIEL_ISOMORPHIC_CARD_ABSTRACTION_H
#define OPEN_SPIEL_ISOMORPHIC_CARD_ABSTRACTION_H

#include <vector>

#include "open_spiel/games/universal_poker/card_abstraction/card_abstraction.h"

extern "C" {
#include "open_spiel/games/universal_poker/hand-isomorphism/src/hand_index.h"
}

namespace open_spiel {
namespace universal_poker {
namespace card_abstraction {

/**
Maps poker hands to and from a tight set of indices. Poker hands are
isomorphic with respect to permutations of the suits and ordering within a
betting round. That is, AsKs, KdAd and KhAh all map to the same index preflop.

Based on Kevin Waugh's https://github.com/kdub0/hand-isomorphism
http://www.cs.cmu.edu/~./kwaugh/publications/isomorphism13.pdf

Limitations: only supports 52 card (13 rank, 4 suit) deck currently
 */
class IsomorphicCardAbstraction: public CardAbstraction {
 private:
  std::vector<hand_indexer_t> indexers_;
  std::vector<int> cards_per_round_;
 public:
  IsomorphicCardAbstraction(std::vector<int> cards_per_round);

  std::pair<logic::CardSet, logic::CardSet>
  abstract(logic::CardSet hole_cards,
           logic::CardSet board_cards) const override;
};

}  // namespace card_abstraction
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ISOMORPHIC_CARD_ABSTRACTION_H
