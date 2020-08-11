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

#ifndef OPEN_SPIEL_ABSTRACTOR_H
#define OPEN_SPIEL_ABSTRACTOR_H

#include "open_spiel/games/universal_poker/logic/card_set.h"

namespace open_spiel {
namespace universal_poker {
namespace card_abstraction {

class CardAbstraction {
 public:
  logic::CardSet deck;
  // Abstract multiple set of cards into buckets to lower number of info states
  virtual std::pair<logic::CardSet, logic::CardSet>
  abstract(logic::CardSet hole_cards, logic::CardSet board_cards) const = 0;
};

class NoopCardAbstraction: public CardAbstraction {
 public:
  std::pair<logic::CardSet, logic::CardSet>
  abstract(logic::CardSet hole_cards,
           logic::CardSet board_cards) const override {

    return {hole_cards, board_cards};
  }
};

}  // namespace card_abstraction
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ABSTRACTOR_H
