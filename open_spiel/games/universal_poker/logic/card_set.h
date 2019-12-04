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

#ifndef OPEN_SPIEL_CARD_SET_H
#define OPEN_SPIEL_CARD_SET_H

#include <cstdint>
#include <string>
#include <vector>

namespace open_spiel {
namespace universal_poker {
namespace logic {

constexpr int MAX_SUITS = 4;

// This is an equivalent of the ACPC evalHandTables.Cardset struct.
// A card is defined by the integer <rank> * MAX_SUITS + <suit>
class CardSet {
 public:
  union CardSetUnion {
    CardSetUnion() : cards(0) {}
    uint16_t bySuit[MAX_SUITS];
    uint64_t cards;
  } cs;

 public:
  CardSet() : cs() {}
  CardSet(std::string cardString);
  CardSet(std::vector<int> cards);
  CardSet(uint8_t cards[], int size);
  CardSet(uint16_t numSuits, uint16_t numRanks);

  std::string ToString() const;
  std::vector<uint8_t> ToCardArray() const;

  void AddCard(uint8_t card);
  void RemoveCard(uint8_t card);
  bool ContainsCards(uint8_t card);

  uint32_t CountCards() const;
  int RankCards();
  bool IsBlocking(CardSet other);

  std::vector<CardSet> SampleCards(int nbCards);
};

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_CARD_SET_H
