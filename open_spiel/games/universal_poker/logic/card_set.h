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

#ifndef OPEN_SPIEL_CARD_SET_H
#define OPEN_SPIEL_CARD_SET_H

#include <cstdint>
#include <string>
#include <vector>

namespace open_spiel {
namespace universal_poker {
namespace logic {

constexpr int kMaxSuits = 4;  // Also defined in ACPC game.h

// This is an equivalent wrapper to acpc evalHandTables.Cardset.
// It stores the cards for each color over 16 * 4 bits. The use of a Union
// allows to access only a specific color (16 bits) using bySuit[color].
// A uint8_t card is defined by the integer <rank> * MAX_SUITS + <suit>
class CardSet {
 public:
  union CardSetUnion {
    CardSetUnion() : cards(0) {}
    uint16_t bySuit[kMaxSuits];
    uint64_t cards;
  } cs;

 public:
  CardSet() : cs() {}
  CardSet(std::string cardString);
  CardSet(std::vector<int> cards);
  // Returns a set containing num_ranks cards per suit for num_suits.
  CardSet(uint16_t num_suits, uint16_t num_ranks);

  std::string ToString() const;
  // Returns the cards present in this set in ascending order.
  std::vector<uint8_t> ToCardArray() const;

  // Add a card, as MAX_RANKS * <suite> + <rank> to the CardSet.
  void AddCard(uint8_t card);
  // Toogle (does not remove) the bit associated to `card`.
  void RemoveCard(uint8_t card);
  bool ContainsCards(uint8_t card) const;

  int NumCards() const;
  // Returns the ranking value of this set of cards as evaluated by ACPC.
  int RankCards() const;

  // Returns all the possible nbCards-subsets of this CardSet.
  std::vector<CardSet> SampleCards(int nbCards);
};

// Returns the lexicographically next permutation of the supplied bits.
// See https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
uint64_t bit_twiddle_permute(uint64_t v);

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_CARD_SET_H
