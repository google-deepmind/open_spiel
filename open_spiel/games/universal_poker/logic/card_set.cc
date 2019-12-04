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

#include <bitset>
#include <sstream>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

constexpr absl::string_view kSuitChars = "cdhs";
constexpr absl::string_view kRankChars = "23456789TJQKA";

extern "C" {
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/evalHandTables"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
}

namespace open_spiel::universal_poker::logic {

using I = uint64_t;

auto dump(I v) { return std::bitset<sizeof(I) * __CHAR_BIT__>(v); }
I bit_twiddle_permute(I v) {
  I t = v | (v - 1);
  I w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzl(v) + 1));

  return w;
}

CardSet::CardSet(std::string cardString) : cs() {
  for (int i = 0; i < cardString.size(); i += 2) {
    char rankChr = cardString[i];
    char suitChr = cardString[i + 1];

    uint8_t rank = (uint8_t)(kRankChars.find(rankChr));
    uint8_t suit = (uint8_t)(kSuitChars.find(suitChr));

    cs.bySuit[suit] |= ((uint16_t)1 << rank);
  }
}

CardSet::CardSet(std::vector<int> cards) : cs() {
  for (int i = 0; i < cards.size(); ++i) {
    int rank = rankOfCard(cards[i]);
    int suit = suitOfCard(cards[i]);

    cs.bySuit[suit] |= ((uint16_t)1 << rank);
  }
}

CardSet::CardSet(uint16_t num_suits, uint16_t num_ranks) : cs() {
  for (uint16_t r = 0; r < num_ranks; r++) {
    for (uint16_t s = 0; s < num_suits; s++) {
      cs.bySuit[s] |= ((uint16_t)1 << r);
    }
  }
}

std::string CardSet::ToString() const {
  std::ostringstream result;
  for (int r = MAX_RANKS - 1; r >= 0; r--) {
    for (int s = MAX_SUITS - 1; s >= 0; s--) {
      uint32_t mask = (uint32_t)1 << r;
      if (cs.bySuit[s] & mask) {
        result << kRankChars[r] << kSuitChars[s];
      }
    }
  }

  return result.str();
}

std::vector<uint8_t> CardSet::ToCardArray() const {
  std::vector<uint8_t> result(CountCards(), 0);

  int i = 0;
  for (int r = MAX_RANKS - 1; r >= 0; r--) {
    for (int s = MAX_SUITS - 1; s >= 0; s--) {
      uint32_t mask = (uint32_t)1 << r;
      if (cs.bySuit[s] & mask) {
        result[i++] = makeCard(r, s);
      }
    }
  }
  return result;
}

void CardSet::AddCard(uint8_t card) {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);

  cs.bySuit[suit] |= ((uint16_t)1 << rank);
}

void CardSet::RemoveCard(uint8_t card) {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);

  cs.bySuit[suit] ^= ((uint16_t)1 << rank);
}

int CardSet::CountCards() const { return __builtin_popcountl(cs.cards); }

int CardSet::RankCards() const {
  ::Cardset csNative;
  csNative.cards = cs.cards;
  return rankCardset(csNative);
}

std::vector<CardSet> CardSet::SampleCards(int nbCards) {
  std::vector<CardSet> combinations;

  uint64_t p = 0;
  for (int i = 0; i < nbCards; ++i) {
    p += (1 << i);
  }

  for (I n = bit_twiddle_permute(p); n > p; p = n, n = bit_twiddle_permute(p)) {
    uint64_t combo = n & cs.cards;
    if (__builtin_popcountl(combo) == nbCards) {
      CardSet c;
      c.cs.cards = combo;
      combinations.emplace_back(c);
    }
  }

  // std::cout << "combinations.size() " << combinations.size() << std::endl;
  return combinations;
}

bool CardSet::ContainsCards(uint8_t card) {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);
  return (cs.bySuit[suit] & ((uint16_t)1 << rank)) > 0;
}

}  // namespace open_spiel::universal_poker::logic
