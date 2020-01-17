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

#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"

#include <algorithm>
#include <set>
#include <utility>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace gin_rummy {

inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 13;
inline constexpr int kNumCards = kNumSuits * kNumRanks;

int CardSuit(int card) { return card / kNumRanks; }
int CardRank(int card) { return card % kNumRanks; }

// All suits are of equal value and suit ordering never factors into gameplay.
constexpr char kRankChar[] = "A23456789TJQK";
constexpr char kSuitChar[] = "scdh";

std::string CardString(int card) {
  if (card < 0) {
    return "XX";
  }
  return {kRankChar[CardRank(card)],
          kSuitChar[CardSuit(card)]};
}

int CardInt(std::string card) {
  SPIEL_CHECK_EQ(card.length(), 2);
  int rank = strchr(kRankChar, card[0]) - kRankChar;
  int suit = strchr(kSuitChar, card[1]) - kSuitChar;
  return suit * kNumRanks + rank;
}

std::vector<std::string> CardIntsToCardStrings(const std::vector<int> &cards) {
  std::vector<std::string> rv;
  for (int card : cards) {
    rv.push_back(CardString(card));
  }
  return rv;
}

std::vector<int> CardStringsToCardInts(const std::vector<std::string> &cards) {
  std::vector<int> rv;
  for (std::string card : cards) {
    rv.push_back(CardInt(card));
  }
  return rv;
}

std::string HandToString(const std::vector<int> &cards) {
  std::string rv;
  absl::StrAppend(&rv, "+--------------------------+\n");
  for (int i = 0; i < kNumSuits; ++i) {
    absl::StrAppend(&rv, "|");
    for (int j = 0; j < kNumRanks; ++j) {
      if (find(cards.begin(),
               cards.end(), (i * kNumRanks) + j) != cards.end()) {
        absl::StrAppend(&rv, CardString((i * kNumRanks) + j));
      } else {
        absl::StrAppend(&rv, "  ");
      }
    }
    absl::StrAppend(&rv, "|\n");
  }
  absl::StrAppend(&rv, "+--------------------------+\n");
  return rv;
}

// Ace = 1, deuce = 2, ... , face cards = 10.
int CardValue(int card_index) {
  int value = CardRank(card_index) + 1;
  if (value >= 10) {
    return 10;
  }
  return value;
}

// Sums point total over all cards.
int TotalCardValue(const std::vector<int> &cards) {
  int total_value = 0;
  for (int card : cards) {
    total_value += CardValue(card);
  }
  return total_value;
}

// Sums point total over all cards.
int TotalCardValue(const std::vector<std::vector<int>> &meld_group) {
  int total_value = 0;
  for (auto meld : meld_group) {
    for (auto card : meld) {
      total_value += CardValue(card);
    }
  }
  return total_value;
}

// Use suits to break ties.
bool CompareRanks(int card_1, int card_2) {
  if (CardRank(card_1) == CardRank(card_2)) {
    return card_1 < card_2;
  }
  return CardRank(card_1) < CardRank(card_2);
}

// Use ranks to break ties.
bool CompareSuits(int card_1, int card_2) {
  if (CardSuit(card_1) == CardSuit(card_2)) {
    return CompareRanks(card_1, card_2);
  }
  return CardSuit(card_1) < CardSuit(card_2);
}

bool IsConsecutive(const std::vector<int> &v) {
  for (int i = 1; i < v.size(); ++i) {
    if (v[i] != v[i - 1] + 1) {
      return false;
    }
  }
  return true;
}


bool IsRankMeld(const std::vector<int> &cards) {
  if (cards.size() != 3 && cards.size() != 4) {
    return false;
  }
  for (int i = 1; i < cards.size(); ++i) {
    if (CardRank(cards[0]) != CardRank(cards[i])) {
      return false;
    }
  }
  return true;
}

bool IsSuitMeld(const std::vector<int> &cards) {
  if (cards.size() < 3) {
    return false;
  }
  // Check all of the same suit.
  for (int i = 1; i < cards.size(); ++i) {
    if (CardSuit(cards[0]) != CardSuit(cards[i])) {
      return false;
    }
  }
  // Check ranks are consecutive.
  std::vector<int> ranks;
  for (int i = 0; i < cards.size(); ++i) {
    ranks.push_back(CardRank(cards[i]));
  }
  std::sort(ranks.begin(), ranks.end());
  return IsConsecutive(ranks);
}

// Returns all possible rank melds that can be formed from cards.
std::vector<std::vector<int>> RankMelds(std::vector<int> cards) {
  std::vector<std::vector<int>> melds;
  if (cards.size() < 3) {
    return melds;
  }
  std::sort(cards.begin(), cards.end(), CompareRanks);
  // First do a sweep for 4 card melds.
  for (int i = 0; i < cards.size() - 3; ++i) {
    // Found 4 card meld - implies there are four 3 card melds as well.
    // We only add two of the 3 card melds here, the other two get added
    // during the 3 card meld sweep.
    if (CardRank(cards[i]) == CardRank(cards[i + 3])) {
      std::vector<int> m1(cards.begin()+i, cards.begin() + i + 4);
      melds.push_back(m1);
      std::vector<int> m2;
      m2.push_back(cards[i]);
      m2.push_back(cards[i+1]);
      m2.push_back(cards[i+3]);
      melds.push_back(m2);
      std::vector<int> m3;
      m3.push_back(cards[i]);
      m3.push_back(cards[i+2]);
      m3.push_back(cards[i+3]);
      melds.push_back(m3);
    }
  }
  // Sweep for 3 card melds.
  for (int i = 0; i < cards.size() - 2; ++i) {
    if (CardRank(cards[i]) == CardRank(cards[i+2])) {
      std::vector<int> m(cards.begin()+i, cards.begin()+i+3);
      melds.push_back(m);
    }
  }
  return melds;
}

// Returns all possible suit melds that can be formed from cards.
std::vector<std::vector<int>> SuitMelds(std::vector<int> cards) {
  std::vector<std::vector<int>> melds;
  if (cards.size() < 3) {
    return melds;
  }
  std::sort(cards.begin(), cards.end(), CompareSuits);
  // Find all suit melds of length 5.
  if (cards.size() >= 5) {
    for (int i = 0; i < cards.size() - 4; ++i) {
      if (cards[i] == cards[i+4] - 4 &&
          CardSuit(cards[i]) == CardSuit(cards[i+4])) {
        std::vector<int> v(cards.begin()+i, cards.begin()+i+5);
        melds.push_back(v);
      }
    }
  }
  // Find all suit melds of length 4.
  if (cards.size() >= 4) {
    for (int i = 0; i < cards.size() - 3; ++i) {
      if (cards[i] == cards[i+3] - 3 &&
          CardSuit(cards[i]) == CardSuit(cards[i+3])) {
        std::vector<int> v(cards.begin()+i, cards.begin()+i+4);
        melds.push_back(v);
      }
    }
  }
  // Find all suit melds of length 3.
  for (int i = 0; i < cards.size() - 2; ++i) {
    if (cards[i] == cards[i+2] - 2 &&
        CardSuit(cards[i]) == CardSuit(cards[i+2])) {
      std::vector<int> v(cards.begin()+i, cards.begin()+i+3);
      melds.push_back(v);
    }
  }
  return melds;
}

// Returns all melds of length 5 or less. Any meld of length 6 or more can
// be expressed as two or more melds of shorter length.
std::vector<std::vector<int>> AllMelds(const std::vector<int> &cards) {
  std::vector<std::vector<int>> rank_melds = RankMelds(cards);
  std::vector<std::vector<int>> suit_melds = SuitMelds(cards);
  for (int i = 0; i < suit_melds.size(); ++i) {
    rank_melds.push_back(suit_melds[i]);
  }
  return rank_melds;
}

bool VectorsIntersect(std::vector<int> *v1, std::vector<int> *v2) {
  std::sort(v1->begin(), v1->end());
  std::sort(v2->begin(), v2->end());
  for (int i = 0; i < v1->size(); ++i) {
    if (std::binary_search(v2->begin(), v2->end(), (*v1)[i])) {
      return true;
    }
  }
  return false;
}

// Returns melds which do not share any common cards with given meld.
std::vector<std::vector<int>> NonOverlappingMelds(
    std::vector<int> *meld, std::vector<std::vector<int>> *melds) {
  std::vector<std::vector<int>> rv;
  for (int i = 0; i < melds->size(); ++i) {
    if (!VectorsIntersect(meld, &(*melds)[i])) {
      rv.push_back((*melds)[i]);
    }
  }
  return rv;
}

// Depth first search used by AllMeldGroups.
void AllPaths(std::vector<int> *meld,
              std::vector<std::vector<int>> *all_melds,
              std::vector<std::vector<int>> *path,
              std::vector<std::vector<std::vector<int>>> *all_paths) {
  path->push_back(*meld);
  std::vector<std::vector<int>> child_melds =
      NonOverlappingMelds(meld, all_melds);
  if (child_melds.size() == 0) {
    all_paths->push_back(*path);
  } else {
    for (auto child_meld : child_melds) {
      AllPaths(&child_meld, &child_melds, path, all_paths);
    }
  }
  path->pop_back();
}

// A meld group is an arrangement of cards into distinct melds.
// Accordingly, no two melds in a meld group can share the same card.
std::vector<std::vector<std::vector<int>>> AllMeldGroups(
    const std::vector<int> &cards) {
  std::vector<std::vector<int>> all_melds = AllMelds(cards);
  std::vector<std::vector<std::vector<int>>> all_meld_groups;
  for (std::vector<int> meld : all_melds) {
    std::vector<std::vector<int>> path;
    AllPaths(&meld, &all_melds, &path, &all_meld_groups);
  }
  return all_meld_groups;
}

// "Best" means any meld group that achieves the lowest possible deadwood
// count for the given cards. In general this is non-unique.
std::vector<std::vector<int>> BestMeldGroup(const std::vector<int> &cards) {
  int best_meld_group_total_value = 0;
  std::vector<std::vector<int>> best_meld_group;
  std::vector<std::vector<std::vector<int>>> all_meld_groups =
      AllMeldGroups(cards);
  for (auto meld_group : all_meld_groups) {
    int meld_group_total_value = TotalCardValue(meld_group);
    if (meld_group_total_value > best_meld_group_total_value) {
      best_meld_group_total_value = meld_group_total_value;
      best_meld_group = meld_group;
    }
  }
  return best_meld_group;
}

// Minimum deadwood count over all meld groups.
int MinDeadwood(const std::vector<int> &hand, int card) {
  std::vector<int> hand_ = hand;
  hand_.push_back(card);
  return MinDeadwood(hand_);
}

// Minimum deadwood count over all meld groups.
int MinDeadwood(const std::vector<int> &hand) {
  std::vector<int> hand_ = hand;
  std::vector<int> deadwood = hand;
  std::vector<std::vector<int>> best_melds = BestMeldGroup(hand_);

  for (auto meld : best_melds) {
    for (auto card : meld) {
      deadwood.erase(remove(deadwood.begin(), deadwood.end(), card),
                     deadwood.end());
    }
  }
  // If we have 11 cards we can discard one.
  if (hand.size() == 11 && deadwood.size() > 0) {
    std::sort(deadwood.begin(), deadwood.end(), CompareRanks);
    deadwood.pop_back();
  }
  int deadwood_total = 0;
  for (int card : deadwood)
    deadwood_total += CardValue(card);
  return deadwood_total;
}

// Returns the one card that can be layed off on a three card rank meld.
int RankMeldLayoff(const std::vector<int> &meld) {
  SPIEL_CHECK_EQ(meld.size(), 3);
  SPIEL_CHECK_TRUE(IsRankMeld(meld));
  std::vector<int> suits = {0, 1, 2, 3};
  for (int card : meld) {
    suits.erase(remove(suits.begin(), suits.end(), CardSuit(card)),
                suits.end());
  }
  return CardRank(meld[0]) + suits[0] * kNumRanks;
}

// Suit melds have two layoffs, except if the meld ends in an ace or king.
std::vector<int> SuitMeldLayoffs(const std::vector<int> &meld) {
  std::vector<int> layoffs;
  int min_card_index = *std::min_element(meld.begin(), meld.end());
  if (CardRank(min_card_index) > 0) {
    layoffs.push_back(min_card_index - 1);
  }
  int max_card_index = *std::max_element(meld.begin(), meld.end());
  if (CardRank(max_card_index) < 12) {
    layoffs.push_back(max_card_index + 1);
  }
  return layoffs;
}

// Finds melds which can be layed legally given a knock card.
// Consider 6s7s8s, 6c7c8c, 8s8c8d. Laying 8s8c8d prevents us from using
// the 6's and 7's in melds, leaving us with 26 points. Laying the two suit
// melds leaves only the 8d for 8 points.
// Returns vector of meld_ids (see MeldToInt).
std::vector<int> LegalMelds(const std::vector<int> &hand, int knock_card) {
  int total_hand_value = TotalCardValue(hand);
  std::set<int> meld_set;
  std::vector<int> hand_(hand);
  std::vector<std::vector<std::vector<int>>> all_meld_groups =
      AllMeldGroups(hand_);
  for (auto meld_group : all_meld_groups) {
    int meld_group_total_value = TotalCardValue(meld_group);
    if (total_hand_value - meld_group_total_value <= knock_card) {
      for (auto meld : meld_group) {
        meld_set.insert(meld_to_int.at(meld));
      }
    }
  }
  std::vector<int> rv(meld_set.begin(), meld_set.end());
  return rv;
}

// Returns the legal discards when a player has knocked. Normally a player can
// discard any card in their hand. When a player knocks, however, they must
// discard a card that preseves the ability to arrange the hand so that the
// total deadwood is less than the knock card.
std::vector<int> LegalDiscards(const std::vector<int> &hand, int knock_card) {
  std::set<int> legal_discards;
  for (int i = 0; i < hand.size(); ++i) {
    std::vector<int> hand_(hand);
    hand_.erase(hand_.begin() + i);
    int deadwood = MinDeadwood(hand_);
    if (deadwood <= knock_card) {
      legal_discards.insert(hand[i]);
    }
  }
  std::vector<int> rv(legal_discards.begin(), legal_discards.end());
  return rv;
}

std::vector<int> AllLayoffs(const std::vector<int> &layed_melds,
                            const std::vector<int> &previous_layoffs) {
  std::set<int> layoffs;
  for (int meld_id : layed_melds) {
    std::vector<int> meld = int_to_meld.at(meld_id);
    if (IsRankMeld(meld) && meld.size() == 3) {
      layoffs.insert(RankMeldLayoff(meld));
    } else if (IsSuitMeld(meld)) {
      std::vector<int> suit_layoffs = SuitMeldLayoffs(meld);
      for (int card : previous_layoffs) {
        if (std::find(suit_layoffs.begin(),
                      suit_layoffs.end(), card) != suit_layoffs.end()) {
          meld.push_back(card);
        }
      }
      suit_layoffs = SuitMeldLayoffs(meld);
      for (int card : suit_layoffs) {
        layoffs.insert(card);
      }
    }
  }
  std::vector<int> rv(layoffs.begin(), layoffs.end());
  return rv;
}

// This mapping should not depend on the order of melds returned by
// AllMelds, which is subject to change.
// See MeldToInt for a description of the mapping.
std::map<std::vector<int>, int> BuildMeldToIntMap() {
  std::map<std::vector<int>, int> rv;
  std::vector<int> full_deck;
  for (int i = 0; i < kNumCards; ++i)
    full_deck.push_back(i);
  std::vector<std::vector<int>> all_melds = AllMelds(full_deck);
  for (int i = 0; i < all_melds.size(); ++i) {
    int meld_id = MeldToInt(all_melds[i]);
    rv.insert(std::pair<std::vector<int>, int>(all_melds[i], meld_id));
  }
  return rv;
}

// Builds the reverse map [0, 185] -> meld.
// May not be fast but only gets run once.
std::map<int, std::vector<int>> BuildIntToMeldMap() {
  std::map<int, std::vector<int>> rv;
  std::vector<int> full_deck;
  for (int i = 0; i < kNumCards; ++i)
    full_deck.push_back(i);
  std::vector<std::vector<int>> all_melds = AllMelds(full_deck);
  for (int i = 0; i < all_melds.size(); ++i) {
    for (auto meld : all_melds) {
      if (MeldToInt(meld) == i) {
        rv.insert(std::pair<int, std::vector<int>>(i, meld));
        break;
      }
    }
  }
  return rv;
}

// Defines a mapping from melds to ints.
// There are 185 distinct melds in total, 65 rank melds and 120 suit melds.
// Rank melds are ordered by ascending rank. For each rank, there are 5 melds.
// The four melds of size 3 are ordered by the suit of the card missing from
// the meld (i.e. 2c2d2h precedes 2s2h2d because the 2s, missing from the first
// meld, precedes the 2c, missing from the second).
// The fifth rank meld is the unique meld containing all four cards of a
// given rank.
// Suit melds are ordered first by size, then by suit (scdh), then by rank.
int MeldToInt(std::vector<int> meld) {
  if (IsRankMeld(meld)) {
    if (meld.size() == 3) {
      std::vector<int> suits;
      for (int i = 0; i < kNumSuits; ++i) suits.push_back(i);
      for (int card : meld) {
        suits.erase(remove(suits.begin(), suits.end(),
                           CardSuit(card)), suits.end());
      }
      return (CardRank(meld[0]) * 5) + suits[0];
    } else if (meld.size() == 4) {
      return (CardRank(meld[0]) * 5) + 4;
    }
  } else if (IsSuitMeld(meld)) {
    std::sort(meld.begin(), meld.end(), CompareRanks);
    if (meld.size() == 3) {
      // 65 rank melds
      return 65 + (CardSuit(meld[0]) * 11) + CardRank(meld[0]);
    } else if (meld.size() == 4) {
      // 44 suit melds of size three
      return 65 + 44 + (CardSuit(meld[0]) * 10) + CardRank(meld[0]);
    } else if (meld.size() == 5) {
      // 40 suit melds of size four
      return 65 + 44 + 40 + (CardSuit(meld[0]) * 9) + CardRank(meld[0]);
    }
  } else {
    SpielFatalError("Not a meld");
  }
}

}  // namespace gin_rummy
}  // namespace open_spiel

