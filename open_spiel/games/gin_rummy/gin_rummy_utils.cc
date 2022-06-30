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

#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"

#include <algorithm>
#include <set>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace gin_rummy {

GinRummyUtils::GinRummyUtils(int num_ranks, int num_suits, int hand_size) :
      num_ranks(num_ranks),
      num_suits(num_suits),
      num_cards(num_ranks * num_suits),
      hand_size(hand_size),
      suit_comp(SuitComparator(num_ranks)),
      rank_comp(RankComparator(num_ranks)),
      int_to_meld(BuildIntToMeldMap()),
      meld_to_int(BuildMeldToIntMap()) {
}

int GinRummyUtils::CardSuit(int card) const { return card / num_ranks; }
int GinRummyUtils::CardRank(int card) const { return card % num_ranks; }

// All suits are of equal value and suit ordering never factors into gameplay.
constexpr char kRankChar[] = "A23456789TJQK";
constexpr char kSuitChar[] = "scdh";

std::string GinRummyUtils::CardString(absl::optional<int> card) const {
  if (!card.has_value()) return "XX";
  SPIEL_CHECK_GE(card.value(), 0);
  SPIEL_CHECK_LT(card.value(), num_cards);
  return {kRankChar[CardRank(card.value())], kSuitChar[CardSuit(card.value())]};
}

int GinRummyUtils::CardInt(std::string card) const {
  SPIEL_CHECK_EQ(card.length(), 2);
  int rank = strchr(kRankChar, card[0]) - kRankChar;
  int suit = strchr(kSuitChar, card[1]) - kSuitChar;
  return suit * num_ranks + rank;
}

std::vector<std::string> GinRummyUtils::CardIntsToCardStrings(
    const VecInt &cards) const {
  std::vector<std::string> rv;
  for (int card : cards) {
    rv.push_back(CardString(card));
  }
  return rv;
}

VecInt GinRummyUtils::CardStringsToCardInts(
    const std::vector<std::string> &cards) const {
  VecInt rv;
  for (const std::string &card : cards) {
    rv.push_back(CardInt(card));
  }
  return rv;
}

// TODO(jhtschultz) should kHandStringSize depend on deck size?
std::string GinRummyUtils::HandToString(const VecInt &cards) const {
  std::string rv;
  constexpr int kHandStringSize = 174;
  rv.reserve(kHandStringSize);
  // Top border
  absl::StrAppend(&rv, "+");
  for (int i = 0; i < num_ranks; ++i)
    absl::StrAppend(&rv, "--");
  absl::StrAppend(&rv, "+\n");
  // One row for each suit
  for (int i = 0; i < num_suits; ++i) {
    absl::StrAppend(&rv, "|");
    for (int j = 0; j < num_ranks; ++j) {
      if (absl::c_linear_search(cards, (i * num_ranks) + j)) {
        absl::StrAppend(&rv, CardString((i * num_ranks) + j));
      } else {
        absl::StrAppend(&rv, "  ");
      }
    }
    absl::StrAppend(&rv, "|\n");
  }
  // Bottom border
  absl::StrAppend(&rv, "+");
  for (int i = 0; i < num_ranks; ++i)
    absl::StrAppend(&rv, "--");
  absl::StrAppend(&rv, "+\n");
  return rv;
}

// Ace = 1, deuce = 2, ... , face cards = 10.
int GinRummyUtils::CardValue(int card_index) const {
  int value = CardRank(card_index) + 1;
  return std::min(10, value);
}

// Sums point total over all cards.
int GinRummyUtils::TotalCardValue(const VecInt &cards) const {
  int total_value = 0;
  for (int card : cards) {
    total_value += CardValue(card);
  }
  return total_value;
}

// Sums point total over all cards.
int GinRummyUtils::TotalCardValue(const VecVecInt &meld_group) const {
  int total_value = 0;
  for (const auto &meld : meld_group) {
    for (auto card : meld) {
      total_value += CardValue(card);
    }
  }
  return total_value;
}

bool GinRummyUtils::IsConsecutive(const VecInt &v) const {
  for (int i = 1; i < v.size(); ++i) {
    if (v[i] != v[i - 1] + 1) return false;
  }
  return true;
}

bool GinRummyUtils::IsRankMeld(const VecInt &cards) const {
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

bool GinRummyUtils::IsSuitMeld(const VecInt &cards) const {
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
  VecInt ranks;
  for (int i = 0; i < cards.size(); ++i) {
    ranks.push_back(CardRank(cards[i]));
  }
  absl::c_sort(ranks);
  return IsConsecutive(ranks);
}

// Returns all possible rank melds that can be formed from cards.
VecVecInt GinRummyUtils::RankMelds(VecInt cards) const {
  VecVecInt melds;
  if (cards.size() < 3) {
    return melds;
  }
  absl::c_sort(cards, rank_comp);
  // First do a sweep for 4 card melds.
  for (int i = 0; i < cards.size() - 3; ++i) {
    // Found 4 card meld - implies there are four 3 card melds as well.
    // We only add two of the 3 card melds here, the other two get added
    // during the 3 card meld sweep.
    if (CardRank(cards[i]) == CardRank(cards[i + 3])) {
      melds.emplace_back(VecInt(cards.begin() + i, cards.begin() + i + 4));
      melds.emplace_back(VecInt{cards[i], cards[i + 1], cards[i + 3]});
      melds.emplace_back(VecInt{cards[i], cards[i + 2], cards[i + 3]});
    }
  }
  // Sweep for 3 card melds.
  for (int i = 0; i < cards.size() - 2; ++i) {
    if (CardRank(cards[i]) == CardRank(cards[i + 2])) {
      melds.emplace_back(VecInt(cards.begin() + i, cards.begin() + i + 3));
    }
  }
  return melds;
}

// Returns all possible suit melds that can be formed from cards.
VecVecInt GinRummyUtils::SuitMelds(VecInt cards) const {
  VecVecInt melds;
  if (cards.size() < 3) {
    return melds;
  }
  absl::c_sort(cards, suit_comp);
  // Find all suit melds of length 5.
  if (cards.size() >= 5) {
    for (int i = 0; i < cards.size() - 4; ++i) {
      if (cards[i] == cards[i + 4] - 4 &&
          CardSuit(cards[i]) == CardSuit(cards[i + 4])) {
        melds.emplace_back(VecInt(cards.begin() + i, cards.begin() + i + 5));
      }
    }
  }
  // Find all suit melds of length 4.
  if (cards.size() >= 4) {
    for (int i = 0; i < cards.size() - 3; ++i) {
      if (cards[i] == cards[i + 3] - 3 &&
          CardSuit(cards[i]) == CardSuit(cards[i + 3])) {
        melds.emplace_back(VecInt(cards.begin() + i, cards.begin() + i + 4));
      }
    }
  }
  // Find all suit melds of length 3.
  for (int i = 0; i < cards.size() - 2; ++i) {
    if (cards[i] == cards[i + 2] - 2 &&
        CardSuit(cards[i]) == CardSuit(cards[i + 2])) {
      melds.emplace_back(VecInt(cards.begin() + i, cards.begin() + i + 3));
    }
  }
  return melds;
}

// Returns all melds of length 5 or less. Any meld of length 6 or more can
// be expressed as two or more melds of shorter length.
VecVecInt GinRummyUtils::AllMelds(const VecInt &cards) const {
  VecVecInt rank_melds = RankMelds(cards);
  VecVecInt suit_melds = SuitMelds(cards);
  rank_melds.insert(rank_melds.end(), suit_melds.begin(), suit_melds.end());
  return rank_melds;
}

bool GinRummyUtils::VectorsIntersect(VecInt *v1, VecInt *v2) const {
  absl::c_sort(*v1);
  absl::c_sort(*v2);
  VecInt::iterator first1 = v1->begin();
  VecInt::iterator last1 = v1->end();
  VecInt::iterator first2 = v2->begin();
  VecInt::iterator last2 = v2->end();

  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2) {
      ++first1;
    } else if (*first2 < *first1) {
      ++first2;
    } else {
      return true;
    }
  }
  return false;
}

// Returns melds which do not share any common cards with given meld.
VecVecInt GinRummyUtils::NonOverlappingMelds(VecInt *meld,
                                             VecVecInt *melds) const {
  VecVecInt rv;
  for (int i = 0; i < melds->size(); ++i) {
    if (!VectorsIntersect(meld, &(*melds)[i])) {
      rv.push_back((*melds)[i]);
    }
  }
  return rv;
}

// Depth first search used by AllMeldGroups.
void GinRummyUtils::AllPaths(VecInt *meld, VecVecInt *all_melds,
                             VecVecInt *path, VecVecVecInt *all_paths) const {
  path->push_back(*meld);
  VecVecInt child_melds = NonOverlappingMelds(meld, all_melds);
  if (child_melds.empty()) {
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
VecVecVecInt GinRummyUtils::AllMeldGroups(const VecInt &cards) const {
  VecVecInt all_melds = AllMelds(cards);
  VecVecVecInt all_meld_groups;
  for (VecInt meld : all_melds) {
    VecVecInt path;
    AllPaths(&meld, &all_melds, &path, &all_meld_groups);
  }
  return all_meld_groups;
}

// "Best" means any meld group that achieves the lowest possible deadwood
// count for the given cards. In general this is non-unique.
VecVecInt GinRummyUtils::BestMeldGroup(const VecInt &cards) const {
  int best_meld_group_total_value = 0;
  VecVecInt best_meld_group;
  VecVecVecInt all_meld_groups = AllMeldGroups(cards);
  for (const auto &meld_group : all_meld_groups) {
    int meld_group_total_value = TotalCardValue(meld_group);
    if (meld_group_total_value > best_meld_group_total_value) {
      best_meld_group_total_value = meld_group_total_value;
      best_meld_group = meld_group;
    }
  }
  return best_meld_group;
}

// Minimum deadwood count over all meld groups.
int GinRummyUtils::MinDeadwood(VecInt hand, absl::optional<int> card) const {
  if (card.has_value()) hand.push_back(card.value());
  return MinDeadwood(hand);
}

// Minimum deadwood count over all meld groups.
int GinRummyUtils::MinDeadwood(const VecInt &hand) const {
  VecInt deadwood = hand;
  VecVecInt best_melds = BestMeldGroup(hand);

  for (const auto &meld : best_melds) {
    for (auto card : meld) {
      deadwood.erase(std::remove(deadwood.begin(), deadwood.end(), card),
                     deadwood.end());
    }
  }
  // If we have just drawn a card, we can discard the one worth the most points.
  if (hand.size() == hand_size + 1 && !deadwood.empty()) {
    absl::c_sort(deadwood, rank_comp);
    deadwood.pop_back();
  }
  int deadwood_total = 0;
  for (int card : deadwood) deadwood_total += CardValue(card);
  return deadwood_total;
}

// Returns the unique card that can be layed off on a given 3-card rank meld.
int GinRummyUtils::RankMeldLayoff(const VecInt &meld) const {
  SPIEL_CHECK_EQ(meld.size(), 3);
  SPIEL_CHECK_TRUE(IsRankMeld(meld));
  VecInt suits = {0, 1, 2, 3};
  for (int card : meld) {
    suits.erase(std::remove(suits.begin(), suits.end(), CardSuit(card)),
                suits.end());
  }
  return CardRank(meld[0]) + suits[0] * num_ranks;
}

// Suit melds have two layoffs, except if the meld ends in an ace or king.
VecInt GinRummyUtils::SuitMeldLayoffs(const VecInt &meld) const {
  VecInt layoffs;
  int min_card_index = *std::min_element(meld.begin(), meld.end());
  if (CardRank(min_card_index) > 0) {
    layoffs.push_back(min_card_index - 1);
  }
  int max_card_index = *std::max_element(meld.begin(), meld.end());
  if (CardRank(max_card_index) < num_ranks - 1) {
    layoffs.push_back(max_card_index + 1);
  }
  return layoffs;
}

// Finds melds which can be layed legally given a knock card.
// Consider 6s7s8s, 6c7c8c, 8s8c8d. Laying 8s8c8d prevents us from using
// the 6's and 7's in melds, leaving us with 26 points. Laying the two suit
// melds leaves only the 8d for 8 points.
// Returns vector of meld_ids (see MeldToInt).
VecInt GinRummyUtils::LegalMelds(const VecInt &hand, int knock_card) const {
  int total_hand_value = TotalCardValue(hand);
  std::set<int> meld_set;
  VecInt hand_(hand);
  VecVecVecInt all_meld_groups = AllMeldGroups(hand_);
  for (const auto &meld_group : all_meld_groups) {
    int meld_group_total_value = TotalCardValue(meld_group);
    if (total_hand_value - meld_group_total_value <= knock_card) {
      for (const auto &meld : meld_group) {
        meld_set.insert(meld_to_int.at(meld));
      }
    }
  }
  return VecInt(meld_set.begin(), meld_set.end());
}

// Returns the legal discards when a player has knocked. Normally a player can
// discard any card in their hand. When a player knocks, however, they must
// discard a card that preseves the ability to arrange the hand so that the
// total deadwood is less than the knock card.
VecInt GinRummyUtils::LegalDiscards(const VecInt &hand, int knock_card) const {
  std::set<int> legal_discards;
  for (int i = 0; i < hand.size(); ++i) {
    VecInt hand_(hand);
    hand_.erase(hand_.begin() + i);
    int deadwood = MinDeadwood(hand_);
    if (deadwood <= knock_card) {
      legal_discards.insert(hand[i]);
    }
  }
  return VecInt(legal_discards.begin(), legal_discards.end());
}

VecInt GinRummyUtils::AllLayoffs(const VecInt &layed_melds,
                         const VecInt &previous_layoffs) const {
  std::set<int> layoffs;
  for (int meld_id : layed_melds) {
    VecInt meld = int_to_meld.at(meld_id);
    if (IsRankMeld(meld) && meld.size() == 3) {
      layoffs.insert(RankMeldLayoff(meld));
    } else if (IsSuitMeld(meld)) {
      VecInt suit_layoffs = SuitMeldLayoffs(meld);
      for (int card : previous_layoffs) {
        if (absl::c_linear_search(suit_layoffs, card)) {
          meld.push_back(card);
        }
      }
      suit_layoffs = SuitMeldLayoffs(meld);
      for (int card : suit_layoffs) {
        layoffs.insert(card);
      }
    }
  }
  return VecInt(layoffs.begin(), layoffs.end());
}

// This mapping should not depend on the order of melds returned by
// AllMelds, which is subject to change.
// See MeldToInt for a description of the mapping.
std::map<VecInt, int> GinRummyUtils::BuildMeldToIntMap() const {
  std::map<VecInt, int> rv;
  VecInt full_deck;
  for (int i = 0; i < num_cards; ++i) full_deck.push_back(i);
  VecVecInt all_melds = AllMelds(full_deck);
  for (int i = 0; i < all_melds.size(); ++i) {
    int meld_id = MeldToInt(all_melds[i]);
    rv.insert(std::pair<VecInt, int>(all_melds[i], meld_id));
  }
  return rv;
}

// Builds the reverse map [0, 185] -> meld.
// May not be fast but only gets run once.
std::map<int, VecInt> GinRummyUtils::BuildIntToMeldMap() const {
  const int kNumCards = 52;
  std::map<int, VecInt> rv;
  VecInt full_deck;
  for (int i = 0; i < kNumCards; ++i) full_deck.push_back(i);
  VecVecInt all_melds = AllMelds(full_deck);
  for (int i = 0; i < all_melds.size(); ++i) {
    for (const auto &meld : all_melds) {
      if (MeldToInt(meld) == i) {
        rv.insert(std::pair<int, VecInt>(i, meld));
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
int GinRummyUtils::MeldToInt(VecInt meld) const {
  const int kNumRanks = 13;
  const int kNumSuits = 4;
  if (IsRankMeld(meld)) {
    if (meld.size() == 3) {
      VecInt suits;
      for (int i = 0; i < kNumSuits; ++i) suits.push_back(i);
      for (int card : meld) {
        suits.erase(std::remove(suits.begin(), suits.end(), CardSuit(card)),
                    suits.end());
      }
      return (CardRank(meld[0]) * 5) + suits[0];
    } else if (meld.size() == 4) {
      return (CardRank(meld[0]) * 5) + 4;
    }
    SpielFatalError("Impossible meld size");
  } else if (IsSuitMeld(meld)) {
    absl::c_sort(meld, rank_comp);
    int offset = 65;  // 65 rank melds
    if (meld.size() == 3) {
      return offset + (CardSuit(meld[0]) * (kNumRanks - 2)) +
             CardRank(meld[0]);
    }
    offset += 44;  // 44 suit melds of size three
    if (meld.size() == 4) {
      return offset + (CardSuit(meld[0]) * (kNumRanks - 3)) +
             CardRank(meld[0]);
    }
    offset += 40;  // 40 suit melds of size four
    if (meld.size() == 5) {
      return offset + (CardSuit(meld[0]) * (kNumRanks - 4)) +
             CardRank(meld[0]);
    }
    SpielFatalError("Impossible meld size");
  } else {
    SpielFatalError("Not a meld");
  }
}

}  // namespace gin_rummy
}  // namespace open_spiel
