// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/dou_dizhu/dou_dizhu_utils.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace dou_dizhu {

// dropping suit information
int CardToRank(int card) {
  if (card == kNumCards - 2 || card == kNumCards - 1) {
    return card - kNumCards + kNumRanks;
  }
  return card % (kNumRanks - 2);
}

int CardToSuit(int card) {
  if (card == kNumCards - 2 || card == kNumCards - 1) {
    SpielFatalError("No Suit defined for Jokers");
  }
  return card / (kNumRanks - 2);
}

std::string RankString(int rank) {
  if (rank < kNumRanks - 2)
    return std::string(1, kRankChar[rank]);
  else if (rank == kNumRanks - 2)
    return "(BWJ)";
  else if (rank == kNumRanks - 1)
    return "(CJ)";
  else
    SpielFatalError("Non valid rank");
}

std::string CardString(int card) {
  int rank = CardToRank(card);
  if (rank >= kNumRanks - 2) {
    return RankString(rank);
  } else {
    int suit = CardToSuit(card);
    SPIEL_CHECK_GE(suit, 0);
    SPIEL_CHECK_LT(suit, kNumSuits);
    return absl::StrFormat("%c%c", kSuitChar[suit], kRankChar[rank]);
  }
}

std::string FormatSingleHand(absl::Span<const int> hand) {
  std::string hand_format;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    for (int i = 0; i < hand[rank]; ++i)
      absl::StrAppend(&hand_format, RankString(rank));
  }
  return hand_format;
}

// resolve ambiguity for cases like 333444555666
std::string FormatAirplaneCombHand(int action) {
  TrioCombParams params = GetAirplaneCombParams(action);
  std::array<int, kNumRanks> hand = ActionToHand(action);
  std::string airplane_comb_str;
  // specify which is chain
  for (int rank = params.chain_head;
       rank < params.chain_head + params.chain_length; ++rank) {
    for (int i = 0; i < 3; ++i)
      absl::StrAppend(&airplane_comb_str, RankString(rank));
  }
  absl::StrAppend(&airplane_comb_str, "-");
  // kickers
  for (int rank = 0; rank < kNumRanks; ++rank) {
    if (rank >= params.chain_head &&
        rank < params.chain_head + params.chain_length)
      continue;
    if (!hand[rank]) continue;
    for (int i = 0; i < hand[rank]; ++i)
      absl::StrAppend(&airplane_comb_str, RankString(rank));
  }
  return airplane_comb_str;
}

// Shared by single-rank and chain-only hands
int GetNumCardsPerRank(int action) {
  int num_cards;
  if (action >= kPlayActionBase && action < kPairActionBase) {
    num_cards = 1;
  } else if (action >= kPairActionBase && action < kTrioActionBase) {
    num_cards = 2;
  } else if ((action >= kTrioActionBase && action < kTrioWithSoloActionBase) ||
             (action >= kAirplaneActionBase &&
              action < kAirplaneWithSoloActionBase)) {
    num_cards = 3;
  } else if (action >= kBombActionBase && action < kRocketActionBase) {
    num_cards = 4;
  } else {
    SpielFatalError("Invalid action ID");
  }

  return num_cards;
}

int GetSingleRankActionBase(int num_cards_same_rank = 1) {
  int action_base;
  switch (num_cards_same_rank) {
    case 1:
      action_base = kPlayActionBase;
      break;
    case 2:
      action_base = kPairActionBase;
      break;
    case 3:
      action_base = kTrioActionBase;
      break;
    case 4:
      action_base = kBombActionBase;
      break;
    default:
      SpielFatalError(
          "The number of cards of the same rank is wrong (single rank).");
  }
  return action_base;
}

SingleRankHandParams GetSingleRankHandParams(int action) {
  const int num_cards = GetNumCardsPerRank(action);
  const int action_base = GetSingleRankActionBase(num_cards);
  SPIEL_CHECK_GE(action, action_base);
  return SingleRankHandParams(action - action_base, num_cards);
}

std::array<int, kNumRanks> SingleRankHand(int action) {
  std::array<int, kNumRanks> hand{};
  SingleRankHandParams params = GetSingleRankHandParams(action);
  hand[params.rank] = params.num_cards;
  return hand;
}

// given a single-rank hand, map it to action id
int SingleRankHandToActionId(absl::Span<const int> hand) {
  int the_rank;
  int counter = 0;

  for (int rank = 0; rank < kNumRanks; ++rank) {
    if (hand[rank] != 0) {
      the_rank = rank;
      counter++;
    }
  }
  SPIEL_CHECK_EQ(counter, 1);
  const int num_cards_same_rank = hand[the_rank];
  int action = GetSingleRankActionBase(num_cards_same_rank);
  action += the_rank;
  return action;
}

// given an arbitrary hand, search for possible single-rank hands
// if prev_action = kInvalidAction, search for all possible such hands
// otherwise, only search for those that are ranked higher than prev_action
void SearchSingleRankActions(std::vector<Action>* actions,
                             absl::Span<const int> hand,
                             int prev_action = kInvalidAction) {
  std::array<int, kNumRanks> used_hands{};
  SingleRankHandParams prev_action_params;
  int start_rank;
  if (prev_action == kInvalidAction) {
    start_rank = 0;
  } else {
    prev_action_params = GetSingleRankHandParams(prev_action);
    start_rank = prev_action_params.rank + 1;
  }
  for (int rank = start_rank; rank < kNumRanks; ++rank) {
    SPIEL_CHECK_LE(hand[rank], kNumSuits);
    SPIEL_CHECK_GE(hand[rank], 0);
    if (rank == kNumRanks - 2 || rank == kNumRanks - 1)
      SPIEL_CHECK_LE(hand[rank], 1);
    if (prev_action == kInvalidAction) {
      for (int i = 0; i < hand[rank]; ++i) {
        used_hands[rank]++;
        actions->push_back(SingleRankHandToActionId(used_hands));
      }
    } else if (hand[rank] >= prev_action_params.num_cards) {
      used_hands[rank] = prev_action_params.num_cards;
      actions->push_back(SingleRankHandToActionId(used_hands));
    }
    used_hands[rank] = 0;
  }
}

int GetChainOnlyActionBase(int num_cards_same_rank = 1) {
  int action_base;
  switch (num_cards_same_rank) {
    case 1:
      action_base = kSoloChainActionBase;
      break;
    case 2:
      action_base = kPairChainActionBase;
      break;
    case 3:
      action_base = kAirplaneActionBase;
      break;
    default:
      SpielFatalError("The number of cards of the same rank is wrong (chain).");
  }
  return action_base;
}

int GetChainOnlyMinLength(int num_cards_same_rank = 1) {
  int chain_length;
  switch (num_cards_same_rank) {
    case 1:
      chain_length = kSoloChainMinLength;
      break;
    case 2:
      chain_length = kPairChainMinLength;
      break;
    case 3:
      chain_length = kAirplaneMinLength;
      break;
    default:
      SpielFatalError("The number of cards of the same rank is wrong (chain).");
  }
  return chain_length;
}

ChainOnlyHandParams GetChainOnlyHandParams(int action) {
  const int num_cards_same_rank = GetNumCardsPerRank(action);
  const int action_base = GetChainOnlyActionBase(num_cards_same_rank);
  const int min_length = GetChainOnlyMinLength(num_cards_same_rank);
  SPIEL_CHECK_GE(action, action_base);
  const int hand_id = action - action_base;
  int chain_length = min_length;
  int base = 0;
  // we label the action Ids by increasing length of the chain
  for (chain_length = min_length; chain_length <= kNumRanks; ++chain_length) {
    int num_chains = kNumRanks - chain_length - 2;
    if (base <= hand_id && hand_id < base + num_chains) break;
    base += num_chains;
  }
  const int chain_head = hand_id - base;
  return ChainOnlyHandParams(chain_head, num_cards_same_rank, chain_length);
}

std::array<int, kNumRanks> ChainOnlyHand(int action) {
  std::array<int, kNumRanks> hand{};
  ChainOnlyHandParams params = GetChainOnlyHandParams(action);
  for (int i = 0; i < params.chain_length; ++i) {
    hand[params.chain_head + i] = params.num_cards_per_rank;
  }
  return hand;
}

int ChainOnlyHandToActionId(absl::Span<const int> hand) {
  int chain_head = -1;
  int chain_length = 0;
  int chain_counter = 0;
  int num_cards_same_rank = 0;
  bool chain_stopped = true;

  if (hand[kNumRanks - 3] || hand[kNumRanks - 2] || hand[kNumRanks - 1])
    SpielFatalError("2s and Jokers cannot be in a chain");

  for (int rank = 0; rank < kNumRanks - 3; ++rank) {
    if (hand[rank] == 0) {
      chain_stopped = true;
    } else {
      if (chain_stopped) {
        chain_head = rank;
        num_cards_same_rank = hand[rank];
        chain_length = 1;
        chain_counter++;
        chain_stopped = false;
      } else if (hand[rank] != num_cards_same_rank) {
        SpielFatalError("Invalid pattern");
      } else {
        chain_length++;
      }
    }
  }

  SPIEL_CHECK_EQ(chain_counter, 1);
  const int min_length = GetChainOnlyMinLength(num_cards_same_rank);
  const int action_base = GetChainOnlyActionBase(num_cards_same_rank);

  if (chain_length < min_length)
    SpielFatalError(absl::StrFormat("The length of chain should be at least %d",
                                    min_length));
  int action = action_base;
  for (int length = min_length; length < chain_length; ++length)
    action += kNumRanks - length - 2;
  action += chain_head;
  return action;
}

void SearchChainOnlyActions(std::vector<Action>* actions,
                            absl::Span<const int> hand,
                            int prev_action = kInvalidAction) {
  ChainOnlyHandParams prev_action_params;

  int start_rank;
  if (prev_action == kInvalidAction) {
    start_rank = 0;
  } else {
    prev_action_params = GetChainOnlyHandParams(prev_action);
    start_rank = prev_action_params.chain_head + 1;
  }

  for (int chain_head = start_rank; chain_head < kNumRanks - 4; ++chain_head) {
    if (!hand[chain_head] || hand[chain_head] == kNumSuits) continue;
    int num_cards = hand[chain_head];
    // 2-s and Jokers cannot be in chain
    for (int chain_length = 2; chain_head + chain_length - 1 < kNumRanks - 3;
         ++chain_length) {
      int chain_tail = chain_head + chain_length - 1;
      num_cards = std::min(num_cards, hand[chain_tail]);
      if (!num_cards) break;
      std::vector<int> all_nums;
      if (prev_action != kInvalidAction) {
        if (num_cards < prev_action_params.num_cards_per_rank) break;
        if (chain_length > prev_action_params.chain_length) break;
        if (chain_length == prev_action_params.chain_length) {
          all_nums.push_back(prev_action_params.num_cards_per_rank);
        }
      } else {
        for (int n = 1; n <= num_cards; ++n) {
          all_nums.push_back(n);
        }
      }

      for (auto n : all_nums) {
        const int min_length = GetChainOnlyMinLength(n);
        if (chain_length >= min_length) {
          std::array<int, kNumRanks> used_rank{};
          for (int i = 0; i < chain_length; ++i) used_rank[chain_head + i] = n;
          actions->push_back(ChainOnlyHandToActionId(used_rank));
        }
      }
    }
  }
}

int GetTrioCombActionBase(int action) {
  int action_base;
  if (kTrioWithSoloActionBase <= action && action < kTrioWithPairActionBase) {
    action_base = kTrioWithSoloActionBase;
  } else if (kTrioWithPairActionBase <= action &&
             action < kAirplaneActionBase) {
    action_base = kTrioWithPairActionBase;
  } else if (kAirplaneWithSoloActionBase <= action &&
             action < kAirplaneWithPairActionBase) {
    action_base = kAirplaneWithSoloActionBase;
  } else if (kAirplaneWithPairActionBase <= action &&
             action < kBombActionBase) {
    action_base = kAirplaneWithPairActionBase;
  } else {
    SpielFatalError("Invalid action Ids");
  }
  return action_base;
}

KickerType GetTrioCombKickerType(int action) {
  KickerType kicker_type;
  if (kTrioWithSoloActionBase <= action && action < kTrioWithPairActionBase) {
    kicker_type = kSolo;
  } else if (kTrioWithPairActionBase <= action &&
             action < kAirplaneActionBase) {
    kicker_type = kPair;
  } else if (kAirplaneWithSoloActionBase <= action &&
             action < kAirplaneWithPairActionBase) {
    kicker_type = kSolo;
  } else if (kAirplaneWithPairActionBase <= action &&
             action < kBombActionBase) {
    kicker_type = kPair;
  } else {
    SpielFatalError("Invalid action Ids");
  }
  return kicker_type;
}

// single trio comb includes trio+solo and trio+pair (excluding airplanes)
TrioCombParams GetSingleTrioCombParams(int action) {
  if (action < kTrioWithSoloActionBase || action >= kAirplaneActionBase)
    SpielFatalError("Must be single trio pattern");

  const int action_base = GetTrioCombActionBase(action);
  const KickerType kicker_type = GetTrioCombKickerType(action);
  const int hand_id = (action - action_base);
  const int num_kickers = kicker_type == kSolo ? kNumRanks - 1 : kNumRanks - 3;
  const int head = hand_id / num_kickers;
  const int kicker_steps = hand_id % num_kickers;

  return TrioCombParams(head, 1, kicker_type, kicker_steps);
}

int GetNumKickersAirplaneSoloComb(int chain_length) {
  int num_comb;
  switch (chain_length) {
    case 2:
      num_comb = kNumKickersAirplaneSoloCombChainOfLengthTwo;
      break;

    case 3:
      num_comb = kNumKickersAirplaneSoloCombChainOfLengthThree;
      break;

    case 4:
      num_comb = kNumKickersAirplaneSoloCombChainOfLengthFour;
      break;

    case 5:
      num_comb = kNumKickersAirplaneSoloCombChainOfLengthFive;
      break;

    default:
      SpielFatalError("The chain length for aiplane+solo must be within 2-5");
      break;
  }
  return num_comb;
}

int GetAirplaneSoloActionBase(int chain_length) {
  int action_base;
  switch (chain_length) {
    case 2:
      action_base = kAirplaneWithSoloActionBase;
      break;

    case 3:
      action_base = kAirplaneWithSoloActionBase + 968;
      break;

    case 4:
      action_base = kAirplaneWithSoloActionBase + 4268;
      break;

    case 5:
      action_base = kAirplaneWithSoloActionBase + 11612;
      break;

    default:
      SpielFatalError("The chain length for aiplane+solo must be within 2-5");
      break;
  }
  return action_base;
}

int GetNumKickersAirplanePairComb(int chain_length) {
  int num_comb;
  switch (chain_length) {
    case 2:
      num_comb = kNumKickersAirplanePairCombChainOfLengthTwo;
      break;

    case 3:
      num_comb = kNumKickersAirplanePairCombChainOfLengthThree;
      break;

    case 4:
      num_comb = kNumKickersAirplanePairCombChainOfLengthFour;
      break;

    default:
      SpielFatalError("The chain length for aiplane+Pair must be within 2-4");
      break;
  }
  return num_comb;
}

int GetAirplanePairActionBase(int chain_length) {
  int action_base;
  switch (chain_length) {
    case 2:
      action_base = kAirplaneWithPairActionBase;
      break;

    case 3:
      action_base = kAirplaneWithPairActionBase + 605;
      break;

    case 4:
      action_base = kAirplaneWithPairActionBase + 1805;
      break;
    default:
      SpielFatalError("The chain length for aiplane+Pair must be within 2-4");
      break;
  }
  return action_base;
}

TrioCombParams GetAirplaneCombParams(int action) {
  if (action < kAirplaneWithSoloActionBase || action >= kBombActionBase)
    SpielFatalError("Must be airplane pattern");

  int action_base = kInvalidAction;
  KickerType kicker_type;

  SPIEL_CHECK_GE(action, kAirplaneWithSoloActionBase);
  SPIEL_CHECK_LT(action, kBombActionBase);
  int start_length = 2, end_length, end_base;

  int (*GetActionBaseFunc)(int), (*GetKickersNumFunc)(int);
  if (kAirplaneWithSoloActionBase <= action &&
      action < kAirplaneWithPairActionBase) {
    kicker_type = kSolo;
    GetActionBaseFunc = &GetAirplaneSoloActionBase;
    GetKickersNumFunc = &GetNumKickersAirplaneSoloComb;
    end_length = 5;
    end_base = kAirplaneWithPairActionBase;
  } else {
    kicker_type = kPair;
    GetActionBaseFunc = &GetAirplanePairActionBase;
    GetKickersNumFunc = &GetNumKickersAirplanePairComb;
    end_length = 4;
    end_base = kBombActionBase;
  }
  int chain_length;
  // label the action Ids in increasing length of chain
  for (chain_length = start_length; chain_length <= end_length;
       ++chain_length) {
    int start_base = GetActionBaseFunc(chain_length);
    int next_base = chain_length == end_length
                        ? end_base
                        : GetActionBaseFunc(chain_length + 1);
    if (start_base <= action && action < next_base) {
      action_base = start_base;
      break;
    }
  }
  const int hand_id = (action - action_base);
  const int num_kickers = GetKickersNumFunc(chain_length);
  const int chain_head = hand_id / num_kickers;
  const int kicker_steps = hand_id % num_kickers;
  SPIEL_CHECK_FALSE(action_base == kInvalidAction);
  return TrioCombParams(chain_head, chain_length, kicker_type, kicker_steps);
}

std::array<int, kNumRanks> SingleTrioCombHand(int action) {
  std::array<int, kNumRanks> hand{};

  TrioCombParams params = GetSingleTrioCombParams(action);

  hand[params.chain_head] = 3;
  const int kicker_steps = params.kicker_id;
  int kicker_rank, counter = 0;

  for (kicker_rank = 0; kicker_rank < kNumRanks; ++kicker_rank) {
    // kicker cannot be the same rank as trio
    if (kicker_rank == params.chain_head) continue;
    if (counter++ == kicker_steps) break;
  }

  hand[kicker_rank] = (params.kicker_type == kSolo ? 1 : 2);
  return hand;
}

int SingleTrioCombHandToActionId(absl::Span<const int> hand) {
  int trio_rank, kicker_rank;
  int trio_counter = 0, kicker_counter = 0;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    if (hand[rank] == 3) {
      trio_counter++;
      trio_rank = rank;
    } else if (hand[rank] == 1 || hand[rank] == 2) {
      kicker_counter++;
      kicker_rank = rank;
    } else if (hand[rank] == 4) {
      SpielFatalError("There cannot be a bomb");
    }
  }
  SPIEL_CHECK_EQ(trio_counter, 1);
  SPIEL_CHECK_EQ(kicker_counter, 1);

  int action;
  if (hand[kicker_rank] == 1)
    action = kTrioWithSoloActionBase;
  else
    action = kTrioWithPairActionBase;
  // one of the rank had already been taken by the trio
  if (hand[kicker_rank] == 1)
    action += trio_rank * (kNumRanks - 1);
  else
    action += trio_rank * (kNumRanks - 3);  // the jokers cannot be the pair
  int kicker_steps = 0;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    if (rank == trio_rank) continue;
    if (rank == kicker_rank) break;
    kicker_steps++;
  }
  action += kicker_steps;
  return action;
}

void SearchSingleTrioCombActions(std::vector<Action>* actions,
                                 absl::Span<const int> hand,
                                 int prev_action = kInvalidAction) {
  TrioCombParams prev_action_params;
  int start_rank;
  if (prev_action == kInvalidAction) {
    start_rank = 0;
  } else {
    prev_action_params = GetSingleTrioCombParams(prev_action);
    start_rank = prev_action_params.chain_head + 1;
  }
  // enumerate possible trio
  for (int rank = start_rank; rank < kNumRanks - 2; ++rank) {
    if (hand[rank] < 3) continue;
    for (int kicker = 0; kicker < kNumRanks; ++kicker) {
      if (!hand[kicker] || kicker == rank) continue;
      std::vector<KickerType> all_kicker_types;
      if (prev_action != kInvalidAction) {
        if (hand[kicker] >= prev_action_params.kicker_type)
          all_kicker_types.push_back(prev_action_params.kicker_type);
      } else {
        for (int i = 1; i <= std::min(hand[kicker], 2); ++i)
          all_kicker_types.push_back(static_cast<KickerType>(i));
      }
      for (auto n : all_kicker_types) {
        std::array<int, kNumRanks> used_hand{};
        used_hand[rank] = 3;
        used_hand[kicker] = static_cast<int>(n);
        actions->push_back(SingleTrioCombHandToActionId(used_hand));
      }
    }
  }
}

// a dfs backtrack algorithm to compute action ids / hands for airplane
// combinations if target_count = -1, then the goal of this algorithm is to find
// the kicker_id of ans_hand, stored in count reference otherwise, the goal is
// to find a hand whose kicker_id is target_count and the result hand is stored
// in ans_hand
bool dfs_airplane_kicker(int chain_length, int depth, int target_count,
                         int& count, int max_search_rank,
                         absl::Span<int> used_rank, absl::Span<int> ans_hand,
                         KickerType kicker_type) {
  if (chain_length == depth) {
    if (target_count == -1) {
      bool found = true;
      for (int rank = 0; rank < kNumRanks; ++rank)
        found = found & (used_rank[rank] == ans_hand[rank]);
      if (found) return true;
    } else if (target_count == count) {
      for (int rank = 0; rank < kNumRanks; ++rank)
        ans_hand[rank] = used_rank[rank];
      return true;
    }
    count++;
  } else {
    for (int rank = 0; rank <= max_search_rank; ++rank) {
      SPIEL_CHECK_NE(used_rank[rank], kNumSuits);
      if (used_rank[rank] == 3) continue;
      if (kicker_type == kPair) {
        SPIEL_CHECK_NE(used_rank[rank], 1);
        if (used_rank[rank] == 2) continue;
      }
      if (rank == kNumRanks - 1 || rank == kNumRanks - 2) {
        if (kicker_type == kPair) continue;
        if (used_rank[rank]) continue;
        // Rocket cannot be kickers
        if (used_rank[2 * kNumRanks - 3 - rank]) continue;
      }
      used_rank[rank] += kicker_type == kSolo ? 1 : 2;
      if (dfs_airplane_kicker(chain_length, depth + 1, target_count, count,
                              rank, used_rank, ans_hand, kicker_type))
        return true;
      used_rank[rank] -= kicker_type == kSolo ? 1 : 2;
    }
  }
  return false;
}

std::array<int, kNumRanks> AirplaneCombHand(int action) {
  std::array<int, kNumRanks> hand{};
  std::array<int, kNumRanks> used_rank{};
  SPIEL_CHECK_GE(action, kAirplaneWithSoloActionBase);
  SPIEL_CHECK_LT(action, kBombActionBase);
  TrioCombParams params = GetAirplaneCombParams(action);
  for (int i = 0; i < params.chain_length; ++i) {
    hand[params.chain_head + i] = used_rank[params.chain_head + i] = 3;
  }
  const int kicker_steps = params.kicker_id;
  int count = 0;
  bool found = dfs_airplane_kicker(params.chain_length, 0, kicker_steps, count,
                                   kNumRanks - 1, absl::MakeSpan(used_rank),
                                   absl::MakeSpan(hand), params.kicker_type);
  SPIEL_CHECK_TRUE(found);
  return hand;
}

// for aiplane combination, we have to specify the chain head
// to resolve ambiguity such as 333444555666
int AirplaneCombHandToActionId(absl::Span<const int> hand, int chain_head,
                               KickerType kicker_type) {
  int chain_length = 0;
  bool chain_begun = false;
  std::vector<int> kickers;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    SPIEL_CHECK_LT(hand[rank], kNumSuits);
    if (!hand[rank]) continue;
    if (!chain_begun && rank != chain_head) {
      if (kicker_type == kSolo) {
        for (int i = 0; i < hand[rank]; ++i) {
          kickers.push_back(rank);
        }
      } else {
        SPIEL_CHECK_EQ(hand[rank], 2);
        kickers.push_back(rank);
      }
    } else if (rank == chain_head) {
      SPIEL_CHECK_EQ(hand[rank], 3);
      chain_begun = true;
      chain_length++;
    } else if (chain_begun && hand[rank] == 3) {
      chain_length++;
    } else if (chain_begun && hand[rank] != 3) {
      chain_begun = false;
      if (kicker_type == kSolo) {
        for (int i = 0; i < hand[rank]; ++i) kickers.push_back(rank);
      } else {
        SPIEL_CHECK_EQ(hand[rank], 2);
        kickers.push_back(rank);
      }
    }
  }

  // handle case where 333444555666 and chain_head=3
  // in this case, the above linear scan algorithm will view 3-4-5-6 as the
  // chain where 6s should be the kickers
  if (chain_length - 1 == static_cast<int>(kickers.size()) + 3) {
    chain_length--;
    for (int i = 0; i < 3; ++i) kickers.push_back(chain_head + chain_length);
  }
  SPIEL_CHECK_EQ(chain_length, static_cast<int>(kickers.size()));

  if (chain_head + chain_length - 1 >= kNumRanks - 3)
    SpielFatalError("2s, Joker cannot be in a chain");
  int action_base;
  if (kicker_type == kSolo)
    action_base = GetAirplaneSoloActionBase(chain_length) +
                  chain_head * GetNumKickersAirplaneSoloComb(chain_length);
  else
    action_base = GetAirplanePairActionBase(chain_length) +
                  chain_head * GetNumKickersAirplanePairComb(chain_length);

  int count = 0;
  std::array<int, kNumRanks> used_rank{};
  for (int i = 0; i < chain_length; ++i) used_rank[chain_head + i] = 3;

  std::array<int, kNumRanks> hand_copy{};
  for (int i = 0; i < kNumRanks; ++i) hand_copy[i] = hand[i];
  bool found = dfs_airplane_kicker(chain_length, 0, -1, count, kNumRanks - 1,
                                   absl::MakeSpan(used_rank),
                                   absl::MakeSpan(hand_copy), kicker_type);
  SPIEL_CHECK_TRUE(found);

  return action_base + count;
}

// a dfs backtrack algorithm that found the action ids of all possible airplane
// combination the action ids are stored in action_ids
void dfs_add_all_airplane_kickers(int chain_head, int chain_length, int depth,
                                  int max_search_rank,
                                  absl::Span<int> used_rank,
                                  absl::Span<const int> ans_hand,
                                  std::vector<Action>* action_ids,
                                  KickerType kicker_type) {
  if (chain_length == depth) {
    std::array<int, kNumRanks> final_hand{};
    for (int i = 0; i < kNumRanks; ++i) final_hand[i] = used_rank[i];
    action_ids->push_back(static_cast<Action>(
        AirplaneCombHandToActionId(final_hand, chain_head, kicker_type)));
  } else {
    for (int rank = 0; rank <= max_search_rank; ++rank) {
      if (rank >= chain_head && rank <= chain_head + chain_length - 1) continue;
      SPIEL_CHECK_NE(used_rank[rank], kNumSuits);
      if (used_rank[rank] == 3) continue;
      if (kicker_type == kPair) {
        SPIEL_CHECK_NE(used_rank[rank], 1);
        if (used_rank[rank] == 2) continue;
      }
      if (rank == kNumRanks - 1 || rank == kNumRanks - 2) {
        if (kicker_type == kPair) continue;
        if (used_rank[rank]) continue;
        if (used_rank[2 * kNumRanks - 3 - rank]) continue;
      }
      int num_use_cards = kicker_type == kSolo ? 1 : 2;
      if (ans_hand[rank] < num_use_cards + used_rank[rank]) continue;
      used_rank[rank] += num_use_cards;
      dfs_add_all_airplane_kickers(chain_head, chain_length, depth + 1, rank,
                                   used_rank, ans_hand, action_ids,
                                   kicker_type);
      used_rank[rank] -= num_use_cards;
    }
  }
}

void SearchAirplaneCombActions(std::vector<Action>* actions,
                               absl::Span<const int> hand,
                               int prev_action = kInvalidAction) {
  TrioCombParams prev_action_params;
  int start_rank;
  if (prev_action == kInvalidAction) {
    start_rank = 0;
  } else {
    prev_action_params = GetAirplaneCombParams(prev_action);
    start_rank = prev_action_params.chain_head + 1;
  }
  for (int chain_head = start_rank; chain_head < kNumRanks - 4; ++chain_head) {
    if (hand[chain_head] < 3) continue;
    int num_cards = hand[chain_head];
    for (int chain_length = 2; chain_head + chain_length - 1 < kNumRanks - 3;
         ++chain_length) {
      int chain_tail = chain_head + chain_length - 1;
      num_cards = std::min(num_cards, hand[chain_tail]);
      if (num_cards < 3) break;
      std::vector<KickerType> all_kicker_types;
      if (prev_action != kInvalidAction) {
        if (chain_length > prev_action_params.chain_length) break;
        if (chain_length == prev_action_params.chain_length) {
          all_kicker_types.push_back(prev_action_params.kicker_type);
        }
      } else {
        all_kicker_types.push_back(kSolo);
        all_kicker_types.push_back(kPair);
      }
      for (auto kicker_type : all_kicker_types) {
        std::array<int, kNumRanks> used_hand{};
        for (int i = 0; i < chain_length; ++i) used_hand[chain_head + i] = 3;
        dfs_add_all_airplane_kickers(chain_head, chain_length, 0, kNumRanks - 1,
                                     absl::MakeSpan(used_hand),
                                     absl::MakeSpan(hand), actions,
                                     kicker_type);
      }
    }
  }
}

std::array<int, kNumRanks> ActionToHand(int action) {
  std::array<int, kNumRanks> hand{};
  if ((action >= kPlayActionBase && action < kSoloChainActionBase) ||
      (action >= kPairActionBase && action < kPairChainActionBase) ||
      (action >= kTrioActionBase && action < kTrioWithSoloActionBase) ||
      (action >= kBombActionBase && action < kRocketActionBase)) {
    hand = SingleRankHand(action);
  } else if ((action >= kSoloChainActionBase && action < kPairActionBase) ||
             (action >= kPairChainActionBase && action < kTrioActionBase) ||
             (action >= kAirplaneActionBase &&
              action < kAirplaneWithSoloActionBase)) {
    hand = ChainOnlyHand(action);
  } else if (action >= kTrioWithSoloActionBase &&
             action < kAirplaneActionBase) {
    hand = SingleTrioCombHand(action);
  } else if (action >= kAirplaneWithSoloActionBase &&
             action < kBombActionBase) {
    hand = AirplaneCombHand(action);
  } else if (action == kRocketActionBase) {
    hand[kNumRanks - 1] = hand[kNumRanks - 2] = 1;
  } else {
    SpielFatalError("Non valid Action Ids");
  }
  return hand;
}

void SearchForLegalActions(std::vector<Action>* legal_actions,
                           absl::Span<const int> hand, int prev_action) {
  if (hand[kNumRanks - 2] && hand[kNumRanks - 1])
    legal_actions->push_back(kRocketActionBase);
  if (prev_action == kInvalidAction) {
    // search for all possible actions
    SearchSingleRankActions(legal_actions, hand, prev_action);
    SearchChainOnlyActions(legal_actions, hand, prev_action);
    SearchSingleTrioCombActions(legal_actions, hand, prev_action);
    SearchAirplaneCombActions(legal_actions, hand, prev_action);
  } else if (prev_action >= kBombActionBase &&
             prev_action < kRocketActionBase) {
    // if previous action is a bomb, then only higher bomb or rocket can be
    // played
    SearchSingleRankActions(legal_actions, hand, prev_action);
  } else {
    // check for bombs
    for (int rank = 0; rank < kNumRanks - 2; ++rank) {
      if (hand[rank] == kNumSuits) {
        std::array<int, kNumRanks> used_rank{};
        used_rank[rank] = kNumSuits;
        legal_actions->push_back(SingleRankHandToActionId(used_rank));
      }
    }

    // then search within each category
    if ((prev_action >= kPlayActionBase &&
         prev_action < kSoloChainActionBase) ||
        (prev_action >= kPairActionBase &&
         prev_action < kPairChainActionBase) ||
        (prev_action >= kTrioActionBase &&
         prev_action < kTrioWithSoloActionBase)) {
      SearchSingleRankActions(legal_actions, hand, prev_action);
    } else if ((prev_action >= kSoloChainActionBase &&
                prev_action < kPairActionBase) ||
               (prev_action >= kPairChainActionBase &&
                prev_action < kTrioActionBase) ||
               (prev_action >= kAirplaneActionBase &&
                prev_action < kAirplaneWithSoloActionBase)) {
      SearchChainOnlyActions(legal_actions, hand, prev_action);
    } else if (prev_action >= kTrioWithSoloActionBase &&
               prev_action < kAirplaneActionBase) {
      SearchSingleTrioCombActions(legal_actions, hand, prev_action);
    } else if (prev_action >= kAirplaneWithSoloActionBase &&
               prev_action < kBombActionBase) {
      SearchAirplaneCombActions(legal_actions, hand, prev_action);
    } else if (prev_action == kRocketActionBase) {
    } else {
      SpielFatalError("Previous actions invalid");
    }
  }
}

}  // namespace dou_dizhu
}  // namespace open_spiel
