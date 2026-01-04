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

#ifndef OPEN_SPIEL_GAMES_DOU_DIZHU_DOU_DIZHU_UTILS_H_
#define OPEN_SPIEL_GAMES_DOU_DIZHU_DOU_DIZHU_UTILS_H_

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dou_dizhu {

enum class Phase { kDeal, kAuction, kPlay, kGameOver };

inline constexpr int kNumPlayers = 3;
inline constexpr int kNumCards = 54;

inline constexpr int kNumBids = 3;
inline constexpr int kNumCardsPerSuit = 13;

// player 0, 1 passes, 2 bids 1, then 0 passes, 1 bids 2, 2 passes, 0 bids 3, 1
// & 2 passes
inline constexpr int kMaxAuctionLength = 9;

// the maximum/minimum utility is achieved if the players play all 13 bombs
// alternatively and dizhu bid maximum bids
inline constexpr int kMaxUtility = kNumBids * 16384;
inline constexpr int kMinUtility = -kNumBids * 8192;

// 13 normal cards + 2 jokers
inline constexpr int kNumRanks = kNumCardsPerSuit + 2;

inline constexpr int kNumCardsLeftOver = 3;

inline constexpr int kNumSuits = 4;

// Observations are: the number of cards of each rank I current have
// Plus the number of cards of each rank that had been played by all players
// Plus the start player
// Plus the face up card
inline constexpr int kObservationTensorSize =
    2 * ((kNumRanks - 2) * (kNumSuits + 1) + 2 * 2) + kNumPlayers +
    kNumPlayers + kNumRanks;

inline constexpr int kDealingActionBase = kNumCards - kNumCardsLeftOver;

inline constexpr int kBiddingActionBase = 0;

inline constexpr int kPass = kBiddingActionBase;

inline constexpr int kPlayActionBase = kBiddingActionBase + 1 + kNumBids;

inline constexpr int kSoloChainMinLength = 5;
inline constexpr int kSoloChainActionBase = kPlayActionBase + 15;

inline constexpr int kPairActionBase = kSoloChainActionBase + 36;

inline constexpr int kPairChainMinLength = 3;
inline constexpr int kPairChainActionBase = kPairActionBase + 13;

inline constexpr int kTrioActionBase = kPairChainActionBase + 52;

inline constexpr int kTrioWithSoloActionBase = kTrioActionBase + 13;

inline constexpr int kTrioWithPairActionBase = kTrioWithSoloActionBase + 182;

inline constexpr int kAirplaneMinLength = 2;
inline constexpr int kAirplaneActionBase = kTrioWithPairActionBase + 156;

inline constexpr int kAirplaneWithSoloMinLength = 2;
inline constexpr int kAirplaneWithSoloActionBase = kAirplaneActionBase + 45;

inline constexpr int kAirplaneWithPairMinLength = 2;
inline constexpr int kAirplaneWithPairActionBase =
    kAirplaneWithSoloActionBase + 22588;

inline constexpr int kBombActionBase = kAirplaneWithPairActionBase + 2939;
inline constexpr int kRocketActionBase = kBombActionBase + 13;

inline constexpr int kNumKickersAirplaneSoloCombChainOfLengthTwo = 88;
inline constexpr int kNumKickersAirplaneSoloCombChainOfLengthThree = 330;
inline constexpr int kNumKickersAirplaneSoloCombChainOfLengthFour = 816;
inline constexpr int kNumKickersAirplaneSoloCombChainOfLengthFive = 1372;

inline constexpr int kNumKickersAirplanePairCombChainOfLengthTwo = 55;
inline constexpr int kNumKickersAirplanePairCombChainOfLengthThree = 120;
inline constexpr int kNumKickersAirplanePairCombChainOfLengthFour = 126;

constexpr char kRankChar[] = "3456789TJQKA2";
// only for dealing phase usages
constexpr char kSuitChar[] = "CDHS";

enum KickerType { kSolo = 1, kPair };

// single rank hand means hands consisting of only a single rank
// includes solo, pair, trio, bombs
struct SingleRankHandParams {
  int rank;
  int num_cards;
  SingleRankHandParams(int r, int n) : rank(r), num_cards(n) {}
  SingleRankHandParams() {}
};

// chain only hand means hands consisting of only consecutive ranks
// includes solo chain, pair chain and airplane
struct ChainOnlyHandParams {
  int chain_head;
  int num_cards_per_rank;
  int chain_length;
  ChainOnlyHandParams(int h, int n, int l)
      : chain_head(h), num_cards_per_rank(n), chain_length(l) {}
  ChainOnlyHandParams() {}
};

// shared by trio+solo, trio+pair, airplane+solo, airplane+pair
struct TrioCombParams {
  int chain_head;
  int chain_length;
  KickerType kicker_type;
  int kicker_id;
  TrioCombParams(int head, int length, KickerType k, int k_id)
      : chain_head(head),
        chain_length(length),
        kicker_type(k),
        kicker_id(k_id) {}
  TrioCombParams() {}
};

int CardToRank(int card);
std::string RankString(int rank);
std::string CardString(int card);
std::string FormatSingleHand(absl::Span<const int> hand);
std::string FormatAirplaneCombHand(int action);

SingleRankHandParams GetSingleRankHandParams(int action);
std::array<int, kNumRanks> SingleRankHand(int action);
int SingleRankHandToActionId(absl::Span<const int> hand);
void SearchSingleRankActions(std::vector<Action>* actions,
                             absl::Span<const int> hand, int prev_action);

ChainOnlyHandParams GetChainOnlyHandParams(int action);
std::array<int, kNumRanks> ChainOnlyHand(int action);
int ChainOnlyHandToActionId(absl::Span<const int> hand);
void SearchChainOnlyActions(std::vector<Action>* actions,
                            absl::Span<const int> hand, int prev_action);

TrioCombParams GetSingleTrioCombParams(int action);
std::array<int, kNumRanks> SingleTrioCombHand(int action);
int SingleTrioCombHandToActionId(absl::Span<const int> hand);
void SearchSingleTrioCombActions(std::vector<Action>* actions,
                                 absl::Span<const int> hand, int prev_action);

TrioCombParams GetAirplaneCombParams(int action);
std::array<int, kNumRanks> AirplaneCombHand(int action);
int AirplaneCombHandToActionId(absl::Span<const int> hand, int chain_head,
                               KickerType kicker_type);
void SearchAirplaneCombActions(std::vector<Action>* actions,
                               absl::Span<const int> hand, int prev_action);

std::array<int, kNumRanks> ActionToHand(int action);
void SearchForLegalActions(std::vector<Action>* legal_actions,
                           absl::Span<const int> hand, int prev_action);

}  // namespace dou_dizhu
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOU_DIZHU_DOU_DIZHU_UTILS_H_
