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

#ifndef OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_
#define OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_

// Scoring for (duplicate) contract bridge.
// See Law 77 of the Laws of Bridge, 2017:
// http://www.worldbridge.org/wp-content/uploads/2017/03/2017LawsofDuplicateBridge-paginated.pdf

#include <array>
#include <string>

namespace open_spiel {
namespace bridge {

enum Denomination { kClubs = 0, kDiamonds, kHearts, kSpades, kNoTrump };
inline constexpr int kNumDenominations = 5;
constexpr char kDenominationChar[] = "CDHSN";

enum DoubleStatus { kUndoubled = 1, kDoubled = 2, kRedoubled = 4 };
inline constexpr int kNumDoubleStates = 3;

inline constexpr int kNumPlayers = 4;
constexpr char kPlayerChar[] = "NESW";

inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumPartnerships = 2;
inline constexpr int kNumBidLevels = 7;   // Bids can be from 7 to 13 tricks.
inline constexpr int kNumOtherCalls = 3;  // Pass, Double, Redouble
inline constexpr int kNumVulnerabilities = 2;  // Vulnerable or non-vulnerable.
inline constexpr int kNumBids = kNumBidLevels * kNumDenominations;
inline constexpr int kNumCalls = kNumBids + kNumOtherCalls;
inline constexpr int kNumCards = kNumSuits * kNumCardsPerSuit;
inline constexpr int kNumCardsPerHand = kNumCards / kNumPlayers;
inline constexpr int kNumTricks = kNumCardsPerHand;
inline constexpr int kMaxScore = 7600;  // See http://www.rpbridge.net/2y66.htm

struct Contract {
  int level = 0;
  Denomination trumps = kNoTrump;
  DoubleStatus double_status = kUndoubled;
  int declarer = -1;

  std::string ToString() const;
  int Index() const;
};

int Score(Contract contract, int declarer_tricks, bool is_vulnerable);

// All possible contracts.
inline constexpr int kNumContracts =
    kNumBids * kNumPlayers * kNumDoubleStates + 1;
constexpr std::array<Contract, kNumContracts> AllContracts() {
  std::array<Contract, kNumContracts> contracts;
  int i = 0;
  contracts[i++] = Contract();
  for (int level : {1, 2, 3, 4, 5, 6, 7}) {
    for (Denomination trumps :
         {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          contracts[i++] = Contract{level, trumps, double_status, declarer};
        }
      }
    }
  }
  return contracts;
}
inline constexpr std::array<Contract, kNumContracts> kAllContracts =
    AllContracts();

}  // namespace bridge
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_
