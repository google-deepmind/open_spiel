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

#ifndef OPEN_SPIEL_GAMES_SPADES_SPADES_SCORING_H_
#define OPEN_SPIEL_GAMES_SPADES_SPADES_SCORING_H_

// Scoring for partnership spades.
// See https://dkmgames.com/CardSharp/Spades/SpadesHelp.php

#include <array>
#include <string>

namespace open_spiel {
namespace spades {

inline constexpr int kNumPlayers = 4;
constexpr char kPlayerChar[] = "NESW";

inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumPartnerships = 2;
inline constexpr int kNumBids = 14;  // Bids can be from 0 to 13 tricks
inline constexpr int kNumCards = kNumSuits * kNumCardsPerSuit;
inline constexpr int kNumCardsPerHand = kNumCards / kNumPlayers;
inline constexpr int kNumTricks = kNumCardsPerHand;
inline constexpr int kMaxScore = 230;  // Bid 13 (130) + Nil (100)

std::array<int, kNumPartnerships> Score(
    const std::array<int, kNumPlayers> contracts,
    const std::array<int, kNumPlayers> taken_tricks,
    const std::array<int, kNumPartnerships> current_scores);

// All possible contracts.
inline constexpr int kNumContracts = kNumBids * kNumPlayers;

constexpr std::array<int, kNumContracts> AllContracts() {
  std::array<int, kNumContracts> contracts = {};
  int bid = 0;
  for (int i = 0; i < kNumContracts; ++i) {
    contracts[i] = bid++;
    if (bid > kNumBids) {
      bid = 0;
    }
  }

  return contracts;
}
inline constexpr std::array<int, kNumContracts> kAllContracts = AllContracts();

}  // namespace spades
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SPADES_SPADES_SCORING_H_
